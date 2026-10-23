"""虛擬考生＋埋錯作答：驗證 26B／31B 兩位裁判是否可靠

設計：
A. 選題：每科 QUESTIONS_PER_SUBJECT 題（預設 5 題 × 9 科 = 45 題），
   優先選前次評測中兩模型分數不同或未滿分的題目。
B. 虛擬考生作答（呼叫 API）：10 種不同特性的考生，每題各作答一次。
   產生者輪流分配：偶數題由 26B 產生、奇數題由 31B 產生，兩邊數量相同，
   用來量測裁判是否偏好「自己家」產生的作答（差異中的差異）。
C. 埋錯作答（不呼叫 API）：由程式改寫參考答案，錯誤類型已知、可當標準：
   E0 完全正確、E1 最終數值錯誤、E2 漏答後段小題、E3 答非所問、
   E4 刪除依據、E5 只寫答案不寫推論、E6 加入虛構條文
D. 閱卷：26B、31B 各盲評一次全部作答（不告知作答者），沿用 ftrc_eval_v2.py 的評分標準。
E. 分析：抓錯率、考生程度鑑別力（Spearman）、兩裁判一致性（加權 kappa）、
   同家族偏好（差異中的差異）。

使用方式（與 ftrc_eval_v2.py 放同一資料夾，在 Spyder 按 F5）：
  無參數          ：完整執行（產生作答 → 兩裁判閱卷 → 報告），可中斷續跑
  --dry-run       ：只選題並產生埋錯作答，輸出檢查用檔案，不呼叫 API
  --report-only   ：只用已存結果重新產生報告
"""
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftrc_eval_v2 as base  # noqa: E402

QUESTIONS_PER_SUBJECT = 5
SEED = 42
GEN_TEMPERATURE = 0.7
GENERATORS = ["gemma-4-26B-A4B-it", "gemma-4-31B-it"]
JUDGES = ["gemma-4-26B-A4B-it", "gemma-4-31B-it"]
SHORT = {"gemma-4-26B-A4B-it": "26B", "gemma-4-31B-it": "31B"}

OUT = os.path.join(os.path.dirname(base.OUT_DIR), "virtual_students")
ITEMS_CSV = os.path.join(OUT, "vs_selected_questions.csv")
ANS_JSONL = os.path.join(OUT, "vs_answers.jsonl")
JUDGE_JSONL = os.path.join(OUT, "vs_judgments.jsonl")
SCORES_CSV = os.path.join(OUT, "vs_scores.csv")
SUMMARY_TXT = os.path.join(OUT, "vs_summary.txt")
INJECT_PREVIEW = os.path.join(OUT, "vs_injected_preview.txt")

# 虛擬考生：level 為預期程度（3 高、2 中、1 低），只用來檢查裁判的鑑別力
PERSONAS = OrderedDict([
    ("S01", (3, "完整推導型：觀念正確，依序回答每一小題，列出公式、代入數字並交代結論。")),
    ("S02", (3, "精簡型：觀念正確，只寫必要的公式、計算與結論，不加背景介紹。")),
    ("S03", (3, "替代解法型：觀念正確，適合時改用另一種有效的方法解題，但不可為了不同而用錯方法。")),
    ("S04", (2, "白話型：用日常語言說明機制，盡量少用專有名詞，但保留題目要求的計算。")),
    ("S05", (1, "概念懂但計算不熟：理解核心概念，但計算常出錯；不確定的計算可以停在已知步驟，不要編造結果。")),
    ("S06", (2, "計算強但說明弱：擅長代入公式算出數字，但很少說明理由，也不引用法規或準則。")),
    ("S07", (1, "初學者：剛學這個科目，只回答自己理解的部分，不會的部分直接說不知道。")),
    ("S08", (1, "時間不足：只來得及認真回答前面的小題，後面的小題來不及寫完，不要補寫。")),
    ("S09", (3, "重視假設型：先辨識題目的假設與條件，遇到條件不足時用條件式回答，再給主要結論。")),
    ("S10", (2, "敘述冗長型：習慣舉例和長篇說明，內容正確與否依自己的理解，不刻意寫錯。")),
])
PERSONA_SYSTEM = ("你正在扮演一位參加金融證照或國家考試的考生。考生特性：{desc}\n"
                  "請完全以這位考生的身分與程度作答。不要提到你在扮演角色，不要說自己是 AI，"
                  "也不要評論自己的程度。")

# 埋錯作答：expected 為預期 FTRC 結果，detect 為判定裁判「抓到」的條件
INJECT_TYPES = OrderedDict([
    ("E0", "完全正確（參考答案原文），預期 ≥ 4.5 分"),
    ("E1", "最終數值錯誤，預期觸發 C1（≤ 1 分）"),
    ("E2", "漏答後段小題，預期觸發 C4（≤ 3 分）"),
    ("E3", "答非所問（同科別題的答案），預期 ≤ 1 分（C3 或 C1）"),
    ("E4", "刪除依據（法規、準則、適用原理），預期 P5 ≤ 0.5"),
    ("E5", "只寫最終答案、不寫推論，預期 P4 ≤ 0.5"),
    ("E6", "加入虛構條文，預期觸發 C2（≤ 2 分）"),
])
FAKE_CITATION = {
    "ACC": "依 IFRS 27 第 45 段之規定，",
    "FSA": "依 IFRS 27 第 45 段之規定，",
    "CRD": "依銀行法第 199 條之規定，",
    "ESG": "依《永續金融揭露準則》第 88 條之規定，",
    "CFIN": "依 Hendricks–Lau 資本結構定理，",
    "ECO": "依 Hendricks–Lau 均衡定理，",
    "STA": "依 Hendricks–Lau 抽樣定理，",
    "INS": "依保險法第 248 條之規定，",
    "TRD": "依 UCP600 第 47 條之規定，",
}


# ===== 選題 =====
def select_items():
    items = base.sample_items(base.load_items())
    judg = base.load_done(base.JUDGMENTS_JSONL, lambda r: (r["ID"], r["model"])) if os.path.exists(base.JUDGMENTS_JSONL) else {}
    rng = random.Random(SEED)
    by_subj = OrderedDict()
    for it in items:
        by_subj.setdefault(it["ID"].split("-")[0], []).append(it)
    picked = []
    for subj, lst in by_subj.items():
        def key(it):
            s = [judg.get((it["ID"], m), {}).get("final_score") for m in base.TESTEE_MODELS]
            if None in s:
                return (2, rng.random())
            return (0 if s[0] != s[1] else (1 if min(s) < 5 else 2), -abs(s[0] - s[1]), rng.random())
        picked += sorted(sorted(lst, key=key)[:QUESTIONS_PER_SUBJECT], key=lambda x: x["ID"])
    return picked


# ===== 埋錯作答 =====
def split_reference(ref):
    if "【最終答案】" in ref:
        body, final = ref.split("【最終答案】", 1)
        return body.strip(), final.strip()
    return ref.strip(), ""


NUM = re.compile(r"(?<![A-Za-z\d])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")


def perturb_numbers(text):
    def rep(m):
        s = m.group(0)
        v = float(s.replace(",", ""))
        if v < 10:
            return s
        nv = v * 1.2 + 1
        if m.group(2):
            out = f"{nv:,.{len(m.group(2)) - 1}f}" if "," in m.group(1) else f"{nv:.{len(m.group(2)) - 1}f}"
        else:
            out = f"{round(nv):,}" if "," in s else str(round(nv))
        return out
    new = NUM.sub(rep, text)
    return new if new != text else None


SUBPART = re.compile(r"(?m)^\s*(?:[⑴⑵⑶⑷⑸⑹]|[（(][一二三四五六][）)]|[㈠㈡㈢㈣㈤]|\([1-6]\)|【[(（]?[一二三四五六1-6][)）]?)")


def is_multipart(q):
    return len(re.findall(r"[⑴⑵⑶⑷⑸]|[（(][一二三四五][）)]|[㈠㈡㈢㈣]", q)) >= 2


def drop_later_parts(body):
    pos = [m.start() for m in SUBPART.finditer(body)]
    if len(pos) >= 2:
        cut = pos[len(pos) // 2] if len(pos) > 2 else pos[1]
        return body[:cut].rstrip() + "\n\n（時間不足，其餘小題未作答。）"
    lines = body.split("\n")
    if len(lines) < 4:
        return None
    return "\n".join(lines[: max(2, int(len(lines) * 0.5))]).rstrip() + "\n\n（時間不足，其餘小題未作答。）"


CITE = re.compile(r"《[^》]{1,30}》|第\s*\d+(?:-\d+)?\s*條(?:第\s*\d+\s*項)?|IFRS\s*\d+|IAS\s*\d+|IFRIC\s*\d+|UCP\s*600(?:\s*第\s*\d+\s*條)?|ISBP\s*\d*")


def strip_basis(body):
    out = re.sub(r"【(?:適用原理|依據|法規依據|準則依據|相關規定)】[^【]*", "", body)
    out = CITE.sub("", out)
    return out if out.strip() and out != body else None


def make_injected(items):
    by_subj = defaultdict(list)
    for it in items:
        by_subj[it["ID"].split("-")[0]].append(it)
    recs = []
    for it in items:
        subj = it["ID"].split("-")[0]
        ref = base.reference_text(it)
        body, final = split_reference(ref)
        variants = {"E0": ref}
        if final:
            f2 = perturb_numbers(final)
            if f2:
                variants["E1"] = body + "\n\n【最終答案】\n" + f2
        if is_multipart(it["題目"]):
            d = drop_later_parts(body)
            if d:
                variants["E2"] = d
        peers = [p for p in by_subj[subj] if p["ID"] != it["ID"]]
        if peers:
            variants["E3"] = base.reference_text(peers[0])
        s = strip_basis(body)
        if s:
            variants["E4"] = s + ("\n\n【最終答案】\n" + final if final else "")
        if final:
            variants["E5"] = "【最終答案】\n" + final
        lines = body.split("\n")
        k = 1 if len(lines) > 1 else 0
        lines.insert(k, FAKE_CITATION.get(subj, "依 Hendricks–Lau 定理，") + "本題應依此處理。")
        variants["E6"] = "\n".join(lines) + ("\n\n【最終答案】\n" + final if final else "")
        for t, txt in variants.items():
            recs.append({"vid": f"{it['ID']}|{t}", "ID": it["ID"], "kind": "inject", "type": t,
                         "generator": "program", "answer": txt})
    return recs


# ===== 閱卷與作答 =====
def generate_personas(client, items, answers):
    for qi, it in enumerate(items):
        gen = GENERATORS[qi % 2]
        for sid, (lvl, desc) in PERSONAS.items():
            vid = f"{it['ID']}|{sid}"
            if vid in answers:
                continue
            print(f"[作答] {vid}（{SHORT[gen]} 產生）...", end="", flush=True)
            try:
                txt = base.chat(client, gen, [{"role": "system", "content": PERSONA_SYSTEM.format(desc=desc)},
                                              {"role": "user", "content": f"科目：{it['科目']}\n\n{it['題目']}"}],
                                temperature=GEN_TEMPERATURE)
            except Exception as e:
                print(f" 失敗：{e}")
                continue
            rec = {"vid": vid, "ID": it["ID"], "kind": "persona", "type": sid, "level": lvl,
                   "generator": gen, "answer": txt}
            base.append_jsonl(ANS_JSONL, rec)
            answers[vid] = rec
            print(" 完成")
            time.sleep(base.SLEEP_SEC)


def judge_all(client, items, answers, judgments):
    by_id = {it["ID"]: it for it in items}
    todo = [(a, j) for a in answers.values() for j in JUDGES if (a["vid"], j) not in judgments]
    for n, (a, j) in enumerate(todo, 1):
        print(f"[閱卷 {n}/{len(todo)}] {a['vid']}（{SHORT[j]} 閱卷）...", end="", flush=True)
        base.MODEL_JUDGE = j
        try:
            v, pts, caps, raw = base.judge(client, by_id[a["ID"]], a["answer"])
        except Exception as e:
            print(f" 失敗：{e}")
            continue
        s, caps, raw_sum, final = base.score(pts, caps)
        rec = {"vid": a["vid"], "judge": j, **s, "raw_score": raw_sum, "final_score": final,
               "caps": ",".join(caps), "cap_reasons": v.get("cap_reasons", ""),
               "evidence": {p: pts[p].get("evidence", "") for p in base.POINTS}}
        base.append_jsonl(JUDGE_JSONL, rec)
        judgments[(a["vid"], j)] = rec
        print(f" {final}／5 {rec['caps']}")
        time.sleep(base.SLEEP_SEC)


# ===== 統計 =====
def mean(v):
    return sum(v) / len(v) if v else float("nan")


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def weighted_kappa(x, y):
    k = 11
    obs = [[0] * k for _ in range(k)]
    for a, b in zip(x, y):
        obs[int(round(a * 2))][int(round(b * 2))] += 1
    n = len(x)
    ra = [sum(r) for r in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = sum((i - j) ** 2 * obs[i][j] for i in range(k) for j in range(k))
    den = sum((i - j) ** 2 * ra[i] * cb[j] / n for i in range(k) for j in range(k))
    return 1 - num / den if den else float("nan")


def detected(t, r):
    caps = r["caps"].split(",") if r["caps"] else []
    f = r["final_score"]
    return {"E0": f >= 4.5, "E1": f <= 1 or "C1" in caps, "E2": f <= 3 or "C4" in caps,
            "E3": f <= 1, "E4": r["P5"] <= 0.5, "E5": r["P4"] <= 0.5, "E6": f <= 2 or "C2" in caps}[t]


def report(items, answers, judgments):
    L = [f"選題 {len(items)} 題；作答 {len(answers)} 份（虛擬考生 "
         f"{sum(a['kind'] == 'persona' for a in answers.values())}、埋錯 {sum(a['kind'] == 'inject' for a in answers.values())}）；"
         f"閱卷 {len(judgments)} 筆", ""]

    L.append("【1】埋錯作答：裁判抓錯率（錯誤類型已知）")
    L.append("類型\t說明\t題數\t26B 抓到\t31B 抓到")
    for t, desc in INJECT_TYPES.items():
        row = []
        for j in JUDGES:
            rs = [judgments[(a["vid"], j)] for a in answers.values()
                  if a["kind"] == "inject" and a["type"] == t and (a["vid"], j) in judgments]
            row.append((sum(detected(t, r) for r in rs), len(rs)))
        n = max(c[1] for c in row)
        L.append(f"{t}\t{desc}\t{n}\t" + "\t".join(f"{d}/{c}（{d / c:.0%}）" if c else "-" for d, c in row))
    L.append("")

    pa = [a for a in answers.values() if a["kind"] == "persona"]
    L.append("【2】虛擬考生：各類考生平均分（兩位裁判）")
    L.append("考生\t預期程度\t26B 閱卷\t31B 閱卷\t說明")
    for sid, (lvl, desc) in PERSONAS.items():
        vals = [[judgments[(a["vid"], j)]["final_score"] for a in pa if a["type"] == sid and (a["vid"], j) in judgments]
                for j in JUDGES]
        L.append(f"{sid}\t{'高中低'[3 - lvl]}\t{mean(vals[0]):.2f}\t{mean(vals[1]):.2f}\t{desc.split('：')[0]}")
    for j in JUDGES:
        xs = [(a["level"], judgments[(a["vid"], j)]["final_score"]) for a in pa if (a["vid"], j) in judgments]
        if xs:
            L.append(f"鑑別力 {SHORT[j]} 閱卷：預期程度與分數的 Spearman = {spearman([x for x, _ in xs], [y for _, y in xs]):.3f}")
    L.append("")

    L.append("【3】兩位裁判一致性（同一份作答）")
    for kind, name in (("persona", "虛擬考生"), ("inject", "埋錯作答"), (None, "全部")):
        both = [a["vid"] for a in answers.values() if (kind is None or a["kind"] == kind)
                and all((a["vid"], j) in judgments for j in JUDGES)]
        if not both:
            continue
        x = [judgments[(v, JUDGES[0])]["final_score"] for v in both]
        y = [judgments[(v, JUDGES[1])]["final_score"] for v in both]
        L.append(f"{name}（{len(both)} 份）：完全一致 {mean([p == q for p, q in zip(x, y)]):.1%}，差距 ≤ 0.5 "
                 f"{mean([abs(p - q) <= 0.5 for p, q in zip(x, y)]):.1%}，加權 kappa {weighted_kappa(x, y):.3f}，"
                 f"相關係數 {pearson(x, y):.3f}，平均（26B／31B 閱卷）{mean(x):.2f}／{mean(y):.2f}")
    L.append("")

    L.append("【4】同家族偏好：產生者 × 裁判（虛擬考生作答）")
    cell = {}
    for g in GENERATORS:
        for j in JUDGES:
            cell[(g, j)] = mean([judgments[(a["vid"], j)]["final_score"] for a in pa
                                 if a["generator"] == g and (a["vid"], j) in judgments])
    L.append("產生者＼裁判\t26B 閱卷\t31B 閱卷")
    for g in GENERATORS:
        L.append(f"{SHORT[g]} 產生\t{cell[(g, JUDGES[0])]:.3f}\t{cell[(g, JUDGES[1])]:.3f}")
    g26, g31 = GENERATORS
    j26, j31 = JUDGES
    did = (cell[(g31, j31)] - cell[(g26, j31)]) - (cell[(g31, j26)] - cell[(g26, j26)])
    L.append(f"差異中的差異 = [31B 閱卷：31B 產生 - 26B 產生] - [26B 閱卷：31B 產生 - 26B 產生] = {did:+.3f}")
    L.append("  ≈ 0：沒有同家族偏好；> 0：兩位裁判都偏好自己產生的作答；< 0：反向。")
    L.append("  （產生者輪流分配題目，兩邊題目不同；題目難度差異在兩位裁判之間會抵銷。）")

    with open(SCORES_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["vid", "ID", "種類", "類型", "產生者"] + [f"{SHORT[j]}閱卷_{k}" for j in JUDGES
                                                         for k in ("總分", "P1", "P2", "P3", "P4", "P5", "封頂")])
        for a in answers.values():
            row = [a["vid"], a["ID"], a["kind"], a["type"], SHORT.get(a["generator"], a["generator"])]
            for j in JUDGES:
                r = judgments.get((a["vid"], j))
                row += [r["final_score"], r["P1"], r["P2"], r["P3"], r["P4"], r["P5"], r["caps"]] if r else [""] * 7
            w.writerow(row)
    return "\n".join(L)


def main():
    os.makedirs(OUT, exist_ok=True)
    items = select_items()
    with open(ITEMS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID", "科目", "虛擬考生產生者"])
        for qi, it in enumerate(items):
            w.writerow([it["ID"], it["科目"], SHORT[GENERATORS[qi % 2]]])
    answers = base.load_done(ANS_JSONL, lambda r: r["vid"]) if os.path.exists(ANS_JSONL) else {}
    for rec in make_injected(items):
        if rec["vid"] not in answers:
            answers[rec["vid"]] = rec
            base.append_jsonl(ANS_JSONL, rec)
    inj = [a for a in answers.values() if a["kind"] == "inject"]
    print(f"選題 {len(items)} 題；埋錯作答 {len(inj)} 份：" +
          "，".join(f"{t} {sum(a['type'] == t for a in inj)}" for t in INJECT_TYPES))

    if "--dry-run" in sys.argv:
        with open(INJECT_PREVIEW, "w", encoding="utf-8") as f:
            for it in items[:3]:
                for t in INJECT_TYPES:
                    a = answers.get(f"{it['ID']}|{t}")
                    if a:
                        f.write(f"=============== {a['vid']}：{INJECT_TYPES[t]}\n{a['answer']}\n\n")
        print(f"已輸出 {ITEMS_CSV} 與埋錯作答範例 {INJECT_PREVIEW}（未呼叫 API）")
        return

    judgments = {}
    if os.path.exists(JUDGE_JSONL):
        judgments = base.load_done(JUDGE_JSONL, lambda r: (r["vid"], r["judge"]))
    if "--report-only" not in sys.argv:
        from openai import OpenAI
        client = OpenAI(base_url=base.API_BASE_URL, api_key=base.API_KEY)
        generate_personas(client, items, answers)
        judge_all(client, items, answers, judgments)
    rep = report(items, answers, judgments)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(rep + "\n")
    print("\n" + "=" * 50 + "\n" + rep)


if __name__ == "__main__":
    main()
