"""非選擇題 FTRC 五層次評測 v2（依新題庫重跑）

與第一版的差異：
1. 題目與參考答案改讀「分科完整資料_JSON」（362 題非選擇，已排除缺圖表的 4 題），
   參考答案用補完後的 CoT＋最終答案，不再用原始草稿。
2. 每科抽 SAMPLES_PER_SUBJECT 題（預設 10 題，共 90 題），兩個模型都作答。
3. 31B 盲評：閱卷時不告知作答模型；每題 P1–P5 依據與封頂原因全部存檔。
4. P5 改為依題目性質認定依據（法規條號、準則編號、公式、理論），統計、經濟題也能得分。
5. 支援中斷續跑：已完成的作答與閱卷會略過。
6. 兩模型比較用 Wilcoxon 符號等級檢定（成對、非常態分數）。

使用方式（PowerShell）：
  $env:APMIC_API_KEY = "你的金鑰"
  python ftrc_eval_v2.py            # 正式執行
  python ftrc_eval_v2.py --dry-run  # 只抽題並輸出題目清單，不呼叫 API
"""
import csv
import glob
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict

# ===== 設定 =====
API_BASE_URL = "https://demo-api.dgx-coolify.apmic.ai/v1"
API_KEY = os.environ.get("APMIC_API_KEY", "EMPTY")  # 金鑰由環境變數提供，不寫在程式裡
TESTEE_MODELS = ["gemma-4-26B-A4B-it", "gemma-4-31B-it"]  # 受測模型
MODEL_JUDGE = "gemma-4-31B-it"  # 閱卷裁判（注意：31B 同時受測，可能有自我偏好）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "分科完整資料_JSON")
OUT_DIR = os.path.join(ROOT, "output", "ftrc_v2")

SAMPLES_PER_SUBJECT = None  # 每科抽題數；None = 全部 362 題
SEED = 42
TESTEE_TEMPERATURE = 0.1
MAX_RETRIES = 3
SLEEP_SEC = 0.5

ANSWERS_JSONL = os.path.join(OUT_DIR, "answers.jsonl")
JUDGMENTS_JSONL = os.path.join(OUT_DIR, "judgments.jsonl")

JUDGE_SYSTEM = """你是金融領域的閱卷委員。你的任務是依照給定的評分標準，對照參考答案為受測回答評分。
務必遵守：只依評分標準判定；每個評分點都要引用受測回答中的具體文字作為依據；
參考答案只用來核對正確性，受測回答用不同但正確的方法也應給分。只輸出 JSON。"""

JUDGE_TEMPLATE = """# 評分標準：FTRC 五層次評分
## 評分點（P1～P5 只能給 0、0.5、1.0）
- P1 核心結論：1.0 與參考答案一致且明確／0.5 方向正確但有保留或部分小題錯誤／0 錯誤或未給結論
- P2 關鍵細節：1.0 關鍵數值、條件、分錄全部正確／0.5 部分正確／0 未提及或錯誤
- P3 情境辨識：1.0 正確辨識題目所屬制度、準則或模型情境／0.5 隱含理解／0 誤判情境
- P4 適用推論：1.0 推論鏈完整（公式→代入→結果，或原則→適用→結論）／0.5 有推論但跳步／0 無推論
- P5 依據排錯：依題目性質認定依據——法規題看法規條號或名稱，會計題看準則（IFRS、IAS 等）編號或名稱，
  計量題（統計、經濟、財務）看公式、定理或模型名稱。1.0 引用具體且正確的依據／0.5 僅籠統提及／0 無依據或依據錯誤

## 封頂規則（triggered_caps）
- C1：P1 得 0 分，總分最高 1.0
- C2：出現幻覺（虛構法條、準則或數據），總分最高 2.0
- C3：答非所問，總分最高 0.0
- C4：多小題中有任一小題未作答，總分最高 3.0

## 題目
{question}

## 參考答案
{reference}

## 受測回答
{answer}

請嚴格回傳以下 JSON 格式：
{{
  "points": [
    {{"point_id": "P1", "score": 1.0, "evidence": "引用受測回答原文並說明"}},
    {{"point_id": "P2", "score": 1.0, "evidence": "..."}},
    {{"point_id": "P3", "score": 1.0, "evidence": "..."}},
    {{"point_id": "P4", "score": 1.0, "evidence": "..."}},
    {{"point_id": "P5", "score": 1.0, "evidence": "..."}}
  ],
  "triggered_caps": [],
  "cap_reasons": ""
}}"""

CAP_LIMITS = {"C1": 1.0, "C2": 2.0, "C3": 0.0, "C4": 3.0}
POINTS = ["P1", "P2", "P3", "P4", "P5"]


# ===== 資料 =====
def load_items():
    items = []
    for fn in sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))):
        for it in json.load(open(fn, encoding="utf-8")):
            if it.get("題型") == "非選擇":
                items.append(it)
    return items


def sample_items(items):
    by_subj = OrderedDict()
    for it in items:
        by_subj.setdefault(it["ID"].split("-")[0], []).append(it)
    rng = random.Random(SEED)
    picked = []
    for subj, lst in by_subj.items():
        lst = sorted(lst, key=lambda x: x["ID"])
        n = len(lst) if SAMPLES_PER_SUBJECT is None else min(SAMPLES_PER_SUBJECT, len(lst))
        picked += sorted(rng.sample(lst, n), key=lambda x: x["ID"])
    return picked


def question_text(item):
    return item["SFT"]["messages"][1]["content"]


def reference_text(item):
    return item["SFT"]["messages"][2]["content"]


def load_done(path, keyf):
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                done[keyf(rec)] = rec
    return done


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ===== API =====
def chat(client, model, messages, **kw):
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = client.chat.completions.create(model=model, messages=messages, **kw)
            return r.choices[0].message.content
        except Exception as e:  # 網路或伺服器錯誤時重試
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def validate_verdict(v):
    pts = {p.get("point_id"): p for p in v.get("points", [])}
    for pid in POINTS:
        if pid not in pts or float(pts[pid].get("score", -1)) not in (0.0, 0.5, 1.0):
            raise ValueError(f"{pid} 缺漏或分數不在 0／0.5／1.0")
    caps = [c for c in v.get("triggered_caps", []) if c in CAP_LIMITS]
    return pts, caps


def judge(client, item, answer):
    prompt = JUDGE_TEMPLATE.format(question=question_text(item), reference=reference_text(item), answer=answer)
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            raw = chat(client, MODEL_JUDGE,
                       [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
                       temperature=0.0, response_format={"type": "json_object"})
            v = json.loads(raw)
            pts, caps = validate_verdict(v)
            return v, pts, caps, raw
        except Exception as e:  # JSON 格式不符時重試
            last = e
            time.sleep(1)
    raise last


def score(pts, caps):
    s = {pid: float(pts[pid]["score"]) for pid in POINTS}
    caps = list(caps)
    if s["P1"] == 0 and "C1" not in caps:  # 規則要求：P1 為 0 必觸發 C1
        caps.append("C1")
    raw = sum(s.values())
    final = min([raw] + [CAP_LIMITS[c] for c in caps])
    return s, caps, raw, final


# ===== 統計 =====
def wilcoxon_signed_rank(x, y):
    """成對 Wilcoxon 符號等級檢定（雙尾）。有 scipy 用 scipy，否則用常態近似。"""
    pairs = [(a - b) for a, b in zip(x, y) if a != b]
    if not pairs:
        return float("nan"), 1.0, 0
    try:
        from scipy import stats
        r = stats.wilcoxon(x, y, zero_method="wilcox")
        return float(r.statistic), float(r.pvalue), len(pairs)
    except Exception:
        absd = sorted((abs(d), i) for i, d in enumerate(pairs))
        ranks = [0.0] * len(pairs)
        i = 0
        while i < len(absd):
            j = i
            while j + 1 < len(absd) and absd[j + 1][0] == absd[i][0]:
                j += 1
            for k in range(i, j + 1):
                ranks[absd[k][1]] = (i + j) / 2 + 1
            i = j + 1
        w_plus = sum(r for r, d in zip(ranks, pairs) if d > 0)
        n = len(pairs)
        mu = n * (n + 1) / 4
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (w_plus - mu) / sd
        return w_plus, math.erfc(abs(z) / math.sqrt(2)), n


def median(v):
    v = sorted(v)
    n = len(v)
    return (v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2) if n else float("nan")


def summarize(picked, results):
    by_model = defaultdict(dict)
    for r in results:
        by_model[r["model"]][r["ID"]] = r
    subj_of = {it["ID"]: it["科目"] for it in picked}
    lines = [f"題數：{len(picked)}（每科 {SAMPLES_PER_SUBJECT} 題，seed {SEED}）",
             f"閱卷裁判：{MODEL_JUDGE}（盲評；31B 同時受測，結果可能受自我偏好影響）", ""]
    for m in TESTEE_MODELS:
        rs = list(by_model[m].values())
        if not rs:
            continue
        fin = [r["final_score"] for r in rs]
        lines.append(f"[{m}] 完成 {len(rs)} 題，平均 {sum(fin) / len(fin):.2f}／5，中位數 {median(fin):.2f}")
        lines.append("  " + "  ".join(f"{p} {sum(r[p] for r in rs) / len(rs):.2f}" for p in POINTS))
        caps = defaultdict(int)
        for r in rs:
            for c in r["caps"].split(","):
                if c:
                    caps[c] += 1
        lines.append(f"  觸發封頂：{dict(caps) if caps else '無'}")
        per = defaultdict(list)
        for r in rs:
            per[subj_of[r["ID"]]].append(r["final_score"])
        for s, v in per.items():
            lines.append(f"    {s}：{sum(v) / len(v):.2f}（{len(v)} 題）")
        lines.append("")
    if len(TESTEE_MODELS) == 2:
        a, b = TESTEE_MODELS
        common = sorted(set(by_model[a]) & set(by_model[b]))
        if common:
            x = [by_model[a][i]["final_score"] for i in common]
            y = [by_model[b][i]["final_score"] for i in common]
            w, p, n = wilcoxon_signed_rank(x, y)
            lines.append(f"Wilcoxon 符號等級檢定（{a} vs {b}，共同完成 {len(common)} 題，非同分 {n} 題）："
                         f"W = {w:.1f}，p = {p:.4f}（{'顯著' if p < 0.05 else '不顯著'}，α = 0.05）")
    return "\n".join(lines)


# ===== 主程式 =====
def main():
    dry = "--dry-run" in sys.argv
    os.makedirs(OUT_DIR, exist_ok=True)
    items = load_items()
    picked = sample_items(items)
    with open(os.path.join(OUT_DIR, "sampled_questions.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID", "科目", "題目"])
        for it in picked:
            w.writerow([it["ID"], it["科目"], it["題目"]])
    print(f"非選擇題共 {len(items)} 題，抽出 {len(picked)} 題；清單已存 sampled_questions.csv")
    if dry:
        return

    from openai import OpenAI
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    answers = load_done(ANSWERS_JSONL, lambda r: (r["ID"], r["model"]))
    judgments = load_done(JUDGMENTS_JSONL, lambda r: (r["ID"], r["model"]))
    total = len(picked) * len(TESTEE_MODELS)
    step = 0

    for it in picked:
        for m in TESTEE_MODELS:
            step += 1
            key = (it["ID"], m)
            print(f"[{step}/{total}] {it['ID']} {m} ...", end="", flush=True)
            if key not in answers:
                try:
                    ans = chat(client, m, [{"role": "user", "content": question_text(it)}],
                               temperature=TESTEE_TEMPERATURE)
                except Exception as e:
                    print(f" 作答失敗：{e}")
                    continue
                answers[key] = {"ID": it["ID"], "model": m, "answer": ans}
                append_jsonl(ANSWERS_JSONL, answers[key])
                time.sleep(SLEEP_SEC)
            if key not in judgments:
                try:
                    v, pts, caps, raw = judge(client, it, answers[key]["answer"])
                except Exception as e:
                    print(f" 閱卷失敗：{e}")
                    continue
                s, caps, raw_sum, final = score(pts, caps)
                judgments[key] = {"ID": it["ID"], "model": m, **s, "raw_score": raw_sum, "final_score": final,
                                  "caps": ",".join(caps), "cap_reasons": v.get("cap_reasons", ""),
                                  "evidence": {pid: pts[pid].get("evidence", "") for pid in POINTS},
                                  "judge_raw": raw}
                append_jsonl(JUDGMENTS_JSONL, judgments[key])
                time.sleep(SLEEP_SEC)
            print(f" 得分 {judgments[key]['final_score']}／5")

    results = list(judgments.values())
    cols = ["ID", "科目", "model", "final_score", "raw_score"] + POINTS + ["caps", "cap_reasons"] + \
           [f"{p}_evidence" for p in POINTS] + ["answer"]
    subj_of = {it["ID"]: it["科目"] for it in picked}
    with open(os.path.join(OUT_DIR, "ftrc_v2_results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda r: (r["ID"], r["model"])):
            row = dict(r, 科目=subj_of.get(r["ID"], ""), answer=answers[(r["ID"], r["model"])]["answer"])
            row.update({f"{p}_evidence": r["evidence"].get(p, "") for p in POINTS})
            w.writerow(row)
    report = summarize(picked, results)
    missing = total - len(results)
    if missing:
        report += f"\n\n尚有 {missing} 筆未完成（作答或閱卷失敗），重新執行會自動續跑。"
    with open(os.path.join(OUT_DIR, "ftrc_v2_summary.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n" + "=" * 50 + "\n" + report)


if __name__ == "__main__":
    main()
