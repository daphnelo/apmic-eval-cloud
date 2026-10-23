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
  --v2            ：A7 修正後重跑 E0、E1、E4、E5（修訂版提示詞＋numeric_check），結果存 output/virtual_students_v2/
  --v2 --offline  ：不呼叫 API，以既有閱卷分數＋numeric_check 做離線估計（API 無法連線時也會自動改用）
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
import numeric_check  # noqa: E402

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


def _fmt_like(orig, nv, dec):
    """依原數字的格式（千分位、小數位數）輸出新數字。"""
    nd = len(dec) - 1 if dec else 0
    out = f"{nv:,.{nd}f}" if "," in orig else f"{nv:.{nd}f}"
    return out


def perturb_final_numbers(final):
    """A5：改寫【最終答案】中每一個答案數值（只改算式結果與單獨出現的數值，不改算式中的輸入；
    略過法規編號、日期、小題編號等雜訊與 0、1），
    回傳 (新最終答案, {原數字字串: 新數字字串})；沒有可改的數字時回傳 (None, {})。"""
    mapping, pieces, last = {}, [], 0
    for m in numeric_check.find_numbers(final, results_only=True):
        intpart, dec = m.group(2), m.group(3)
        tok = intpart + (dec or "")
        v = float(tok.replace(",", ""))
        if v in (0.0, 1.0):
            continue
        nv = v * 1.2 + 1 if v >= 10 else v * 1.2 + (0.5 if dec else 1)
        new = _fmt_like(intpart, nv, dec)
        if new == tok:
            continue
        mapping[tok] = new
        pieces += [final[last:m.start(2)], new]
        last = m.end(3) if dec else m.end(2)
    if not mapping:
        return None, {}
    # 最終答案中同一數字作為算式輸入再出現時一併改（如「＝89 … 89／(7%－5%)」），避免最終答案自相矛盾
    return sync_body_numbers("".join(pieces) + final[last:], mapping, results_only=False), mapping


def sync_body_numbers(body, mapping, results_only=True):
    """A5：正文中與最終答案相同的數字（算式結果）同步改成新數字，避免只有最終答案與推論前後矛盾。
    只改 ≥ 10 或含小數的數字（個位整數多為期數、項次，改了會破壞題意）。"""
    pieces, last = [], 0
    for m in numeric_check.find_numbers(body, results_only=results_only):
        intpart, dec = m.group(2), m.group(3)
        tok = intpart + (dec or "")
        if tok in mapping and (dec or float(tok.replace(",", "")) >= 10):
            pieces += [body[last:m.start(2)], mapping[tok]]
            last = m.end(3) if dec else m.end(2)
    return "".join(pieces) + body[last:]


def make_e1(body, final):
    """A5：只用於參考答案【最終答案】含數值的題目；改最終答案數值並同步正文。"""
    if not final or not numeric_check.is_numeric_reference(final):
        return None
    f2, mapping = perturb_final_numbers(final)
    if not f2:
        return None
    return sync_body_numbers(body, mapping) + "\n\n【最終答案】\n" + f2


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

# A6：E4 刪除規則補齊（第 X～Y 條、第 X 條 a 項、段落編號、括號內準則與法規名稱、制度名稱、定理與模型名稱）
LAW_NAMES = ("保險法|銀行法|公司法|證券交易法|民法|票據法|海商法|所得稅法|營業稅法|加值型及非加值型營業稅法|"
             "金融控股公司法|信託法|商業會計法|洗錢防制法|消費者保護法|強制汽車責任保險法|管理外匯條例|"
             "保險業資本適足性管理辦法|保險業各種準備金提存辦法|銀行資本適足性及資本等級管理辦法|"
             "上市上櫃公司治理實務守則|公司治理實務守則|商業事件審理法|國際會計準則|國際財務報導準則|"
             "財務會計準則|企業會計準則|審計準則|授信準則|授信規範|徵信準則|永續報告書作業辦法|溫室氣體盤查議定書|"
             "溫室氣體排放量盤查登錄管理辦法|氣候變遷因應法")
STANDARD = (r"(?:IFRS|IAS|IFRIC|SIC|SFAS|ASC|ISA|ISO|US\s*GAAP|GAAP)\s*\d*(?:\s*[-–]\s*\d+)?"
            r"(?:\s*(?:號)?\s*(?:第\s*)?[A-Z]{0,2}\d+(?:\.\d+)?\s*段)?")
RULEBOOK = (r"(?:(?<![A-Za-z])ICS(?![A-Za-z])|e?UCP\s*\d{3}|URC\s*\d{3}|URDG\s*\d{3}|ISBP\s*\d{3}|ISP\s*98|Incoterms\s*(?:®\s*)?\d{4}|"
            r"Basel\s*(?:I{1,3}|[1-4](?:\.\d)?)|巴塞爾(?:資本)?協定\s*(?:I{1,3}|[一二三])?|GHG\s*Protocol|"
            r"TCFD|SASB|GRI(?:\s*\d+)?|ISSB|IFRS\s*S[12]|PCAF|赤道原則|責任投資原則|PRI|永續會計準則)")
ARTICLE = (r"(?:(?:" + LAW_NAMES + r")\s*)?(?:及|與|、)?第\s*\d+(?:\s*之\s*\d+)?\s*(?:[～~至、\-－]\s*\d+\s*)?條"
           r"(?:\s*之\s*\d+)?(?:\s*第\s*\d+\s*項)?(?:\s*[a-zA-Z]\s*項)?(?:\s*第\s*\d+\s*款)?(?:\s*[a-zA-Z]\s*款)?")
PARA_NO = (r"(?<![A-Za-z0-9])(?:BCZ|BC|IE|AG|IG|DO|B)\d{1,3}(?:\s*[～~\-–]\s*(?:BCZ|BC|IE|AG|IG|DO|B)?\d{1,3})?"
           r"(?![A-Za-z0-9])")  # 準則段落編號（如 B49、BC23、IE5）
SYSTEM = (r"(?:RBC|TW-ICS|Solvency\s*II|IRB)\s*(?:制度|架構|規範|比率)?|"
          r"(?:風險基礎資本|資本適足率|清償能力|[A-Za-z\-]{2,10})\s*制度")
THEORY = (r"[一-鿿A-Za-z\-–]{1,10}(?:定理|定律|模型|理論|假說|法則|公式|原理|恆等式|原則(?!上))|"
          r"(?:乘數|排擠|所得|替代|財富|槓桿)效果|年金現值因子|10\s*%\s*測試|有效利率法|"
          r"\b(?:CAPM|APT|MM|Black[-–]Scholes|Gordon|Fisher|Fama[-–]French|DuPont|杜邦)\b")
# 學者與文獻：Fama & French(1992, 1993)、Samuelson、Bodie指出
AUTHOR = (r"[A-Z][A-Za-z]+(?:\s*(?:&|and|、)\s*[A-Z][A-Za-z]+)*\s*[（(]\d{4}(?:\s*[,，]\s*\d{4})*[)）]|"
          r"[（(]?[A-Z][a-z]+(?:\s*(?:&|and|、)\s*[A-Z][a-z]+)+\s*指出")
BASIS_SECTION = re.compile(r"【[^】（(]{0,8}(?:適用原理|依據|規定|準則|原理|原則|法規|理論|觀念|觀點|公式)[^】（(]{0,6}】[^\n【]*")
BASIS_PAREN = re.compile(r"[（(][^（）()]{0,40}?(?:IFRS|IAS|準則|公報|" + LAW_NAMES + r"|法第|條例|辦法|UCP|ISBP|Incoterms|"
                         r"第\s*\d+\s*(?:條|號|段)|(?:BCZ|BC|IE|AG|IG|DO|B)\d{1,3}|定理|模型|原則|制度|觀點)[^（）()]{0,40}?[)）]")
BASIS_CLAUSE = re.compile(r"(?:依據?|根據|按照|依照|參照|適用)[^，。；：\n]{0,30}?(?:之?規定|規範|準則|原則|定理|模型|理論|公式)"
                          r"(?:[，,]|(?=[。；\n]))")
_TOKEN = "|".join([ARTICLE, STANDARD, RULEBOOK, r"《[^》]{1,40}》", "(?:" + LAW_NAMES + ")", SYSTEM, THEORY, AUTHOR, PARA_NO])
BASIS_TOKEN = re.compile(_TOKEN)
# 連續列舉的依據一起刪（如「SASB、TCFD」「RBC/TW-ICS」），避免留下「、」「/」空殼
BASIS_LIST = re.compile(rf"(?:{_TOKEN})(?:\s*[、/／,，及與和]\s*(?:{_TOKEN}))*")


def residual_basis(text):
    """檢查文字中是否還留有依據（用來判定 E4 是否為有效樣本）。"""
    t = re.sub(r"[XY]\d+(?:/\d+){0,2}", " ", text)
    return [m.group(0) for m in BASIS_TOKEN.finditer(t) if m.group(0).strip()]


def strip_basis(body):
    out = BASIS_SECTION.sub("", body)
    out = BASIS_PAREN.sub("", out)
    out = BASIS_CLAUSE.sub("", out)
    out = BASIS_LIST.sub("", out)
    # 清理刪除後留下的空殼
    out = re.sub(r"[（(]\s*[、，,]*\s*[)）]|【\s*】|「\s*」", "", out)
    out = re.sub(r"(?:依據?|根據|按照|依照)\s*(?:之|的)?\s*(?:規定|規範)\s*[，,]", "", out)
    out = re.sub(r"(?m)^(\s*[•・\-]?\s*)[：:]\s*", r"\1", out)
    out = re.sub(r"[、，,/／]\s*(?=[。；：:）)、，,/／])", "", out)
    out = re.sub(r"(?<=[（(、，,：:])\s*[、，,/／]+", "", out)
    out = re.sub(r"[（(]\s*[)）]", "", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out if out.strip() and out != body else None


def make_e4(body, final):
    """A6：刪除正文與最終答案中的依據；刪除後仍殘留依據者不採用（只留有效樣本）。"""
    s = strip_basis(body)
    if not s:
        return None
    f4 = (strip_basis(final) or final) if final else ""
    txt = s + ("\n\n【最終答案】\n" + f4 if f4 else "")
    return None if residual_basis(txt) else txt


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
        e1 = make_e1(body, final)
        if e1:
            variants["E1"] = e1
        if is_multipart(it["題目"]):
            d = drop_later_parts(body)
            if d:
                variants["E2"] = d
        peers = [p for p in by_subj[subj] if p["ID"] != it["ID"]]
        if peers:
            variants["E3"] = base.reference_text(peers[0])
        e4 = make_e4(body, final)
        if e4:
            variants["E4"] = e4
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


def judge_all(client, items, answers, judgments, use_numeric_check=None, judge_jsonl=None):
    judge_jsonl = judge_jsonl or JUDGE_JSONL
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
        rec = {"vid": a["vid"], "judge": j,
               **base.post_check(by_id[a["ID"]], a["answer"], s, caps, raw_sum, final, use_numeric_check),
               "cap_reasons": v.get("cap_reasons", ""),
               "evidence": {p: pts[p].get("evidence", "") for p in base.POINTS}}
        base.append_jsonl(judge_jsonl, rec)
        final = rec["final_score"]
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


# ===== A7：修正後重跑（E0、E1、E4、E5） =====
OUT_V2 = os.path.join(os.path.dirname(base.OUT_DIR), "virtual_students_v2")
V2_ITEMS_CSV = os.path.join(OUT_V2, "vs_v2_items.csv")
V2_ANS_JSONL = os.path.join(OUT_V2, "vs_v2_answers.jsonl")
V2_JUDGE_JSONL = os.path.join(OUT_V2, "vs_v2_judgments.jsonl")
V2_SUMMARY_MD = os.path.join(OUT_V2, "vs_v2_summary.md")
V2_TYPES = ("E0", "E1", "E4", "E5")
E4_MIN = 30  # E4 有效樣本目標
TARGETS = OrderedDict([  # 驗收項目：(說明, 目標, 比較方向)
    ("E1", ("E1 抓到率", 0.90, ">=")),
    ("E5", ("E5 純結論作答扣 P4", 0.90, ">=")),
    ("E4n", ("E4 有效樣本數", 30, ">=")),
    ("E4", ("E4 扣 P5 比率", 0.80, ">=")),
    ("E0", ("E0 正確答案被誤判", 0.05, "<=")),
])


def supplement_items(items):
    """題庫中未入選的非選擇題，依科別輪流排序（用來補足 E4 樣本）。"""
    chosen = {it["ID"] for it in items}
    by_subj = OrderedDict()
    for it in base.sample_items(base.load_items()):
        if it["ID"] not in chosen:
            by_subj.setdefault(it["ID"].split("-")[0], []).append(it)
    rng = random.Random(SEED)
    for lst in by_subj.values():
        rng.shuffle(lst)
    order = []
    while any(by_subj.values()):
        for lst in by_subj.values():
            if lst:
                order.append(lst.pop(0))
    return order


def build_v2_answers(items):
    """E0、E5 沿用既有作答；E1、E4 以修正後的 make_injected() 重新產生。
    45 題的 E4 不足 E4_MIN 份時，從其他非選擇題依科別輪流補題（只產生 E4，標記 supplement）。"""
    old = base.load_done(ANS_JSONL, lambda r: r["vid"]) if os.path.exists(ANS_JSONL) else {}
    new = {r["vid"]: r for r in make_injected(items)}
    recs = []
    for it in items:
        for t in V2_TYPES:
            vid = f"{it['ID']}|{t}"
            rec = old.get(vid) if t in ("E0", "E5") else None
            rec = rec or new.get(vid)
            if rec:
                recs.append(dict(rec))
    extra = []
    n4 = sum(r["type"] == "E4" for r in recs)
    for it in supplement_items(items):
        if n4 >= E4_MIN:
            break
        body, final = split_reference(base.reference_text(it))
        e4 = make_e4(body, final)
        if e4:
            recs.append({"vid": f"{it['ID']}|E4", "ID": it["ID"], "kind": "inject", "type": "E4",
                         "generator": "program", "supplement": True, "answer": e4})
            extra.append(it)
            n4 += 1
    return recs, extra


def is_pure_conclusion(answer):
    """E5 中「純結論」作答：最終答案沒有算式（無＝、=，也沒有「數字 運算子 數字」）。"""
    _, final, _ = numeric_check.split_final(answer)
    return not re.search(r"[＝=]|\d\s*[×÷/／＋－+\-]\s*\d", final)


def rescore_offline(item, answer, jrec):
    """以既有 LLM 閱卷分數（修正前提示詞）加上 numeric_check 重算。"""
    s = {p: float(jrec[p]) for p in base.POINTS}
    caps = [c for c in jrec["caps"].split(",") if c]
    return base.post_check(item, answer, s, caps, sum(s.values()), jrec["final_score"], True)


def llm_only(rec):
    """取出套用 numeric_check 之前的 LLM 分數（若有）。"""
    if "llm_final_score" not in rec:
        return rec
    return {**{p: rec[f"llm_{p}"] for p in base.POINTS}, "final_score": rec["llm_final_score"], "caps": rec["llm_caps"]}


def rates(answers, judgments, judge, fn=lambda r: r):
    """回傳各驗收項目的 (分子, 分母)。"""
    out = {}
    for key, t, filt in (("E1", "E1", None), ("E5", "E5", is_pure_conclusion), ("E4", "E4", None), ("E0", "E0", None)):
        rs = [fn(judgments[(a["vid"], judge)]) for a in answers.values()
              if a["type"] == t and (filt is None or filt(a["answer"])) and (a["vid"], judge) in judgments]
        hit = sum(detected(t, r) for r in rs)
        out[key] = (len(rs) - hit if key == "E0" else hit, len(rs))
    out["E4n"] = (sum(a["type"] == "E4" for a in answers.values()), None)
    return out


def fmt(v):
    n, d = v
    if d is None:
        return f"{n} 份"
    return f"{n}/{d}（{n / d:.0%}）" if d else "-"


def passed(key, v):
    n, d = v
    if d == 0:
        return None
    x = n if d is None else n / d
    _, tgt, op = TARGETS[key]
    return x >= tgt if op == ">=" else x <= tgt


def persona_side_effects(by_id):
    """副作用檢查：numeric_check 對既有虛擬考生作答（非埋錯、無標準答案）會改動多少 LLM 判定。"""
    ans = [a for a in base.load_done(ANS_JSONL, lambda r: r["vid"]).values() if a["kind"] == "persona"]
    old_j = base.load_done(JUDGE_JSONL, lambda r: (r["vid"], r["judge"]))
    num = [a for a in ans if numeric_check.numeric_c1(base.reference_text(by_id[a["ID"]]), "")["status"] != "不適用"]
    trig = [a for a in num if numeric_check.numeric_c1(base.reference_text(by_id[a["ID"]]), a["answer"])["trigger_c1"]]
    p1 = lambda a: [old_j.get((a["vid"], j), {}).get("P1") for j in JUDGES]  # noqa: E731
    both1 = sum(all(x == 1.0 for x in p1(a)) for a in trig)
    both0 = sum(all(x == 0.0 for x in p1(a)) for a in trig)
    capped = [a for a in ans if numeric_check.p4_cap(a["answer"])["cap_p4"] is not None]
    p4 = lambda a: [old_j.get((a["vid"], j), {}).get("P4") for j in JUDGES]  # noqa: E731
    cap_hi = sum(any(x == 1.0 for x in p4(a)) for a in capped)
    by_s = defaultdict(int)
    for a in capped:
        by_s[a["type"]] += 1
    return [
        "## 副作用檢查：既有虛擬考生作答（450 份，非埋錯）",
        "",
        f"- 數值比對：適用題目的作答 {len(num)} 份，觸發 C1 {len(trig)} 份（{len(trig) / len(num):.0%}）；"
        f"其中兩位裁判原本都給 P1 = 0 的 {both0} 份、都給 P1 = 1 的 {both1} 份，其餘為 0.5 或不一致。",
        f"  兩位裁判都給 P1 = 1 卻被觸發的 {both1} 份，逐份檢查皆為作答未算出參考答案的關鍵數值（CFIN-012 有 8 份未算 β、"
        "權益成本、WACC、NPV，只以「現金流入大於投資」判斷值得投資；ESG-006 未寫 450 噸；TRD-022 未算出報價），"
        "屬裁判寬鬆而非程式誤判；正式評測前仍建議抽查這類案例。",
        f"- 推論長度：P4 上限 0.5 共 {len(capped)} 份（" + "、".join(f"{k} {v}" for k, v in sorted(by_s.items())) +
        f"），其中至少一位裁判原給 P4 = 1 的 {cap_hi} 份；集中在 S06「計算強但說明弱」，符合新版 P4「只有一行算式、沒有說明」= 0.5 的定義。",
        "",
    ]


def report_v2(items_all, answers_v2, judgments_v2, offline, offline_reason=""):
    by_id = {it["ID"]: it for it in items_all}
    old_ans = base.load_done(ANS_JSONL, lambda r: r["vid"])
    old_j = base.load_done(JUDGE_JSONL, lambda r: (r["vid"], r["judge"]))
    old_ans = {v: a for v, a in old_ans.items() if a["kind"] == "inject" and a["type"] in V2_TYPES}
    before = {j: rates(old_ans, old_j, j) for j in JUDGES}
    n_pure = sum(is_pure_conclusion(a["answer"]) for a in answers_v2.values() if a["type"] == "E5")
    n_sup = sum(bool(a.get("supplement")) for a in answers_v2.values())
    L = ["# 裁判評分改正：修正前後對照（E0、E1、E4、E5）", ""]
    L.append(f"- 選題：45 題（同修正前）；E4 另從其他非選擇題依科別輪流補 {n_sup} 題，只產生 E4。")
    L.append("- 作答：" + "、".join(f"{t} {sum(a['type'] == t for a in answers_v2.values())} 份" for t in V2_TYPES) +
             f"（E5 中純結論 {n_pure} 份）。E0、E5 沿用既有作答；E1、E4 由修正後的 `make_injected()` 重新產生。")
    L.append("- 修正前：`output/virtual_students/vs_judgments.jsonl`（舊提示詞、無 numeric_check）。")
    L.append("- 「抓到」判定沿用 `virtual_students.detected()`：E0 ≥ 4.5 分；E1 ≤ 1 分或 C1；E4 P5 ≤ 0.5；E5 P4 ≤ 0.5。")
    L.append("- E5「純結論」定義：最終答案沒有算式（無「＝」也沒有「數字 運算子 數字」）。修正前比率依同一定義以既有閱卷重算"
             "（規劃書以另一種切分得 24 份、50%／58%，此處為 "
             f"{n_pure} 份）。")
    L.append("")

    if not offline:
        after = {j: rates(answers_v2, judgments_v2, j) for j in JUDGES}
        after_llm = {j: rates(answers_v2, judgments_v2, j, llm_only) for j in JUDGES}
        L.append(f"閱卷：{len(judgments_v2)} 筆（26B、31B 各閱卷一次，修訂版提示詞＋numeric_check）。")
        L.append("")
        L.append("## 驗收表")
        L.append("")
        L.append("| 項目 | 修正前 26B | 修正前 31B | 修正後 26B | 修正後 31B | 目標 | 達標 |")
        L.append("|---|---|---|---|---|---|---|")
        for k, (name, tgt, op) in TARGETS.items():
            ok = [passed(k, after[j][k]) for j in JUDGES]
            tg = f"{op.replace('>=', '≥').replace('<=', '≤')} {tgt:.0%}" if isinstance(tgt, float) else f"≥ {tgt} 份"
            L.append(f"| {name} | {fmt(before[JUDGES[0]][k])} | {fmt(before[JUDGES[1]][k])} | "
                     f"{fmt(after[JUDGES[0]][k])} | {fmt(after[JUDGES[1]][k])} | {tg} | "
                     f"{'✅' if all(ok) else '❌'} |")
        L.append("")
        L.append("## 拆解：修訂版提示詞（LLM 原始分數） vs 加上 numeric_check")
        L.append("")
        L.append("| 項目 | 26B 只用 LLM | 26B＋numeric_check | 31B 只用 LLM | 31B＋numeric_check |")
        L.append("|---|---|---|---|---|")
        for k in ("E1", "E5", "E4", "E0"):
            L.append(f"| {TARGETS[k][0]} | {fmt(after_llm[JUDGES[0]][k])} | {fmt(after[JUDGES[0]][k])} | "
                     f"{fmt(after_llm[JUDGES[1]][k])} | {fmt(after[JUDGES[1]][k])} |")
        L.append("")
        fails = [(k, j) for k in TARGETS for j in JUDGES if passed(k, after[j][k]) is False]
        L.append("## 未達標項目")
        L.append("")
        L.append("\n".join(f"- {TARGETS[k][0]}（{SHORT[j]} 閱卷）：{fmt(after[j][k])}" for k, j in fails) if fails else "- 無")
        L.append("")
        L += persona_side_effects(by_id)
        return "\n".join(L)

    # ===== 離線模式：無法呼叫 API =====
    L.append(f"> **A7 未執行（無法呼叫 API）**：{offline_reason}")
    L.append("> 以下「修正後」為離線估計：既有 LLM 閱卷分數（舊提示詞）＋ numeric_check；新產生的 E1、E4 沒有 LLM 閱卷，")
    L.append("> 只能計算 numeric_check 單獨保證的下限（numeric_check 觸發 C1 時總分必 ≤ 1，不論 LLM 給幾分）。")
    L.append("")
    adj = {}
    for j in JUDGES:
        adj_j = {}
        for a in old_ans.values():
            r = old_j.get((a["vid"], j))
            if r:
                adj_j[(a["vid"], j)] = rescore_offline(by_id[a["ID"]], a["answer"], r)
        adj[j] = rates(old_ans, adj_j, j)
    new_e1 = [a for a in answers_v2.values() if a["type"] == "E1"]
    trig = sum(numeric_check.numeric_c1(base.reference_text(by_id[a["ID"]]), a["answer"])["trigger_c1"] for a in new_e1)
    old_e1 = [a for a in old_ans.values() if a["type"] == "E1"]
    old_trig = sum(numeric_check.numeric_c1(base.reference_text(by_id[a["ID"]]), a["answer"])["trigger_c1"] for a in old_e1)
    old_na = sum(numeric_check.numeric_c1(base.reference_text(by_id[a["ID"]]), a["answer"])["status"] == "不適用" for a in old_e1)
    e5 = [a for a in answers_v2.values() if a["type"] == "E5" and is_pure_conclusion(a["answer"])]
    cap5 = sum(numeric_check.p4_cap(a["answer"])["cap_p4"] is not None for a in e5)
    e0 = [a for a in answers_v2.values() if a["type"] == "E0"]
    e0_trig = sum(numeric_check.numeric_c1(base.reference_text(by_id[a["ID"]]), a["answer"])["trigger_c1"] or
                  numeric_check.p4_cap(a["answer"])["cap_p4"] is not None for a in e0)
    n4 = sum(a["type"] == "E4" for a in answers_v2.values())
    L.append("## 驗收表")
    L.append("")
    L.append("| 項目 | 修正前 26B | 修正前 31B | 修正後（離線估計） | 目標 | 達標 |")
    L.append("|---|---|---|---|---|---|")
    rows = [
        ("E1", f"新 E1：numeric_check 觸發 C1 {fmt((trig, len(new_e1)))}，即兩位裁判抓到率 ≥ {trig / len(new_e1):.0%}"
               f"；舊 E1 重算：26B {fmt(adj[JUDGES[0]]['E1'])}、31B {fmt(adj[JUDGES[1]]['E1'])}（numeric_check 單獨觸發 {fmt((old_trig, len(old_e1)))}）",
         trig / len(new_e1) >= 0.9),
        ("E5", f"推論檢查 P4 上限 0.5 {fmt((cap5, len(e5)))}，即兩位裁判扣 P4 ≥ {cap5 / len(e5):.0%}"
               f"（舊分數重算：26B {fmt(adj[JUDGES[0]]['E5'])}、31B {fmt(adj[JUDGES[1]]['E5'])}）", cap5 / len(e5) >= 0.9),
        ("E4n", f"{n4} 份（45 題產生 {n4 - n_sup} 份＋補題 {n_sup} 份；刪除後皆無殘留依據）", n4 >= E4_MIN),
        ("E4", "未執行（需 LLM 閱卷判定 P5；numeric_check 不處理 P5）", None),
        ("E0", f"numeric_check 誤觸發 {fmt((e0_trig, len(e0)))}；舊分數重算被誤判：26B {fmt(adj[JUDGES[0]]['E0'])}、"
               f"31B {fmt(adj[JUDGES[1]]['E0'])}（修訂版提示詞是否變嚴需 A7 驗證）", None),
    ]
    for k, txt, ok in rows:
        name, tgt, op = TARGETS[k]
        tg = f"{op.replace('>=', '≥').replace('<=', '≤')} {tgt:.0%}" if isinstance(tgt, float) else f"≥ {tgt} 份"
        L.append(f"| {name} | {fmt(before[JUDGES[0]][k])} | {fmt(before[JUDGES[1]][k])} | {txt} | {tg} | "
                 f"{'✅（程式下限）' if ok else ('待 A7' if ok is None else '❌')} |")
    L.append("")
    L.append("## 未達標／未執行項目與原因")
    L.append("")
    L.append(f"- **E4 扣 P5 比率：未執行。** P5 只能由 LLM 裁判判定，需以修訂版提示詞重新閱卷（A7）；原因：{offline_reason}。")
    L.append(f"- **E0 正確答案被誤判：未執行（部分）。** numeric_check 對 E0 誤觸發 {e0_trig}/{len(e0)}，程式本身不會讓 E0 變嚴；"
             "但修訂版 P1、P4 規則是否讓 LLM 變嚴，需 A7 重新閱卷才能確認。")
    L.append("- **E1、E5：** 表中為 numeric_check 單獨保證的下限，已達標；LLM 本身（修訂版提示詞）的抓到率需 A7 才能量測。"
             f"舊 E1 中 numeric_check 未觸發的 {len(old_e1) - old_trig} 份：{old_na} 份參考答案最終答案不含關鍵數值（不適用數值比對），"
             f"{len(old_e1) - old_trig - old_na} 份舊埋錯只改了部分數字（過半數值仍一致），正是 A5 要修正的埋錯瑕疵；"
             "新 E1 已改為只用於數值答案、改動全部答案數值並同步正文。")
    L.append("- 補跑方式：設定 `APMIC_API_KEY` 且網路可連到 API 端點後執行 `python eval/virtual_students.py --v2`，"
             "會產生 `vs_v2_judgments.jsonl` 並以實際閱卷結果覆寫本報告。")
    L.append("")
    L += persona_side_effects(by_id)
    return "\n".join(L)


def main_v2():
    os.makedirs(OUT_V2, exist_ok=True)
    items = select_items()
    recs, extra = build_v2_answers(items)
    items_all = items + extra
    with open(V2_ITEMS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID", "科目", "來源"])
        for it in items_all:
            w.writerow([it["ID"], it["科目"], "E4 補題" if it in extra else "原 45 題"])
    answers = base.load_done(V2_ANS_JSONL, lambda r: r["vid"]) if os.path.exists(V2_ANS_JSONL) else {}
    for rec in recs:
        if rec["vid"] not in answers:
            answers[rec["vid"]] = rec
            base.append_jsonl(V2_ANS_JSONL, rec)
    print("v2 作答：" + "，".join(f"{t} {sum(a['type'] == t for a in answers.values())}" for t in V2_TYPES))
    judgments = base.load_done(V2_JUDGE_JSONL, lambda r: (r["vid"], r["judge"])) if os.path.exists(V2_JUDGE_JSONL) else {}
    offline, reason = "--offline" in sys.argv, "以 --offline 執行"
    extra_reason = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--offline-reason=")), "")
    if not offline and "--report-only" not in sys.argv:
        if base.API_KEY in ("", "EMPTY"):
            offline, reason = True, "環境變數 APMIC_API_KEY 未設定"
        else:
            from openai import OpenAI
            client = OpenAI(base_url=base.API_BASE_URL, api_key=base.API_KEY)
            try:  # 先用一次小請求測試連線
                base.chat(client, JUDGES[0], [{"role": "user", "content": "ping"}], max_tokens=1)
            except Exception as e:
                offline, reason = True, f"連線測試失敗（{type(e).__name__}）"
            if not offline:
                judge_all(client, items_all, answers, judgments, True, V2_JUDGE_JSONL)
    if "--report-only" in sys.argv and not judgments:
        offline, reason = True, "尚無 v2 閱卷結果"
    md = report_v2(items_all, answers, judgments, offline, reason + (f"；{extra_reason}" if extra_reason else ""))
    with open(V2_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print("\n" + md)


def main():
    if "--v2" in sys.argv:
        return main_v2()
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
