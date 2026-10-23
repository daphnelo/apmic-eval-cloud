"""數值比對與推論長度檢查（改正行動 A2＋A4）

原則：數字對不對、推論有沒有寫，能用程式客觀檢查的，不交給 LLM 裁判判斷。

A2 數值比對（觸發 C1）
  1. 擷取作答的【最終答案】段落（無此標記時取最後一段；若最後一段就是全文，取最後一句）。
  2. 擷取參考答案【最終答案】中的「關鍵數值」（算式結果與單獨出現的數值，不含 0、1）。會先排除不屬於答案的數字：
     小題編號（(1)、（2））、法規與準則編號（第 X 條、IFRS 9、IAS 2、UCP600、Incoterms 2000）、
     日期與會計年度（X7/1/1、20X1、109年8月）、第 N 年／期、角度（45°）、版本號（公司治理3.0）等。
     參考答案只含 0、1（多半是「0～1」這類範圍描述）時不視為數值題。
  3. 逐一比對：每個參考數值配對一個尚未使用、相對誤差 ≤ 0.5% 的作答數值即視為一致
     （重複出現的數值，如分錄借貸同額，要各自對得上）。
     比對時考慮常見的等價寫法：正負號（−3.464 與 3.464）、百分比（12.56% 與 0.1256）、
     萬／億單位（414.085萬 與 4,140,850）。
  4. 判定：一致比率 < MATCH_RATIO_MIN（50%）即「觸發 C1」。
     門檻理由：參考答案的最終答案常同時列出中間值與題目給定的輸入（如 9/12、25%），
     正確作答不一定重抄每一個；但過半關鍵數值都對不上，代表最終答案本身是錯的。
     只有 1 個關鍵數值時，等同要求該數值一致。
  5. 作答最終答案完全沒有數字時回傳「無法判定」，不觸發 C1（交給 LLM 依 P1 規則判定）。
  6. 作答沒有【最終答案】標記時先比對最後一段；最後一段一致比率不足時改以全文比對
     （多小題作答的最後一段常只是最後一小題或結語，直接判 C1 會冤枉正確作答）。
  參考答案只取「算式結果與單獨出現的數值」，略過算式中的運算元（如 (500−400)×17% 的 500、400、17%），
  因為正確作答不一定重抄題目給定的輸入。
  只處理參考答案【最終答案】含數值的題目；其他題目回傳「不適用」。

A4 推論長度（P4 上限 0.5）
  計算【最終答案】以外的推論字數（中英文字與數字，不含空白與標點）與步驟數
  （以換行與「。；」切分後的非空片段數）。
  推論字數 < MIN_REASON_CHARS（40 字）或步驟數 < MIN_REASON_STEPS（2 步）時回傳「P4 上限 0.5」。
  門檻理由：依新版 P4 錨點，「只有一行算式、沒有說明」最多 0.5 分；完整推論至少要有
  「公式或原則 → 代入條件」兩步，且 40 字約等於一句原則加一行代入算式。
  以既有 45 題參考答案檢查，推論部分最短的也遠高於門檻（見 `python numeric_check.py --stats`），
  虛擬考生中「精簡型」作答也不會被誤擋，故門檻只攔截「只寫答案」或「一行帶過」的作答。
"""
import re

FINAL_TAG = "【最終答案】"
REL_TOL = 0.005          # 相對誤差 ≤ 0.5% 視為一致
MATCH_RATIO_MIN = 0.5    # 參考數值一致比率低於此值 → 觸發 C1
MIN_REASON_CHARS = 40    # 推論字數門檻
MIN_REASON_STEPS = 2     # 推論步驟數門檻

# ===== 擷取最終答案 =====
def split_final(text):
    """回傳 (推論部分, 最終答案部分, 是否有【最終答案】標記)。"""
    text = (text or "").strip()
    if FINAL_TAG in text:
        body, final = text.rsplit(FINAL_TAG, 1)
        return body.strip(), final.strip(), True
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) >= 2:
        return "\n\n".join(paras[:-1]).strip(), paras[-1].strip(), False
    # 只有一段：取最後一句當最終答案
    sents = [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]
    if len(sents) >= 2:
        return "".join(sents[:-1]).strip(), sents[-1].strip(), False
    return "", text, False


# ===== 擷取數值 =====
# 不屬於答案的數字：先整段移除再擷取
_NOISE = [
    r"第\s*\d+\s*(?:[～~至\-－]\s*\d+\s*)?(?:條|項|款|段|章|節|題|小題|年|期|季|次|個月|月|天|週)",
    r"(?:IFRS|IAS|IFRIC|SIC|SFAS|ASC|ISA|ISO|ISBP|UCP|URC|URDG|ISP|Incoterms|Basel|TW-ICS)\s*[-–]?\s*\d+(?:\.\d+)?",
    r"\d{0,2}[XY]\d+(?:\s*/\s*\d{1,2}){0,2}",          # 會計年度 X7、20X1、X7/1/1
    r"\d{1,4}\s*/\s*\d{1,2}\s*/\s*\d{1,2}",             # 日期 2024/1/1
    r"\d{1,4}\s*年(?:\s*\d{1,2}\s*月)?(?:\s*\d{1,2}\s*日)?",  # 109年8月、第 0 年、2024年
    r"\d{1,2}\s*月\s*\d{1,2}\s*日",
    r"[（(]\s*\d{1,2}\s*[)）]",                         # 小題編號 (1)、（2）
    r"^\s*\d{1,2}\s*[.、]\s",                       # 行首編號 1. 2、
    r"\d+(?:\.\d+)?\s*°",                               # 角度 45°
    r"(?<=[一-鿿])\d\.0(?![\d%])",              # 版本號 公司治理3.0
    r"[A-Za-z]+[\-_]?\d+",                              # 英文代號 B49、CO2、z_2
    r"[A-Za-z]_?[（(][^（）()]{0,6}[)）]",                # 下標 z_(α/2)
]
_NOISE_RE = re.compile("|".join(f"(?:{p})" for p in _NOISE), re.M)
_NUM_RE = re.compile(r"(?<![\d.])([−\-－]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|％|萬|億|千)?")
_UNIT = {"萬": 1e4, "億": 1e8, "千": 1e3}


_BIN_OPS = "×*÷/／^＋+−-－"


def _operandish(ch):
    """運算子另一側是數字、括號或變數時，才算是真正的二元運算。"""
    return bool(ch) and (ch.isdigit() or ch in "()（）√%％" or re.match(r"[A-Za-zα-ωΑ-Ω]", ch) is not None)


def _is_operand(clean, m):
    """數字是算式中的運算元（如 (500−400)×17% 中的 500、400、17%），而非算式結果。
    分錄中的「80,000／貸」、清單中的「−」等，運算子另一側不是數字或變數，不算運算元。"""
    after = clean[m.end():m.end() + 6].lstrip(" )）")
    if after[:1] and after[0] in _BIN_OPS and _operandish(after[1:].lstrip(" ")[:1]):
        return True
    start = m.start() + len(m.group(1))   # 不含正負號
    before = clean[max(0, start - 6):start].rstrip(" (（")
    if not (before[-1:] and before[-1] in _BIN_OPS):
        return False
    prev = before[:-1].rstrip(" ")[-1:]
    if before[-1] in "−-－＋+" and m.group(1) == "" and not _operandish(prev):
        return False
    return _operandish(prev) or before[-1] not in "−-－＋+"


def find_numbers(text, results_only=False):
    """回傳數字的 regex match（位置對應原文，雜訊以等長空白遮蔽），供擷取與改寫數字使用。"""
    clean = _NOISE_RE.sub(lambda m: " " * len(m.group(0)), text or "")
    return [m for m in _NUM_RE.finditer(clean) if not (results_only and _is_operand(clean, m))]


def extract_numbers(text, results_only=False):
    """回傳數值清單，每個元素為 (原字串, 可接受的等價數值集合)。

    results_only=True 時略過算式中的運算元，只留算式結果與單獨出現的數值（用於參考答案）。"""
    out = []
    for m in find_numbers(text, results_only):
        sign, intpart, dec, unit = m.groups()
        v = abs(float(intpart.replace(",", "") + (dec or "")))
        forms = {v}
        if unit in ("%", "％"):
            forms.add(v / 100)
        elif unit in _UNIT:
            forms.add(v * _UNIT[unit])
        out.append((m.group(0).strip(), forms))
    return out


def _close(a, b):
    if a == b:
        return True
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b))


def _match_all(ref_nums, ans_nums):
    """逐一比對：每個參考數值配對一個尚未使用的作答數值（重複出現的數值要各自對得上）。
    回傳 (一致清單, 不一致清單)。"""
    used = [False] * len(ans_nums)
    matched, missing = [], []
    for s, rf in ref_nums:
        k = next((i for i, (_, af) in enumerate(ans_nums)
                  if not used[i] and any(_close(r, a) for r in rf for a in af)), None)
        if k is None:
            missing.append(s)
        else:
            used[k] = True
            matched.append(s)
    return matched, missing


def key_numbers(ref_final):
    """參考答案最終答案的關鍵數值：算式結果與單獨出現的數值，排除 0 與 1（多為門檻或比較，如 NPV>0）。"""
    return [(s, fs) for s, fs in extract_numbers(ref_final, results_only=True) if min(fs) not in (0.0, 1.0)]


def is_numeric_reference(ref_final):
    return bool(key_numbers(ref_final))


# ===== A2：數值比對 =====
def numeric_c1(reference, answer):
    """回傳 dict：status ∈ {不適用, 無法判定, 一致, 觸發C1}、比對明細。"""
    _, ref_final, _ = split_final(reference)
    if FINAL_TAG not in (reference or "") or not is_numeric_reference(ref_final):
        return {"status": "不適用", "trigger_c1": False}
    ref_nums = key_numbers(ref_final)
    _, ans_final, has_tag = split_final(answer)
    ans_nums = extract_numbers(ans_final)
    scope = "最終答案" if has_tag else "最後一段"
    if not ans_nums and has_tag:
        return {"status": "無法判定", "trigger_c1": False, "ref_numbers": [s for s, _ in ref_nums],
                "reason": "作答最終答案沒有數值"}
    matched, missing = _match_all(ref_nums, ans_nums)
    if not has_tag and len(matched) / len(ref_nums) < MATCH_RATIO_MIN:
        # 沒有【最終答案】標記時，最後一段常只是最後一小題或結語；改以全文比對，避免誤判正確作答
        all_nums = extract_numbers(answer)
        all_matched, all_missing = _match_all(ref_nums, all_nums)
        if len(all_matched) > len(matched) or not ans_nums:
            ans_nums, matched, missing, scope = all_nums, all_matched, all_missing, "全文（無最終答案標記）"
        if not ans_nums:
            return {"status": "無法判定", "trigger_c1": False, "ref_numbers": [s for s, _ in ref_nums],
                    "reason": "作答沒有數值"}
    ratio = len(matched) / len(ref_nums)
    trig = ratio < MATCH_RATIO_MIN
    return {"status": "觸發C1" if trig else "一致", "trigger_c1": trig, "match_ratio": round(ratio, 3),
            "matched": matched, "missing": missing, "answer_numbers": [s for s, _ in ans_nums],
            "final_tag": has_tag, "scope": scope,
            "reason": (f"參考答案最終答案 {len(ref_nums)} 個數值僅 {len(matched)} 個一致（< {MATCH_RATIO_MIN:.0%}）"
                       if trig else f"{len(matched)}/{len(ref_nums)} 個數值一致")}


# ===== A4：推論長度 =====
_CHAR_RE = re.compile(r"[一-鿿A-Za-z0-9]")


def reasoning_stats(answer):
    body, _, _ = split_final(answer)
    chars = len(_CHAR_RE.findall(body))
    steps = len([s for s in re.split(r"[\n。；;]", body) if _CHAR_RE.search(s)])
    return chars, steps


def p4_cap(answer):
    chars, steps = reasoning_stats(answer)
    short = chars < MIN_REASON_CHARS or steps < MIN_REASON_STEPS
    return {"cap_p4": 0.5 if short else None, "reason_chars": chars, "reason_steps": steps,
            "reason": (f"推論 {chars} 字／{steps} 步，低於門檻（{MIN_REASON_CHARS} 字、{MIN_REASON_STEPS} 步）→ P4 上限 0.5"
                       if short else f"推論 {chars} 字／{steps} 步")}


def check(reference, answer):
    """一次回傳兩項檢查結果。"""
    return {"numeric": numeric_c1(reference, answer), "p4": p4_cap(answer)}


# ===== 套用到 FTRC 分數 =====
def apply(s, caps, check_result, cap_limits, points=("P1", "P2", "P3", "P4", "P5")):
    """在 ftrc_eval_v2.score() 之後套用程式檢查結果。

    s：score() 回傳的各評分點分數；caps：封頂清單。回傳 (新 s, 新 caps, 新 raw, 新 final, 調整說明)。
    """
    s = dict(s)
    caps = list(caps)
    notes = []
    if check_result["numeric"]["trigger_c1"]:
        if s["P1"] != 0:
            notes.append(f"P1 {s['P1']}→0（數值比對：{check_result['numeric']['reason']}）")
            s["P1"] = 0.0
        if "C1" not in caps:
            caps.append("C1")
            notes.append("觸發 C1（數值比對）")
    cap = check_result["p4"]["cap_p4"]
    if cap is not None and s["P4"] > cap:
        notes.append(f"P4 {s['P4']}→{cap}（{check_result['p4']['reason']}）")
        s["P4"] = cap
    raw = sum(s[p] for p in points)
    final = min([raw] + [cap_limits[c] for c in caps])
    return s, caps, raw, final, "；".join(notes)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if "--stats" in sys.argv:
        import virtual_students as vs
        import ftrc_eval_v2 as base
        items = vs.select_items()
        rs = [(it["ID"], *reasoning_stats(base.reference_text(it))) for it in items]
        rs.sort(key=lambda x: x[1])
        print("參考答案推論字數最短 5 題：", rs[:5])
        num = [it["ID"] for it in items if is_numeric_reference(split_final(base.reference_text(it))[1])]
        print(f"參考答案最終答案含數值：{len(num)}/{len(items)} 題：{num}")
