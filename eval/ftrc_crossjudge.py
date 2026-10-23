"""非選擇題 FTRC 交叉閱卷：改由 26B 閱卷，量化 31B 的自我偏好

做法：
1. 讀取 ftrc_eval_v2.py 已存的 724 份回答（answers.jsonl），不重新作答。
2. 用同一份評分標準、同一份參考答案，改由 gemma-4-26B-A4B-it 閱卷，
   結果另存 judgments_judge26B.jsonl（支援中斷續跑）。
3. 與原本 31B 閱卷的結果（judgments.jsonl）合併成 2 × 2 分數表：
   作答模型（26B／31B）× 閱卷模型（26B／31B）。
4. 輸出：
   - 自我偏好：同一份回答，作答者自己閱卷 vs 對方閱卷的分數差
   - 兩位裁判的一致性：完全一致率、差距 ≤ 0.5 比率、加權 kappa、相關係數
   - 公平比較：只用 26B 閱卷、以及兩位裁判平均分，分別比較兩模型（Wilcoxon）

使用方式：與 ftrc_eval_v2.py 放在同一個資料夾，在 Spyder 開啟本檔按 F5。
"""
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftrc_eval_v2 as base  # noqa: E402  沿用題目、評分標準、計分規則與 API 設定

CROSS_JUDGE = "gemma-4-26B-A4B-it"
ORIG_JUDGE = base.MODEL_JUDGE  # gemma-4-31B-it
CROSS_JSONL = os.path.join(base.OUT_DIR, "judgments_judge26B.jsonl")
OUT_CSV = os.path.join(base.OUT_DIR, "crossjudge_scores.csv")
OUT_TXT = os.path.join(base.OUT_DIR, "crossjudge_summary.txt")
SHORT = {"gemma-4-26B-A4B-it": "26B", "gemma-4-31B-it": "31B"}
POINTS = base.POINTS


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def pearson(x, y):
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def weighted_kappa(x, y):
    """二次加權 kappa；分數 0～5，每級 0.5，共 11 類。"""
    cats = [i / 2 for i in range(11)]
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    n = len(x)
    obs = [[0] * k for _ in range(k)]
    for a, b in zip(x, y):
        obs[idx[round(a * 2) / 2]][idx[round(b * 2) / 2]] += 1
    ra = [sum(r) for r in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * obs[i][j]
            den += w * ra[i] * cb[j] / n
    return 1 - num / den if den else float("nan")


def wil(x, y, label):
    w, p, n = base.wilcoxon_signed_rank(x, y)
    return f"{label}：W = {w:.1f}，非同分 {n} 題，p = {p:.4g}（{'顯著' if p < 0.05 else '不顯著'}）"


def run_judging(items, answers):
    from openai import OpenAI
    client = OpenAI(base_url=base.API_BASE_URL, api_key=base.API_KEY)
    base.MODEL_JUDGE = CROSS_JUDGE  # base.judge() 依此變數選擇裁判
    done = base.load_done(CROSS_JSONL, lambda r: (r["ID"], r["model"]))
    keys = [(it["ID"], m) for it in items for m in base.TESTEE_MODELS if (it["ID"], m) in answers]
    for n, key in enumerate(keys, 1):
        print(f"[{n}/{len(keys)}] {key[0]} {SHORT[key[1]]} 作答，26B 閱卷 ...", end="", flush=True)
        if key in done:
            print(f" 已完成 {done[key]['final_score']}／5")
            continue
        it = items_by_id[key[0]]
        try:
            v, pts, caps, raw = base.judge(client, it, answers[key]["answer"])
        except Exception as e:
            print(f" 閱卷失敗：{e}")
            continue
        s, caps, raw_sum, final = base.score(pts, caps)
        rec = {"ID": key[0], "model": key[1], "judge": CROSS_JUDGE, **s, "raw_score": raw_sum,
               "final_score": final, "caps": ",".join(caps), "cap_reasons": v.get("cap_reasons", ""),
               "evidence": {pid: pts[pid].get("evidence", "") for pid in POINTS}, "judge_raw": raw}
        base.append_jsonl(CROSS_JSONL, rec)
        done[key] = rec
        print(f" 得分 {final}／5")
        time.sleep(base.SLEEP_SEC)
    base.MODEL_JUDGE = ORIG_JUDGE
    return done


def summarize(items, orig, cross):
    subj = {it["ID"]: it["科目"] for it in items}
    m26, m31 = base.TESTEE_MODELS
    ids = sorted(i for i in subj if all((i, m) in orig and (i, m) in cross for m in (m26, m31)))
    S = {(a, j): [((orig if j == "31B" else cross)[(i, a)])["final_score"] for i in ids]
         for a in (m26, m31) for j in ("31B", "26B")}
    L = [f"四格完整題數：{len(ids)}（作答 26B／31B × 閱卷 26B／31B）", ""]

    L.append("【1】2 × 2 平均分（滿分 5）")
    L.append("作答＼閱卷\t31B 閱卷\t26B 閱卷")
    for a in (m26, m31):
        L.append(f"{SHORT[a]} 作答\t{mean(S[(a, '31B')]):.3f}\t{mean(S[(a, '26B')]):.3f}")
    L.append("")

    L.append("【2】自我偏好（同一份回答：作答者自己閱卷 - 對方閱卷）")
    b31 = mean(S[(m31, "31B")]) - mean(S[(m31, "26B")])
    b26 = mean(S[(m26, "26B")]) - mean(S[(m26, "31B")])
    L.append(f"31B 自評偏差：{b31:+.3f} 分")
    L.append("  " + wil(S[(m31, "31B")], S[(m31, "26B")], "31B 回答：31B 閱卷 vs 26B 閱卷"))
    L.append(f"26B 自評偏差：{b26:+.3f} 分")
    L.append("  " + wil(S[(m26, "26B")], S[(m26, "31B")], "26B 回答：26B 閱卷 vs 31B 閱卷"))
    L.append(f"淨自我偏好（兩者平均）：{(b31 + b26) / 2:+.3f} 分；兩裁判寬嚴差（31B 閱卷 - 26B 閱卷，兩模型平均）："
             f"{(mean(S[(m31, '31B')]) + mean(S[(m26, '31B')]) - mean(S[(m31, '26B')]) - mean(S[(m26, '26B')])) / 2:+.3f} 分")
    L.append("")

    L.append("【3】兩位裁判一致性（同一份回答）")
    for a in (m26, m31):
        x, y = S[(a, "31B")], S[(a, "26B")]
        L.append(f"{SHORT[a]} 回答：完全一致 {mean([p == q for p, q in zip(x, y)]):.1%}，差距 ≤ 0.5 "
                 f"{mean([abs(p - q) <= 0.5 for p, q in zip(x, y)]):.1%}，加權 kappa {weighted_kappa(x, y):.3f}，"
                 f"相關係數 {pearson(x, y):.3f}")
    L.append("")

    L.append("【4】公平比較（31B vs 26B 作答）")
    L.append(f"只用 31B 閱卷（原結果）：31B {mean(S[(m31, '31B')]):.3f}，26B {mean(S[(m26, '31B')]):.3f}")
    L.append("  " + wil(S[(m26, "31B")], S[(m31, "31B")], "Wilcoxon"))
    L.append(f"只用 26B 閱卷（偏好對 26B 有利）：31B {mean(S[(m31, '26B')]):.3f}，26B {mean(S[(m26, '26B')]):.3f}")
    L.append("  " + wil(S[(m26, "26B")], S[(m31, "26B")], "Wilcoxon"))
    avg26 = [(p + q) / 2 for p, q in zip(S[(m26, "31B")], S[(m26, "26B")])]
    avg31 = [(p + q) / 2 for p, q in zip(S[(m31, "31B")], S[(m31, "26B")])]
    L.append(f"兩位裁判平均：31B {mean(avg31):.3f}，26B {mean(avg26):.3f}")
    L.append("  " + wil(avg26, avg31, "Wilcoxon"))
    L.append("")

    L.append("【5】各科（兩位裁判平均分）")
    L.append("科目\t題數\t26B\t31B\t差距")
    per = defaultdict(lambda: [[], []])
    for k, i in enumerate(ids):
        per[subj[i]][0].append(avg26[k])
        per[subj[i]][1].append(avg31[k])
    for s, (x, y) in per.items():
        L.append(f"{s}\t{len(x)}\t{mean(x):.2f}\t{mean(y):.2f}\t{mean(y) - mean(x):+.2f}")
    L.append("")
    L.append("說明：兩位裁判都對照同一份參考答案、同一份 FTRC 標準；若 26B 閱卷下 31B 仍顯著較高，"
             "表示 31B 較佳的結論不是自我偏好造成。")

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID", "科目", "26B作答_31B閱卷", "26B作答_26B閱卷", "31B作答_31B閱卷", "31B作答_26B閱卷",
                    "26B平均", "31B平均"])
        for k, i in enumerate(ids):
            w.writerow([i, subj[i], S[(m26, "31B")][k], S[(m26, "26B")][k], S[(m31, "31B")][k],
                        S[(m31, "26B")][k], avg26[k], avg31[k]])
    return "\n".join(L)


if __name__ == "__main__":
    items = base.sample_items(base.load_items())
    items_by_id = {it["ID"]: it for it in items}
    answers = base.load_done(base.ANSWERS_JSONL, lambda r: (r["ID"], r["model"]))
    orig = base.load_done(base.JUDGMENTS_JSONL, lambda r: (r["ID"], r["model"]))
    print(f"已存回答 {len(answers)} 份，31B 閱卷結果 {len(orig)} 筆")
    if "--report-only" not in sys.argv:
        cross = run_judging(items, answers)
    else:
        cross = base.load_done(CROSS_JSONL, lambda r: (r["ID"], r["model"]))
    report = summarize(items, orig, cross)
    missing = len(answers) - len(cross)
    if missing > 0:
        report += f"\n\n尚有 {missing} 筆 26B 閱卷未完成，重新執行會自動續跑。"
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n" + "=" * 50 + "\n" + report)
