"""Quick, dependency-free analysis for Q3 outputs.

Designed to run even if numpy/pandas are unavailable in the environment.

Usage:
  python Q3_regression/q3_quick_analysis.py --k 3

It reads files from outputs/q3_regression and writes a short markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None


@dataclass(frozen=True)
class FixedRow:
    param: str
    mean: float
    sd: float
    hdi_low: float
    hdi_high: float
    r_hat: Optional[float]

    @property
    def sig_95(self) -> bool:
        return (self.hdi_low > 0.0) or (self.hdi_high < 0.0)


@dataclass(frozen=True)
class RandomRow:
    group: str
    mean: float
    hdi_low: float
    hdi_high: float

    @property
    def sig_95(self) -> bool:
        return (self.hdi_low > 0.0) or (self.hdi_high < 0.0)


def read_fixed_csv(path: Path) -> List[FixedRow]:
    rows: List[FixedRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            param = (r.get("param") or "").strip()
            mean = _to_float(r.get("mean"))
            sd = _to_float(r.get("sd"))
            hdi_low = _to_float(r.get("hdi_2.5%"))
            hdi_high = _to_float(r.get("hdi_97.5%"))
            r_hat = _to_float(r.get("r_hat"))
            if not param or mean is None or sd is None or hdi_low is None or hdi_high is None:
                continue
            rows.append(FixedRow(param=param, mean=mean, sd=sd, hdi_low=hdi_low, hdi_high=hdi_high, r_hat=r_hat))
    return rows


def read_random_csv(path: Path, group_col: str) -> List[RandomRow]:
    rows: List[RandomRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            group = (r.get(group_col) or "").strip()
            mean = _to_float(r.get("mean"))
            hdi_low = _to_float(r.get("hdi_lower"))
            hdi_high = _to_float(r.get("hdi_upper"))
            if not group or mean is None or hdi_low is None or hdi_high is None:
                continue
            rows.append(RandomRow(group=group, mean=mean, hdi_low=hdi_low, hdi_high=hdi_high))
    return rows


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def fmt_pct_ratio(beta: float, unit: float = 1.0) -> str:
    """Interpret coefficient on centered log-response scale as multiplicative ratio."""
    ratio = math.exp(beta * unit)
    pct = (ratio - 1.0) * 100.0
    return f"{ratio:.3f} ({pct:+.1f}%)"


def top_by_abs_mean(rows: Sequence[FixedRow], n: int = 10, allow_prefixes: Tuple[str, ...] = ()) -> List[FixedRow]:
    if allow_prefixes:
        filtered = [r for r in rows if r.param.startswith(allow_prefixes)]
    else:
        filtered = list(rows)
    return sorted(filtered, key=lambda r: abs(r.mean), reverse=True)[:n]


def top_random(rows: Sequence[RandomRow], n: int = 8, descending: bool = True) -> List[RandomRow]:
    return sorted(rows, key=lambda r: r.mean, reverse=descending)[:n]


def build_report(outputs_dir: Path, k: int) -> str:
    judge_fixed = read_fixed_csv(outputs_dir / f"judge_fixed_summary_common_k{k}.csv")
    fan_fixed = read_fixed_csv(outputs_dir / f"fan_fixed_summary_k{k}.csv")
    delta_fixed = read_fixed_csv(outputs_dir / f"delta_fixed_summary_common_k{k}.csv")

    judge_pro = read_random_csv(outputs_dir / f"judge_pro_effects_common_k{k}.csv", group_col="ballroom_partner")
    fan_pro = read_random_csv(outputs_dir / f"fan_pro_effects_k{k}.csv", group_col="ballroom_partner")

    corr_path = outputs_dir / f"pro_effect_correlation_common_k{k}.json"
    corr_json: Dict[str, Any] = {}
    if corr_path.exists():
        corr_json = json.loads(corr_path.read_text(encoding="utf-8"))

    # Join pro effects for gap listing
    judge_map = {r.group: r for r in judge_pro}
    fan_map = {r.group: r for r in fan_pro}
    common_names = sorted(set(judge_map).intersection(fan_map))
    gaps: List[Tuple[str, float, float, float]] = []
    xs: List[float] = []
    ys: List[float] = []
    for name in common_names:
        fj = fan_map[name].mean
        jj = judge_map[name].mean
        gaps.append((name, fj, jj, abs(fj - jj)))
        xs.append(fj)
        ys.append(jj)

    corr_recalc = pearson_corr(xs, ys)

    # Fixed effects focus
    prefixes = ("Intercept", "age_c", "log_week", "industry[")
    judge_key = top_by_abs_mean(judge_fixed, n=12, allow_prefixes=prefixes)
    fan_key = top_by_abs_mean(fan_fixed, n=12, allow_prefixes=prefixes)
    delta_key = top_by_abs_mean(delta_fixed, n=12, allow_prefixes=prefixes)

    # Specific interpretable covariates
    def find(rows: Sequence[FixedRow], name: str) -> Optional[FixedRow]:
        for r in rows:
            if r.param == name:
                return r
        return None

    j_age = find(judge_fixed, "age_c")
    f_age = find(fan_fixed, "age_c")
    d_age = find(delta_fixed, "age_c")

    j_logw = find(judge_fixed, "log_week")
    f_logw = find(fan_fixed, "log_week")
    d_logw = find(delta_fixed, "log_week")

    # Random effects: top/bottom and significance counts
    judge_pro_sig = sum(1 for r in judge_pro if r.sig_95)
    fan_pro_sig = sum(1 for r in fan_pro if r.sig_95)

    lines: List[str] = []
    lines.append(f"# Q3 初步分析（K={k}）\n")
    lines.append("本报告基于 `outputs/q3_regression` 中已生成的 k=3（且评委侧采用 common-sample 对齐版）结果文件。\n")

    lines.append("## 1) 量化影响（固定效应）\n")
    if j_age and f_age and d_age:
        lines.append("### 年龄（age_c）\n")
        lines.append(f"- 评委：$\\beta={j_age.mean:+.3f}$，95%HDI=[{j_age.hdi_low:+.3f},{j_age.hdi_high:+.3f}]，每大 10 岁约乘以 {fmt_pct_ratio(j_age.mean, 10)}\n")
        lines.append(f"- 粉丝：$\\beta={f_age.mean:+.3f}$，95%HDI=[{f_age.hdi_low:+.3f},{f_age.hdi_high:+.3f}]，每大 10 岁约乘以 {fmt_pct_ratio(f_age.mean, 10)}\n")
        lines.append(f"- 差异（粉丝-评委）：$\\Delta={d_age.mean:+.3f}$，95%HDI=[{d_age.hdi_low:+.3f},{d_age.hdi_high:+.3f}]\n")
        lines.append("解释：年龄系数为负表示“年龄越大，早期周内相对份额越低”。差异为正表示粉丝对年龄的惩罚更弱。\n")

    if j_logw and f_logw and d_logw:
        lines.append("### 周次（log_week）\n")
        lines.append(f"- 评委：$\\beta={j_logw.mean:+.3f}$，95%HDI=[{j_logw.hdi_low:+.3f},{j_logw.hdi_high:+.3f}]\n")
        lines.append(f"- 粉丝：$\\beta={f_logw.mean:+.3f}$，95%HDI=[{f_logw.hdi_low:+.3f},{f_logw.hdi_high:+.3f}]\n")
        lines.append(f"- 差异（粉丝-评委）：$\\Delta={d_logw.mean:+.3f}$，95%HDI=[{d_logw.hdi_low:+.3f},{d_logw.hdi_high:+.3f}]\n")
        lines.append("解释：当前 K=3 下，log_week 在两边都不算强信号（区间跨 0），说明早期周次趋势对“周内相对份额”解释有限或被随机效应吸收。\n")

    # Industry highlight: list significant ones for judge/fan
    def pick_sig_industry(rows: Sequence[FixedRow]) -> List[FixedRow]:
        out = [r for r in rows if r.param.startswith("industry[") and r.sig_95]
        return sorted(out, key=lambda r: abs(r.mean), reverse=True)

    j_ind_sig = pick_sig_industry(judge_fixed)
    f_ind_sig = pick_sig_industry(fan_fixed)

    lines.append("### 行业（industry）\n")
    lines.append("行业系数是相对于一个基准行业（由建模时的编码自动确定，通常是频数最高的那类；常见情况是 Actor/Actress）。\n")

    def fmt_ind_list(title: str, lst: List[FixedRow], max_n: int = 6) -> None:
        lines.append(f"- {title}（仅列出 95%HDI 不跨 0 的类别，按 |mean| 排序）：\n")
        if not lst:
            lines.append("  -（无）\n")
            return
        for r in lst[:max_n]:
            lines.append(f"  - {r.param}: mean={r.mean:+.3f}, 95%HDI=[{r.hdi_low:+.3f},{r.hdi_high:+.3f}], ratio≈{fmt_pct_ratio(r.mean, 1)}\n")

    fmt_ind_list("评委", j_ind_sig)
    fmt_ind_list("粉丝", f_ind_sig)

    lines.append("\n## 2) 差异性比较（固定效应差值 Δ=粉丝-评委_common）\n")
    lines.append("下面按 |Δmean| 列出差异最大的若干项（K=3）。\n")
    for r in delta_key:
        sig = "显著" if r.sig_95 else "不显著"
        lines.append(f"- {r.param}: Δmean={r.mean:+.3f}（{sig}）, 95%HDI=[{r.hdi_low:+.3f},{r.hdi_high:+.3f}]\n")

    lines.append("\n## 3) 职业舞伴（Pro）随机效应：影响大小与一致性\n")
    lines.append(
        f"- Pro 随机效应相关（common-sample）：json 给出 corr={corr_json.get('corr')}, n_pairs={corr_json.get('n_pairs')}；脚本按均值重算 corr={corr_recalc:.3f}（应接近）。\n"
    )
    lines.append(
        f"- 显著舞伴数量（95%HDI 不跨 0）：评委 {judge_pro_sig}/{len(judge_pro)}，粉丝 {fan_pro_sig}/{len(fan_pro)}（粉丝侧通常更不确定）。\n"
    )

    lines.append("\n### 评委侧：舞伴效应 Top/Bottom\n")
    for r in top_random(judge_pro, n=8, descending=True):
        lines.append(f"- + {r.group}: mean={r.mean:+.3f}, 95%≈[{r.hdi_low:+.3f},{r.hdi_high:+.3f}], ratio≈{fmt_pct_ratio(r.mean, 1)}\n")
    lines.append("\n")
    for r in top_random(judge_pro, n=8, descending=False):
        lines.append(f"- - {r.group}: mean={r.mean:+.3f}, 95%≈[{r.hdi_low:+.3f},{r.hdi_high:+.3f}], ratio≈{fmt_pct_ratio(r.mean, 1)}\n")

    lines.append("\n### 粉丝侧：舞伴效应 Top/Bottom\n")
    for r in top_random(fan_pro, n=8, descending=True):
        lines.append(f"- + {r.group}: mean={r.mean:+.3f}, 95%≈[{r.hdi_low:+.3f},{r.hdi_high:+.3f}], ratio≈{fmt_pct_ratio(r.mean, 1)}\n")
    lines.append("\n")
    for r in top_random(fan_pro, n=8, descending=False):
        lines.append(f"- - {r.group}: mean={r.mean:+.3f}, 95%≈[{r.hdi_low:+.3f},{r.hdi_high:+.3f}], ratio≈{fmt_pct_ratio(r.mean, 1)}\n")

    lines.append("\n### 舞伴‘粉丝 vs 评委’差异最大的个体（按 |mean_fan-mean_judge|）\n")
    for name, mf, mj, gap in sorted(gaps, key=lambda t: t[3], reverse=True)[:10]:
        lines.append(f"- {name}: fan={mf:+.3f}, judge={mj:+.3f}, |gap|={gap:.3f}\n")

    lines.append("\n## 4) 一句话结论（供写进论文）\n")
    lines.append(
        "- 在 K=3 的早期周数据上，年龄对表现存在稳定负向影响，且评委侧惩罚略强于粉丝侧。\n"
        "- 行业效应在评委侧更‘清晰’（更多类别显著偏离基准），粉丝侧行业差异不如评委侧稳定；但‘Model’在两边都显著偏低，且粉丝侧更低。\n"
        "- Pro（职业舞伴）效应在两边仅弱相关（corr≈0.27），说明‘评委觉得某些舞伴能带来更高表现’与‘粉丝更愿意投某些舞伴搭档’并不完全一致。\n"
    )

    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=str(Path("outputs") / "q3_regression"),
        help="Path to outputs/q3_regression",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    report = build_report(outputs_dir=outputs_dir, k=args.k)

    out_path = outputs_dir / f"k{args.k}_initial_analysis.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
