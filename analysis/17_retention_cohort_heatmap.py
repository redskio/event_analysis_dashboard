"""첫 방문일 코호트 × N-day 리텐션 히트맵을 만든다.

실행: 프로젝트 루트에서 python analysis/17_retention_cohort_heatmap.py
입력: data/processed/events_clean.csv (CLAUDE.md '이번 분석 기준' 적용 완료)
비교: outputs/tables/retention_nday.csv (N별 합계가 같아야 한다)
산출: outputs/tables/retention_cohort_daily.csv,
      outputs/figures/retention_cohort_heatmap_n30.png (모든 코호트, N=0~30),
      outputs/figures/retention_cohort_heatmap_full.png (모든 코호트, 모든 N)

정의는 14_retention_curve.py와 같다.
- 코호트(행): 방문자의 최초 관측 UTC 날짜(Day 0). 방문자는 한 행에만 속한다.
- 칸: Day 0으로부터 정확히 N일째 UTC 날짜에 이벤트(종류 무관)가 있는 고유 방문자 수와, 코호트 인원 대비 비율.
- Day 0 + N이 마지막 관측일(2015-09-15)을 넘는 칸은 관찰할 수 없으므로 비워 둔다. 0명인 칸과 구분한다.

해석 주의
- 시작 4주(05-03~05-30) 코호트의 최초 관측은 실제 첫 방문으로 단정할 수 없다.
- 05-03은 03시 이후, 09-15는 21시 이전만 관측된다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

SRC = Path("data/processed/events_clean.csv")
REFERENCE = Path("outputs/tables/retention_nday.csv")
TABLE = Path("outputs/tables/retention_cohort_daily.csv")
FIG_N30 = Path("outputs/figures/retention_cohort_heatmap_n30.png")
FIG_FULL = Path("outputs/figures/retention_cohort_heatmap_full.png")
FIRST_4WEEKS_END = pd.Timestamp("2015-05-31")  # 시작 4주(05-03~05-30)의 끝 경계(미만)
PARTIAL_DAYS = {pd.Timestamp("2015-05-03"), pd.Timestamp("2015-09-15")}
COLOR_TOP_QUANTILE = 0.99  # N >= 1 칸 재방문율의 이 분위수를 가장 진한 색으로 맞춘다
COLOR_GAMMA = 0.5  # 작은 비율 구간의 차이가 보이도록 제곱근 눈금을 쓴다

df = pd.read_csv(SRC, usecols=["timestamp", "visitorid"], dtype={"timestamp": "int64", "visitorid": "int64"})
df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()  # UTC 기준, 시간대 변환 없음
active = df[["visitorid", "date"]].drop_duplicates()
active = active.assign(day0=active.groupby("visitorid")["date"].transform("min"))
active["n"] = (active["date"] - active["day0"]).dt.days

last_date = active["date"].max()
cohort_days = pd.date_range(active["date"].min(), last_date, freq="D")
max_n = int(active["n"].max())
ns = pd.RangeIndex(max_n + 1)

returned = active.pivot_table(index="day0", columns="n", values="visitorid", aggfunc="count", fill_value=0).reindex(index=cohort_days, columns=ns, fill_value=0)
sizes = returned[0]
observable = pd.DataFrame({n: (last_date - cohort_days).days >= n for n in ns}, index=cohort_days)
rates = (returned / sizes.where(sizes > 0).to_numpy()[:, None] * 100).where(observable)
counts = returned.where(observable)

# 검증
assert sizes.sum() == df["visitorid"].nunique()
assert (returned.where(~observable, 0).to_numpy() == 0).all()  # 관찰할 수 없는 칸에 값이 없어야 한다
reference = pd.read_csv(REFERENCE, encoding="utf-8-sig").set_index("n_day")
assert (counts.sum() == reference["returned_visitors"].reindex(ns)).all()
assert (observable.mul(sizes, axis=0).sum() == reference["observable_visitors"].reindex(ns)).all()

table = (
    counts.rename_axis(index="cohort_date", columns="n_day").stack().dropna().astype("int64").rename("returned_visitors").reset_index()
    .merge(sizes.rename("cohort_visitors").rename_axis("cohort_date").reset_index(), on="cohort_date")
)
table["retention_pct"] = (table["returned_visitors"] / table["cohort_visitors"] * 100).round(4)
table["cohort_in_first_4weeks"] = table["cohort_date"] < FIRST_4WEEKS_END
table["cohort_date"] = table["cohort_date"].dt.strftime("%Y-%m-%d")
table = table[["cohort_date", "cohort_in_first_4weeks", "cohort_visitors", "n_day", "returned_visitors", "retention_pct"]]
TABLE.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(TABLE, index=False, encoding="utf-8-sig")

# 그래프 --------------------------------------------------------------
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED = "#1f2933", "#616e7c"
CMAP = LinearSegmentedColormap.from_list("cohort_blue", ["#f4f8fd", "#1f4e8c"])
CMAP.set_bad("white")
later = rates.loc[:, 1:].to_numpy()
color_top = float(np.nanquantile(later, COLOR_TOP_QUANTILE))
NORM = PowerNorm(gamma=COLOR_GAMMA, vmin=0, vmax=color_top)
overall = {n: reference.loc[n, "retention_pct"] for n in (1, 7, 30)}
early_rows = int((cohort_days < FIRST_4WEEKS_END).sum())


def row_label(day):
    return f"{day:%Y-%m-%d}" + (" *" if day in PARTIAL_DAYS else "")


def draw(path, last_n, dpi):
    """코호트 전체 × N=0~last_n 히트맵. 칸마다 재방문자 수와 재방문율을 쓴다."""
    columns = last_n + 1
    cell_w, cell_h, left, right, top, bottom = 0.62, 0.31, 1.15, 1.15, 2.0, 1.25
    width, height = left + right + columns * cell_w, top + bottom + len(cohort_days) * cell_h
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([left / width, bottom / height, columns * cell_w / width, len(cohort_days) * cell_h / height])

    shown = rates.loc[:, :last_n].to_numpy()
    ax.pcolormesh(np.arange(columns + 1), np.arange(len(cohort_days) + 1), np.ma.masked_invalid(shown), cmap=CMAP, norm=NORM, edgecolors="white", linewidth=0.6)
    ax.set_xlim(0, columns)
    ax.set_ylim(len(cohort_days), 0)

    shown_counts = counts.loc[:, :last_n].to_numpy()
    for row in range(shown.shape[0]):
        for col in range(columns):
            rate = shown[row, col]
            if np.isnan(rate):
                continue
            text_color = "white" if min(rate, color_top) ** COLOR_GAMMA > 0.55 * color_top**COLOR_GAMMA else INK
            label = f"{int(shown_counts[row, col]):,}\n{'100%' if col == 0 else f'{rate:.2f}%'}"
            ax.text(col + 0.5, row + 0.5, label, ha="center", va="center", fontsize=6.6, linespacing=1.05, color=text_color)

    # 행·열 머리글은 칸이 많아도 찾기 쉽게 양쪽에 둔다
    positions = np.arange(columns) + 0.5
    ax.set_xticks(positions, [str(n) for n in range(columns)], fontsize=8.5, color=INK)
    ax.tick_params(axis="x", length=0, labeltop=True, labelbottom=True, pad=3)
    ax.set_yticks(np.arange(len(cohort_days)) + 0.5, [row_label(d) for d in cohort_days], fontsize=8.5, color=INK)
    ax.tick_params(axis="y", length=0, labelleft=True, labelright=True, pad=4)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("N-day (첫 방문일로부터 경과 일수, UTC)", fontsize=10, color=MUTED, labelpad=6)
    ax.set_ylabel("첫 방문일 코호트 (UTC)", fontsize=10, color=MUTED, labelpad=6)

    # 시작 4주 구분선
    ax.axhline(early_rows, color=INK, linewidth=1.4)
    ax.text(columns - 0.15, early_rows - 0.12, "▲ 시작 4주 코호트: 최초 관측을 실제 첫 방문으로 단정할 수 없음", ha="right", va="bottom", fontsize=9, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))

    fig.text(
        left / width, 1 - 0.5 / height,
        f"첫 방문일 코호트별 N-day 재방문: 전체 D1 {overall[1]:.2f}%, D7 {overall[7]:.2f}%, D30 {overall[30]:.2f}%",
        ha="left", va="center", fontsize=15, fontweight="bold", color=INK,
    )
    fig.text(
        left / width, 1 - 1.05 / height,
        "칸 위쪽은 정확히 N일째에 돌아온 방문자 수, 아래쪽은 코호트 인원 대비 재방문율입니다. N=0은 코호트 인원입니다. 빈칸은 데이터 기간(2015-09-15까지)을 넘어 관찰할 수 없는 칸이고, 0명과 다릅니다.\n"
        f"색은 재방문율입니다. 작은 값의 차이가 보이도록 제곱근 눈금을 쓰고, {color_top:.2f}%(N≥1 칸의 상위 1% 값) 이상은 같은 진하기입니다. * 일부 시간만 관측된 날(05-03은 03시 이후, 09-15는 21시 이전)입니다.",
        ha="left", va="center", fontsize=9, color=MUTED, linespacing=1.5,
    )
    # 색 범례
    legend_w = min(3.2, columns * cell_w * 0.5)
    cax = fig.add_axes([(width - right - legend_w) / width, 1 - 0.58 / height, legend_w / width, 0.14 / height])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal")
    ticks = [t for t in (0, 0.5, 1, 2, 4, 8) if t <= color_top] + [color_top]
    bar.set_ticks(ticks, labels=[f"{t:g}%" for t in ticks[:-1]] + [f"{color_top:.1f}%+"])
    bar.ax.tick_params(labelsize=8, length=0, colors=MUTED, labelbottom=False, labeltop=True)
    bar.outline.set_visible(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


draw(FIG_N30, 30, 150)
draw(FIG_FULL, max_n, 120)
print("저장:", TABLE, FIG_N30, FIG_FULL)

# 요약 ----------------------------------------------------------------
print("코호트:", len(cohort_days), "| 관찰 가능 칸:", int(observable.to_numpy().sum()), "| 색 상한:", round(color_top, 3))
print("코호트 인원 최소/중앙/최대:", int(sizes.min()), int(sizes.median()), int(sizes.max()), "| 최소 코호트:", sizes.idxmin().date())
early = cohort_days < FIRST_4WEEKS_END
summary = pd.DataFrame({
    "시작 4주": [counts.loc[early, n].sum() / sizes[early & observable[n]].sum() * 100 for n in (1, 7, 14, 30)],
    "이후": [counts.loc[~early, n].sum() / sizes[~early & observable[n]].sum() * 100 for n in (1, 7, 14, 30)],
}, index=["D1", "D7", "D14", "D30"]).round(3)
print(summary.to_string())
full_rows = rates.dropna(subset=[1])[1]
print("D1 최고/최저 코호트:", full_rows.idxmax().date(), round(full_rows.max(), 2), "/", full_rows.idxmin().date(), round(full_rows.min(), 2))
d1_by_weekday = pd.DataFrame({"size": sizes, "ret": counts[1]}).dropna().groupby(lambda d: d.day_of_week).sum()
print("첫 방문 요일별 D1(월=0):", (d1_by_weekday["ret"] / d1_by_weekday["size"] * 100).round(2).to_dict())
for n in (7, 14, 30):
    ratio = rates[n].dropna()
    print(f"D{n} 코호트 중앙값 {ratio.median():.3f}%, 범위 {ratio.min():.3f}~{ratio.max():.3f}%")
print("0명 칸(N>=1) 비율:", round((counts.loc[:, 1:].to_numpy() == 0).sum() / observable.loc[:, 1:].to_numpy().sum() * 100, 2), "%")
