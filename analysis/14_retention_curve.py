"""방문자의 N-day 리텐션 곡선을 만든다.

실행: 프로젝트 루트에서 python analysis/14_retention_curve.py
입력: data/processed/events_clean.csv (CLAUDE.md '이번 분석 기준' 적용 완료)
산출: outputs/tables/retention_nday.csv, outputs/figures/retention_nday.png

정의
- 활동: view·addtocart·transaction 중 하나라도 발생한 UTC 날짜
- Day 0: 방문자별 최초 관측 UTC 날짜. N은 Day 0과의 UTC 날짜 차이다.
- 재방문자(N): Day 0으로부터 정확히 N일째 날짜에 활동한 고유 방문자. 그 전후 날짜의 활동은 세지 않는다.
- 재방문율(N): 재방문자(N) / 관찰 가능 방문자(N) * 100

관찰 기간 보정: Day 0 + N이 마지막 관측일(2015-09-15)을 넘는 방문자는 N일째를 관찰할 수 없으므로
그 N의 분모에서 뺀다. 데이터를 제거하는 것이 아니며 N별 제외 인원은 표의 unobservable_visitors에 남긴다.

해석 주의
- 2015-05-03~05-30에 처음 관측된 방문자는 실제 신규로 단정할 수 없어(CLAUDE.md), 시작 4주와 이후 코호트를 나눈 값도 표에 함께 낸다.
- N >= 108은 분모가 시작 4주 코호트만으로 구성되어 N이 커지면서 코호트 구성이 달라진다.
- 2015-05-03은 03시 이후, 2015-09-15는 21시 이전만 포함한다. 두 날짜도 분석 기준에 따라 그대로 사용한다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path("data/processed/events_clean.csv")
TABLE = Path("outputs/tables/retention_nday.csv")
FIGURE = Path("outputs/figures/retention_nday.png")
FIRST_4WEEKS_END = pd.Timestamp("2015-05-31", tz="UTC")  # 시작 4주(05-03~05-30)의 끝 경계(미만)

df = pd.read_csv(SRC, usecols=["timestamp", "visitorid"], dtype={"timestamp": "int64", "visitorid": "int64"})
df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.floor("D")
n_events = len(df)

# 방문자-날짜 단위로 줄이고 Day 0과 N을 구한다
active = df[["visitorid", "date"]].drop_duplicates().copy()
active["day0"] = active.groupby("visitorid")["date"].transform("min")
active["n"] = (active["date"] - active["day0"]).dt.days
last_date = active["date"].max()
max_n = int(active["n"].max())


def retention(visitor_days):
    """N별 관찰 가능 방문자, 재방문자, 재방문율을 계산한다. 0~max_n 전체 N을 채운다."""
    cohort = visitor_days[visitor_days["n"] == 0].groupby("day0").size()
    days_observable = (last_date - cohort.index).days.to_numpy()  # 각 Day 0 코호트가 관찰 가능한 최대 N
    eligible = pd.Series(
        [int(cohort.to_numpy()[days_observable >= n].sum()) for n in range(max_n + 1)],
        index=pd.RangeIndex(max_n + 1, name="n"),
    )
    returned = visitor_days.groupby("n")["visitorid"].nunique().reindex(eligible.index, fill_value=0)
    return eligible, returned, returned.div(eligible.where(eligible > 0)) * 100


eligible, returned, rate = retention(active)
early = active["day0"] < FIRST_4WEEKS_END
eligible_early, returned_early, rate_early = retention(active[early])
eligible_later, returned_later, rate_later = retention(active[~early])

n_visitors = int(eligible.loc[0])
table = pd.DataFrame(
    {
        "n_day": eligible.index,
        "observable_visitors": eligible.to_numpy(),
        "unobservable_visitors": n_visitors - eligible.to_numpy(),
        "returned_visitors": returned.to_numpy(),
        "retention_pct": rate.round(4).to_numpy(),
        "first4w_observable_visitors": eligible_early.to_numpy(),
        "first4w_returned_visitors": returned_early.to_numpy(),
        "first4w_retention_pct": rate_early.round(4).to_numpy(),
        "after4w_observable_visitors": eligible_later.to_numpy(),
        "after4w_returned_visitors": returned_later.to_numpy(),
        "after4w_retention_pct": rate_later.round(4).to_numpy(),
    }
)

# 검증
assert n_visitors == df["visitorid"].nunique()
assert returned.loc[0] == n_visitors and rate.loc[0] == 100
assert (returned <= eligible).all()
assert eligible.is_monotonic_decreasing
assert returned.sum() == len(active)  # 재방문자 합계 = 방문자-날짜 수
assert (table["first4w_observable_visitors"] + table["after4w_observable_visitors"] == table["observable_visitors"]).all()
assert (table["first4w_returned_visitors"] + table["after4w_returned_visitors"] == table["returned_visitors"]).all()
only_first4w_from = int(table.loc[table["after4w_observable_visitors"] == 0, "n_day"].min())

TABLE.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(TABLE, index=False, encoding="utf-8-sig")

print("입력 행:", n_events, "| 고유 방문자:", n_visitors, "| 방문자-날짜:", len(active))
print("관측 날짜(UTC):", active["date"].min().date(), "~", last_date.date(), "| 최대 N:", max_n)
print("시작 4주 코호트:", int(eligible_early.loc[0]), "| 이후 코호트:", int(eligible_later.loc[0]))
print("분모가 시작 4주 코호트만 남는 N:", only_first4w_from, "이상")
print("Day 1 이후 재방문 경험 방문자:", active.loc[active["n"] > 0, "visitorid"].nunique())
show = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 107, 108, 120, 135]
print(table.set_index("n_day").loc[show, ["observable_visitors", "returned_visitors", "retention_pct", "first4w_retention_pct", "after4w_retention_pct"]].to_string())

# 그래프 --------------------------------------------------------------
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID, LINE, SHADE = "#1f2933", "#616e7c", "#e8ecf1", "#1f4e8c", "#f1f3f6"

fig, (axa, axb) = plt.subplots(1, 2, figsize=(15.5, 7), dpi=160, gridspec_kw={"width_ratios": [1, 1.75]})
x, y = table["n_day"].to_numpy(), table["retention_pct"].to_numpy()

# A: Day 0 포함 전체
axa.plot(x, y, color=LINE, linewidth=2)
axa.set_ylim(0, 105)
axa.set_xticks([0, 30, 60, 90, 120])
axa.annotate("Day 0  100%", (0, 100), xytext=(8, -3), textcoords="offset points", fontsize=10.5, color=INK)
axa.annotate(f"Day 1  {y[1]:.2f}%", (1, y[1]), xytext=(8, 14), textcoords="offset points", fontsize=10.5, color=INK)
axa.set_title("Day 0 포함 전체 곡선", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)

# B: Day 1 이후 확대
axb.axvspan(only_first4w_from - 0.5, max_n + 1, color=SHADE, zorder=0)
axb.text((only_first4w_from + max_n) / 2, 1.52, f"N ≥ {only_first4w_from}\n시작 4주 코호트만\n분모에 남는 구간", ha="center", va="center", fontsize=10, color=MUTED, linespacing=1.4)
axb.plot(x[1:], y[1:], color=LINE, linewidth=2)
offsets = {1: (9, -2), 2: (8, 5), 7: (9, 5), 14: (6, 9), 30: (0, 11), 60: (0, 11), 90: (0, 11)}
for n, off in offsets.items():
    axb.plot(n, y[n], "o", color=LINE, markersize=6, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    axb.annotate(f"D{n}  {y[n]:.2f}%", (n, y[n]), xytext=off, textcoords="offset points", ha="left" if n <= 14 else "center", fontsize=10.5, color=INK)
axb.set_ylim(0, 3.1)
axb.set_xticks([1, 7, 14, 30, 60, 90, 120, 135])
axb.set_title("Day 1 이후 확대 (Y축 0~3%)", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)

for ax in (axa, axb):
    ax.set_xlim(-2 if ax is axa else 0, max_n + 1)
    ax.set_xlabel("N-day (최초 관측일로부터 경과 일수, UTC)", fontsize=11, color=MUTED, labelpad=8)
    ax.set_ylabel("재방문율 (%)", fontsize=11, color=MUTED, labelpad=8)
    ax.tick_params(colors=MUTED, length=0, labelsize=10.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

fig.suptitle(
    f"N-day 리텐션: 다음 날 {y[1]:.2f}%, 7일째 {y[7]:.2f}%, 30일째 {y[30]:.2f}%가 재방문",
    x=0.02, ha="left", fontsize=16.5, fontweight="bold", color=INK,
)
fig.text(
    0.02, 0.012,
    f"방문자 {n_visitors:,}명. Day 0은 관측 기간 내 최초 관측일이며 N일째 당일에 활동한 방문자만 셉니다. 분모는 N일째를 관찰할 수 있는 방문자입니다.\n"
    f"시작 4주(05-03~05-30) 코호트는 실제 신규로 단정할 수 없고 재방문율이 더 높게 나타납니다(D1 {rate_early.loc[1]:.2f}% vs 이후 코호트 {rate_later.loc[1]:.2f}%, D30 {rate_early.loc[30]:.2f}% vs {rate_later.loc[30]:.2f}%).",
    fontsize=9.3, color=MUTED, linespacing=1.5,
)
fig.tight_layout(rect=(0, 0.075, 1, 0.95), w_pad=3)
FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE, facecolor="white")
print("저장:", TABLE, FIGURE)
