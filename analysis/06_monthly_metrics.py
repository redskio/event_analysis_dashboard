"""주요 지표 5개의 월별 추이 표와 그래프를 만든다. (UTC 월 기준)

실행: 프로젝트 루트에서 python analysis/06_monthly_metrics.py
산출: outputs/tables/monthly_metrics.csv, outputs/figures/monthly_metrics.png

지표 정의는 05_key_metrics.py와 같고, 월 안에서 계산한다.
  활성 사용자: 해당 월에 이벤트가 1건 이상인 고유 방문자
  구매자: 해당 월에 transaction이 1건 이상인 고유 방문자
  구매 횟수: 해당 월 transaction 행의 고유 transactionid
  장바구니 횟수: 해당 월 addtocart 행 수
  이벤트 횟수: 해당 월 전체 행 수
월 합계의 활성 사용자·구매자는 여러 달에 걸친 방문자를 중복해서 세므로 합산하지 않는다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EVENTS = Path("data/processed/events_clean.csv")
TABLE = Path("outputs/tables/monthly_metrics.csv")
FIGURE = Path("outputs/figures/monthly_metrics.png")
START = pd.Timestamp("2015-05-03 03:00:04.384", tz="UTC")
END = pd.Timestamp("2015-09-15 21:00:00", tz="UTC")

df = pd.read_csv(EVENTS, usecols=["timestamp", "visitorid", "event", "transactionid", "year", "month"], dtype={"transactionid": "Int64"})
df["ym"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
tx = df[df["event"] == "transaction"]

monthly = pd.DataFrame(
    {
        "active_users": df.groupby("ym")["visitorid"].nunique(),
        "buyers": tx.groupby("ym")["visitorid"].nunique(),
        "purchase_count": tx.groupby("ym")["transactionid"].nunique(),
        "addtocart_count": df[df["event"] == "addtocart"].groupby("ym").size(),
        "event_count": df.groupby("ym").size(),
    }
).fillna(0).astype("int64")

# 월별 관측 일수 (데이터 시작·종료 시각으로 잘린 달은 일부만 포함)
starts = pd.to_datetime(monthly.index + "-01", utc=True)
ends = starts + pd.offsets.MonthBegin(1)
lo, hi = starts.where(starts > START, START), ends.where(ends < END, END)
monthly["observed_days"] = ((hi - lo) / pd.Timedelta(days=1)).round(2)
monthly["calendar_days"] = (ends - starts).days
monthly["is_partial_month"] = monthly["observed_days"] < monthly["calendar_days"] - 1e-9
for col in ["purchase_count", "addtocart_count", "event_count"]:
    monthly[col + "_per_day"] = (monthly[col] / monthly["observed_days"]).round(1)

TABLE.parent.mkdir(parents=True, exist_ok=True)
monthly.to_csv(TABLE, index_label="year_month", encoding="utf-8-sig")
print(monthly.T.to_string())

# 그래프: 지표마다 단위가 달라 축을 공유하지 않고 패널을 나눈다(이중 축 사용 안 함).
panels = [
    ("active_users", "활성 사용자 수(명)"),
    ("buyers", "구매자 수(명)"),
    ("purchase_count", "구매 횟수(건)"),
    ("addtocart_count", "장바구니에 담은 횟수(회)"),
    ("event_count", "이벤트 횟수(건)"),
]
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID, LINE = "#1f2933", "#616e7c", "#e8ecf1", "#2a6fbb"
labels = [f"{int(m[5:])}월" for m in monthly.index]
x = list(range(len(monthly)))
partial = monthly["is_partial_month"].tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), dpi=160)
for ax, (col, title) in zip(axes.flat, panels):
    y = monthly[col].tolist()
    ax.plot(x, y, color=LINE, linewidth=2, zorder=2)
    for i, (xi, yi, p) in enumerate(zip(x, y, partial)):
        ax.scatter(xi, yi, s=55, zorder=3, facecolor="white" if p else LINE, edgecolor=LINE, linewidth=2)
        below = i > 0 and yi < y[i - 1] * 0.7  # 급락한 점은 선과 겹치지 않게 아래에 표기
        ax.annotate(f"{yi:,}", (xi, yi), textcoords="offset points", xytext=(0, -20 if below else 9), ha="center", fontsize=9.5, color=INK)
    ax.set_title(title, loc="left", fontsize=12.5, color=INK, fontweight="bold")
    ax.set_xticks(x, labels, fontsize=10.5, color=INK)
    ax.set_xlim(-0.4, len(x) - 0.6)
    ax.set_ylim(0, max(y) * 1.2)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.tick_params(axis="y", colors=MUTED, labelsize=9, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_visible(False)
ax = axes.flat[5]
ax.axis("off")
ax.scatter([], [], s=55, facecolor=LINE, edgecolor=LINE, linewidth=2, label="한 달 전체 관측")
ax.scatter([], [], s=55, facecolor="white", edgecolor=LINE, linewidth=2, label="일부만 관측한 달")
ax.legend(loc="upper left", bbox_to_anchor=(0.05, 0.95), frameon=False, fontsize=11, labelcolor=INK, handletextpad=0.5)
partial_note = "\n".join(
    f"{int(m[5:])}월: {monthly.loc[m, 'observed_days']:.1f}일 / {monthly.loc[m, 'calendar_days']}일"
    for m in monthly.index[monthly["is_partial_month"]]
)
ax.text(0.05, 0.62, "관측 일수\n" + partial_note, transform=ax.transAxes, fontsize=10.5, color=MUTED, va="top", linespacing=1.6)
fig.suptitle("주요 지표 월별 추이 (UTC 기준)", x=0.02, ha="left", fontsize=17, fontweight="bold", color=INK)
fig.text(0.02, 0.015, "2015-05-03 03:00 ~ 2015-09-15 21:00 UTC, 완전 중복 제거. 활성 사용자·구매자는 월별 고유 방문자라 월 간 합산할 수 없음.", fontsize=9.5, color=MUTED)
fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.08, wspace=0.28, hspace=0.32)
FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE, facecolor="white")
print("저장:", FIGURE)
