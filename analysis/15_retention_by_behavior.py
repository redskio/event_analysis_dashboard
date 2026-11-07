"""구매·장바구니 담기·둘 다 한 사용자의 N-day 재방문율을 비교한다.

실행: 프로젝트 루트에서 python analysis/15_retention_by_behavior.py
입력: data/processed/events_clean.csv (CLAUDE.md '이번 분석 기준' 적용 완료)
산출: outputs/tables/retention_by_behavior.csv, outputs/figures/retention_by_behavior.png

행동군 (서로 겹친다. 예: 장바구니 사용자에는 구매자도 포함된다)
- 구매: transaction 1회 이상. Day 0 = 첫 transaction UTC 날짜
- 장바구니 담기: addtocart 1회 이상. Day 0 = 첫 addtocart UTC 날짜
- 장바구니+구매: 둘 다 1회 이상(순서 무관). Day 0 = 첫 addtocart와 첫 transaction 중 늦은 UTC 날짜(두 조건을 다 채운 날)

Day 0을 최초 관측일이 아니라 해당 행동일로 잡는 이유: 최초 관측일로 잡으면 나중에 구매한 사용자는
구매일에 이미 '재방문'한 것이 되어, 행동군에 들어가는 조건 자체가 재방문율을 부풀린다.

재방문자(N): Day 0으로부터 정확히 N일째 날짜에 전체 이벤트(view·addtocart·transaction) 중 하나라도 있는 고유 방문자
재방문율(N): 재방문자(N) / 관찰 가능 방문자(N) * 100. 분모는 Day 0 + N이 마지막 관측일(2015-09-15) 이내인 방문자다.
95% 신뢰구간은 Wilson 방식이다.

해석 주의
- 2015-05-03~05-30의 행동은 이전 기록이 없어 실제 첫 구매·첫 담기로 단정할 수 없다. 시작 4주와 이후 Day 0을 나눈 값을 표에 함께 낸다.
- N >= 108은 분모에 시작 4주 Day 0만 남는다.
- 구매 행이 50행 이상인 방문자(도매 또는 테스트 계정 여부 미확인)는 제외하지 않으며, 이들을 뺀 민감도 값을 표에 함께 낸다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path("data/processed/events_clean.csv")
KEY_METRICS = Path("outputs/tables/key_metrics.csv")
TABLE = Path("outputs/tables/retention_by_behavior.csv")
FIGURE = Path("outputs/figures/retention_by_behavior.png")
FIRST_4WEEKS_END = pd.Timestamp("2015-05-31", tz="UTC")  # 시작 4주(05-03~05-30)의 끝 경계(미만)
HEAVY_BUYER_ROWS = 50
Z95 = 1.959964

df = pd.read_csv(SRC, usecols=["timestamp", "visitorid", "event"], dtype={"timestamp": "int64", "visitorid": "int64", "event": "str"})
df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.floor("D")
last_date = df["date"].max()
max_n = (last_date - df["date"].min()).days
active = df[["visitorid", "date"]].drop_duplicates()

first_buy = df[df["event"] == "transaction"].groupby("visitorid")["date"].min()
first_cart = df[df["event"] == "addtocart"].groupby("visitorid")["date"].min()
both = pd.concat([first_cart.rename("cart"), first_buy.rename("buy")], axis=1, join="inner")
GROUPS = {"구매": first_buy, "장바구니 담기": first_cart, "장바구니+구매": both.max(axis=1)}

buy_rows = df[df["event"] == "transaction"].groupby("visitorid").size()
heavy = buy_rows[buy_rows >= HEAVY_BUYER_ROWS].index


def retention(day0):
    """Day 0 시리즈(index=visitorid)로 N별 관찰 가능 방문자와 재방문자를 계산한다. 0~max_n 전체를 채운다."""
    index = pd.RangeIndex(max_n + 1, name="n_day")
    joined = active.merge(day0.rename("day0"), left_on="visitorid", right_index=True)
    joined["n"] = (joined["date"] - joined["day0"]).dt.days
    returned = joined[joined["n"] >= 0].groupby("n")["visitorid"].nunique().reindex(index, fill_value=0)
    observable_until = (last_date - day0).dt.days.to_numpy()
    eligible = pd.Series([int((observable_until >= n).sum()) for n in index], index=index)
    return eligible, returned


def pct(returned, eligible):
    return (returned / eligible.where(eligible > 0) * 100).round(4)


def wilson(returned, eligible):
    p = returned / eligible.where(eligible > 0)
    denom = 1 + Z95**2 / eligible
    center = (p + Z95**2 / (2 * eligible)) / denom
    half = Z95 * np.sqrt(p * (1 - p) / eligible + Z95**2 / (4 * eligible**2)) / denom
    return ((center - half) * 100).round(4), ((center + half) * 100).round(4)


frames = []
for name, day0 in GROUPS.items():
    e, r = retention(day0)
    low, high = wilson(r, e)
    early = day0 < FIRST_4WEEKS_END
    e_early, r_early = retention(day0[early])
    e_later, r_later = retention(day0[~early])
    e_light, r_light = retention(day0[~day0.index.isin(heavy)])

    # 검증
    assert e.loc[0] == r.loc[0] == len(day0)
    assert (r <= e).all() and e.is_monotonic_decreasing
    assert (e_early + e_later == e).all() and (r_early + r_later == r).all()

    frames.append(
        pd.DataFrame(
            {
                "group": name,
                "group_visitors": len(day0),
                "n_day": e.index,
                "observable_visitors": e.to_numpy(),
                "unobservable_visitors": len(day0) - e.to_numpy(),
                "returned_visitors": r.to_numpy(),
                "retention_pct": pct(r, e).to_numpy(),
                "ci95_low_pct": low.to_numpy(),
                "ci95_high_pct": high.to_numpy(),
                "first4w_observable_visitors": e_early.to_numpy(),
                "first4w_retention_pct": pct(r_early, e_early).to_numpy(),
                "after4w_observable_visitors": e_later.to_numpy(),
                "after4w_retention_pct": pct(r_later, e_later).to_numpy(),
                "excl_heavy_buyers_retention_pct": pct(r_light, e_light).to_numpy(),
            }
        )
    )
table = pd.concat(frames, ignore_index=True)

metrics = pd.read_csv(KEY_METRICS, encoding="utf-8-sig")
buyers_ref = int(metrics.loc[metrics["metric"].str.startswith("구매자 수"), "event_table"].iloc[0])
assert len(first_buy) == buyers_ref
assert set(both.index) == set(first_buy.index) & set(first_cart.index)
only_first4w_from = table[table["after4w_observable_visitors"] == 0].groupby("group")["n_day"].min()
assert only_first4w_from.nunique() == 1
only_first4w_from = int(only_first4w_from.iloc[0])

TABLE.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(TABLE, index=False, encoding="utf-8-sig")

print("행동군 인원:", {k: len(v) for k, v in GROUPS.items()})
print("장바구니 사용자 중 구매자:", f"{len(both) / len(first_cart) * 100:.2f}%", "| 구매자 중 장바구니 사용자:", f"{len(both) / len(first_buy) * 100:.2f}%")
print("첫 담기·첫 구매 날짜 관계(장바구니+구매):", {"담기가 먼저": int((both["cart"] < both["buy"]).sum()), "같은 날": int((both["cart"] == both["buy"]).sum()), "구매가 먼저": int((both["cart"] > both["buy"]).sum())})
print(f"구매 {HEAVY_BUYER_ROWS}행 이상 방문자:", len(heavy), "| 분모가 시작 4주 Day 0만 남는 N:", only_first4w_from, "이상")
SHOW = [1, 2, 3, 7, 14, 30, 60, 90]
cols = ["observable_visitors", "returned_visitors", "retention_pct", "ci95_low_pct", "ci95_high_pct", "first4w_retention_pct", "after4w_retention_pct", "excl_heavy_buyers_retention_pct"]
print(table[table["n_day"].isin(SHOW)].set_index(["group", "n_day"])[cols].to_string())

# 그래프 --------------------------------------------------------------
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID, SHADE = "#1f2933", "#616e7c", "#e8ecf1", "#f1f3f6"
STYLE = {
    "구매": {"color": "#2a78d6", "linestyle": "-"},
    "장바구니 담기": {"color": "#eb6834", "linestyle": "-"},
    "장바구니+구매": {"color": "#1baf7a", "linestyle": (0, (3, 2))},  # 구매와 거의 겹쳐 점선으로 구분
}
wide = table.pivot(index="n_day", columns="group", values="retention_pct")

fig = plt.figure(figsize=(15.5, 9), dpi=160)
grid = fig.add_gridspec(2, 2, height_ratios=[3.4, 1], width_ratios=[1, 1.35], hspace=0.3, wspace=0.16, left=0.055, right=0.985, top=0.87, bottom=0.085)
axa, axb, axt = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])

for name, style in STYLE.items():
    label = f"{name} ({len(GROUPS[name]):,}명)"
    axa.plot(wide.loc[1:30].index, wide.loc[1:30, name], linewidth=2, label=label, **style)
    axb.plot(wide.loc[30:].index, wide.loc[30:, name], linewidth=1.6, **style)

axb.axvspan(only_first4w_from - 0.5, max_n + 1, color=SHADE, zorder=0)
top_b = np.ceil(wide.loc[30:].max().max() * 2 + 0.5) / 2
axb.text((only_first4w_from + max_n) / 2, top_b * 0.9, f"N ≥ {only_first4w_from}: 시작 4주 Day 0만\n분모에 남는 구간", ha="center", va="center", fontsize=10, color=MUTED, linespacing=1.4)
axa.set_ylim(0, 13)
axb.set_ylim(0, top_b)
axa.set_xlim(0, 30.5)
axb.set_xlim(30, max_n + 1)
axa.set_xticks([1, 7, 14, 21, 28, 30])
axb.set_xticks([30, 60, 90, 120, 135])
axa.set_title("N = 1~30", loc="left", fontsize=13, fontweight="bold", color=INK, pad=10)
axb.set_title("N = 30~135 (Y축 확대, 재방문자가 적어 변동이 큼)", loc="left", fontsize=13, fontweight="bold", color=INK, pad=10)
axa.legend(loc="upper right", frameon=False, fontsize=11.5, labelcolor=INK, title="행동군 (Day 0 = 해당 행동일)", title_fontsize=10.5)
for ax in (axa, axb):
    ax.set_xlabel("N-day (Day 0으로부터 경과 일수, UTC)", fontsize=11, color=MUTED, labelpad=6)
    ax.set_ylabel("재방문율 (%)", fontsize=11, color=MUTED, labelpad=6)
    ax.tick_params(colors=MUTED, length=0, labelsize=10.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

axt.axis("off")
cells = [[f"{wide.loc[n, name]:.2f}%" for n in SHOW] for name in STYLE]
tbl = axt.table(cellText=cells, rowLabels=list(STYLE), colLabels=[f"D{n}" for n in SHOW], cellLoc="center", rowLoc="left", bbox=[0.13, 0, 0.87, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor(GRID)
    cell.set_linewidth(0.8)
    cell.visible_edges = "horizontal"
    cell.get_text().set_color(MUTED if row == 0 else INK)

v = wide.loc
fig.suptitle(
    f"구매 사용자의 재방문율이 장바구니 사용자보다 높음: D1 {v[1, '구매']:.2f}% vs {v[1, '장바구니 담기']:.2f}%, D7 {v[7, '구매']:.2f}% vs {v[7, '장바구니 담기']:.2f}%, D30 {v[30, '구매']:.2f}% vs {v[30, '장바구니 담기']:.2f}%",
    x=0.02, y=0.975, ha="left", fontsize=15.5, fontweight="bold", color=INK,
)
fig.text(
    0.02, 0.012,
    f"재방문은 Day 0으로부터 정확히 N일째 날짜의 전체 이벤트 기준이며 Day 0(100%)은 그리지 않았습니다. 분모는 N일째를 관찰할 수 있는 방문자입니다. 행동군은 서로 겹칩니다(장바구니 사용자의 {len(both) / len(first_cart) * 100:.1f}%가 구매자, 구매자의 {len(both) / len(first_buy) * 100:.1f}%가 장바구니 사용자).\n"
    "시작 4주(05-03~05-30)의 행동은 실제 첫 구매·첫 담기로 단정할 수 없습니다. 이 비교는 관찰된 집단 차이이며, 구매나 담기가 재방문을 늘렸다는 인과 근거는 아닙니다.",
    fontsize=9.3, color=MUTED, linespacing=1.5,
)
FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE, facecolor="white")
print("저장:", TABLE, FIGURE)
