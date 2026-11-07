"""view → addtocart → transaction 퍼널과 단계별 전환율 (CLAUDE.md '이번 분석 기준' 적용).

실행: 프로젝트 루트에서 python analysis/02_funnel.py
입력: data/processed/events_clean.csv (UTC, 완전 중복 제거, 2015-09-15 21:00 UTC 미만, 시작 4주 포함)
산출: outputs/tables/funnel_conversion.csv, outputs/figures/funnel_view_cart_purchase.png

주 기준(방문자·순서 기준): 고유 visitorid, 상품 무관.
  1단계 view: view가 1건 이상인 방문자
  2단계 addtocart: 최초 view 시각 이후(동일 시각 포함) addtocart가 있는 방문자
  3단계 transaction: 위 addtocart 최초 시각 이후(동일 시각 포함) transaction이 있는 방문자
단계별 전환율은 '앞 단계를 수행한 방문자 중 다음 단계를 수행한 방문자'의 비율이다.
시작 4주(2015-05-03~05-30)는 제외하지 않고 포함하되, 그 이전 행동이 관측되지 않으므로
첫 조회가 시작 4주 안인 방문자와 이후인 방문자를 나누어 비교값을 함께 낸다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SRC = Path("data/processed/events_clean.csv")
TABLE = Path("outputs/tables/funnel_conversion.csv")
FIGURE = Path("outputs/figures/funnel_view_cart_purchase.png")
FIRST_4WEEKS_END_MS = int(pd.Timestamp("2015-05-31", tz="UTC").timestamp() * 1000)
STEPS = ["view", "addtocart", "transaction"]

df = pd.read_csv(SRC, usecols=["timestamp", "visitorid", "event", "itemid"], dtype={"visitorid": "int64", "itemid": "int64", "event": "str"})
assert not df.duplicated().any()


def ordered_funnel(d, keys):
    """keys 단위로 view → addtocart → transaction 순서를 만족한 시각을 단계별로 돌려준다(없으면 결측)."""
    view = d[d["event"] == "view"].groupby(keys, as_index=False)["timestamp"].min().rename(columns={"timestamp": "t_view"})
    cart = d[d["event"] == "addtocart"].merge(view, on=keys)
    cart = cart[cart["timestamp"] >= cart["t_view"]].groupby(keys, as_index=False)["timestamp"].min().rename(columns={"timestamp": "t_cart"})
    buy = d[d["event"] == "transaction"].merge(cart, on=keys)
    buy = buy[buy["timestamp"] >= buy["t_cart"]].groupby(keys, as_index=False)["timestamp"].min().rename(columns={"timestamp": "t_buy"})
    return view.merge(cart, on=keys, how="left").merge(buy, on=keys, how="left")


def visitor_counts(per):
    return [per["visitorid"].nunique(), per.loc[per["t_cart"].notna(), "visitorid"].nunique(), per.loc[per["t_buy"].notna(), "visitorid"].nunique()]


def row(name, counts):
    v, c, b = counts
    return {
        "basis": name,
        "view": v,
        "addtocart": c,
        "transaction": b,
        "view_to_cart_pct": round(c / v * 100, 2),
        "cart_to_purchase_pct": round(b / c * 100, 2),
        "view_to_purchase_pct": round(b / v * 100, 2),
        "view_dropoff": v - c,
        "cart_dropoff": c - b,
    }


main_per = ordered_funnel(df, ["visitorid"])
early = main_per["t_view"] < FIRST_4WEEKS_END_MS  # 첫 조회가 시작 4주 안인 방문자
same_item_per = ordered_funnel(df, ["visitorid", "itemid"])
unordered = [df.loc[df["event"] == e, "visitorid"].nunique() for e in STEPS]

result = pd.DataFrame(
    [
        row("주 기준: 방문자, 순서 충족, 상품 무관", visitor_counts(main_per)),
        row("첫 조회가 시작 4주(05-03~05-30) 안인 방문자", visitor_counts(main_per[early])),
        row("첫 조회가 시작 4주 이후인 방문자", visitor_counts(main_per[~early])),
        row("방문자, 순서 충족, 같은 상품", visitor_counts(same_item_per)),
        row("방문자, 순서 무시(이벤트별 고유 방문자)", unordered),
    ]
)
TABLE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(TABLE, index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(result.drop(columns=["view_dropoff", "cart_dropoff"]).to_string())

# 검증: 첫 조회 이후 장바구니 방문자 수는 10_first_view_table.py의 값과 같아야 한다(파일이 있을 때).
fv = Path("data/processed/first_view_visitors.csv")
if fv.exists():
    carted = int(pd.read_csv(fv, usecols=["carted_after_first_view"])["carted_after_first_view"].sum())
    print("교차검증: 첫 조회 이후 장바구니 방문자(10번 테이블)", carted, "| 퍼널 2단계", int(result.iloc[0]["addtocart"]))
assert result.iloc[1]["view"] + result.iloc[2]["view"] == result.iloc[0]["view"]

# 그래프: 선형 축을 유지하고 수치와 단계별 전환율을 직접 표기한다.
main = result.iloc[0]
counts = [int(main[e]) for e in STEPS]
labels = ["View", "장바구니 담기", "구매"]
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, BAR, TRACK = "#1f2933", "#616e7c", "#2a6fbb", "#e8ecf1"

fig, ax = plt.subplots(figsize=(10, 5.4), dpi=160)
y = [2, 1, 0]
ax.barh(y, [counts[0]] * 3, color=TRACK, height=0.5)
ax.barh(y, counts, color=BAR, height=0.5)
ax.set_yticks(y, labels, fontsize=13, color=INK)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, counts[0] * 1.02)
ax.set_ylim(-0.75, 2.7)
ax.xaxis.set_major_formatter(lambda v, _: f"{v / 10000:,.0f}만")
ax.tick_params(axis="x", colors=MUTED, length=0)
ax.grid(axis="x", color=TRACK, linewidth=0.8)
ax.set_axisbelow(True)
for side in ax.spines.values():
    side.set_visible(False)
for pos, value in zip(y, counts):
    ax.text(counts[0] * 0.015, pos + 0.33, f"{value:,}명 (View 대비 {value / counts[0] * 100:.2f}%)", fontsize=11, color=INK, va="bottom")
box = dict(facecolor="white", edgecolor="none", pad=3)
ax.text(counts[0] * 0.5, 1.48, f"▼ View → 장바구니 전환율 {main['view_to_cart_pct']:.2f}%  (이탈 {int(main['view_dropoff']):,}명)", ha="center", va="center", fontsize=11.5, color=INK, fontweight="bold", bbox=box)
ax.text(counts[0] * 0.5, 0.48, f"▼ 장바구니 → 구매 전환율 {main['cart_to_purchase_pct']:.2f}%  (이탈 {int(main['cart_dropoff']):,}명)", ha="center", va="center", fontsize=11.5, color=INK, fontweight="bold", bbox=box)
fig.suptitle(f"View 방문자 중 구매까지 이어진 비율은 {main['view_to_purchase_pct']:.2f}%", x=0.02, ha="left", fontsize=16, fontweight="bold", color=INK)
fig.text(
    0.02, 0.02,
    "고유 방문자 기준, 앞 단계 이후 발생한 행동만 집계(상품 무관). 2015-05-03 03:00 ~ 2015-09-15 21:00 UTC, 완전 중복 제거, 시작 4주 포함.\n"
    "단계별 전환율 = 앞 단계를 수행한 방문자 중 다음 단계를 수행한 방문자의 비율.",
    fontsize=9, color=MUTED, linespacing=1.5,
)
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE, facecolor="white")
print("저장:", FIGURE, "|", TABLE)
