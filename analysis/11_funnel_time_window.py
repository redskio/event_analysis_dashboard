"""조회부터 구매까지 걸린 시간을 24시간·7일로 제한한 퍼널을 만들고 제한 없는 퍼널과 비교한다.

실행: 프로젝트 루트에서 python analysis/11_funnel_time_window.py
입력: data/processed/events_clean.csv (CLAUDE.md '이번 분석 기준' 적용 완료)
산출: outputs/tables/funnel_time_window.csv, outputs/tables/funnel_time_to_convert.csv,
      outputs/figures/funnel_time_window.png

퍼널 정의는 02_funnel.py와 같다(고유 visitorid, 상품 무관, 순서 충족, 동일 시각은 이후로 인정).
  t_view: 첫 view / t_cart: t_view 이후 첫 addtocart / t_buy: t_cart 이후 첫 transaction
시간 제한 W: 모든 단계가 첫 view로부터 W 이내여야 한다(t_cart - t_view <= W, t_buy - t_view <= W).
제한 없는 퍼널은 W가 없는 02_funnel.py의 주 기준과 같다.

관찰 기간 보정: 데이터가 2015-09-15 21:00 UTC에서 끝나므로, 늦게 첫 조회한 방문자는 7일을 다 관찰할 수 없다.
주 비교는 첫 조회가 종료 시각 7일 전(2015-09-08 21:00 UTC) 이전인 방문자('성숙 코호트')로 세 퍼널을 모두 계산한다.
같은 방문자 집단이어야 제한 없음·7일·24시간이 공정하게 비교된다. 전체 방문자 기준은 참고용으로 함께 낸다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path("data/processed/events_clean.csv")
TABLE = Path("outputs/tables/funnel_time_window.csv")
TIME_TABLE = Path("outputs/tables/funnel_time_to_convert.csv")
FIGURE = Path("outputs/figures/funnel_time_window.png")
HOUR_MS, DAY_MS = 3_600_000, 86_400_000
DATA_END_MS = int(pd.Timestamp("2015-09-15 21:00", tz="UTC").timestamp() * 1000)
MATURE_CUT_MS = DATA_END_MS - 7 * DAY_MS  # 이 시각까지 첫 조회한 방문자는 7일을 끝까지 관찰할 수 있다
SCENARIOS = [("제한 없음", None), ("7일 이내", 7 * DAY_MS), ("24시간 이내", DAY_MS)]

df = pd.read_csv(SRC, usecols=["timestamp", "visitorid", "event"], dtype={"visitorid": "int64", "event": "str"})

view = df[df["event"] == "view"].groupby("visitorid", as_index=False)["timestamp"].min().rename(columns={"timestamp": "t_view"})
cart = df[df["event"] == "addtocart"].merge(view, on="visitorid")
cart = cart[cart["timestamp"] >= cart["t_view"]].groupby("visitorid", as_index=False)["timestamp"].min().rename(columns={"timestamp": "t_cart"})
buy = df[df["event"] == "transaction"].merge(cart, on="visitorid")
buy = buy[buy["timestamp"] >= buy["t_cart"]].groupby("visitorid", as_index=False)["timestamp"].min().rename(columns={"timestamp": "t_buy"})
per = view.merge(cart, on="visitorid", how="left").merge(buy, on="visitorid", how="left")
per["view_to_cart_ms"] = per["t_cart"] - per["t_view"]
per["view_to_buy_ms"] = per["t_buy"] - per["t_view"]
mature = per["t_view"] <= MATURE_CUT_MS


def funnel(p, window):
    c = p["t_cart"].notna() if window is None else p["view_to_cart_ms"] <= window
    b = p["t_buy"].notna() if window is None else p["view_to_buy_ms"] <= window
    v, ca, bu = len(p), int(c.sum()), int(b.sum())
    return v, ca, bu


rows = []
for cohort, p in [("성숙 코호트(첫 조회 ~2015-09-08 21:00 UTC)", per[mature]), ("전체 방문자(참고)", per)]:
    for name, window in SCENARIOS:
        v, ca, bu = funnel(p, window)
        rows.append(
            {
                "cohort": cohort, "scenario": name, "view": v, "addtocart": ca, "transaction": bu,
                "view_to_cart_pct": round(ca / v * 100, 2),
                "cart_to_purchase_pct": round(bu / ca * 100, 2),
                "view_to_purchase_pct": round(bu / v * 100, 2),
            }
        )
result = pd.DataFrame(rows)
TABLE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(TABLE, index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(result.to_string(index=False))

# 소요 시간 분포(성숙 코호트에서 제한 없이 전환한 방문자)
m = per[mature]
tt = []
for label, col, p in [("View → 장바구니", "view_to_cart_ms", m[m["t_cart"].notna()]), ("View → 구매", "view_to_buy_ms", m[m["t_buy"].notna()])]:
    s = p[col]
    tt.append(
        {
            "stage": label, "visitors": len(s),
            "median_hours": round(s.median() / HOUR_MS, 2), "p75_hours": round(s.quantile(0.75) / HOUR_MS, 2), "p90_hours": round(s.quantile(0.9) / HOUR_MS, 2),
            "within_1h_pct": round((s <= HOUR_MS).mean() * 100, 1), "within_24h_pct": round((s <= DAY_MS).mean() * 100, 1),
            "within_7d_pct": round((s <= 7 * DAY_MS).mean() * 100, 1), "after_7d_pct": round((s > 7 * DAY_MS).mean() * 100, 1),
        }
    )
time_table = pd.DataFrame(tt)
time_table.to_csv(TIME_TABLE, index=False, encoding="utf-8-sig")
print("\n소요 시간(성숙 코호트, 제한 없이 전환한 방문자)\n", time_table.to_string(index=False))
print("\n성숙 코호트 방문자:", int(mature.sum()), "/ 전체", len(per), "| 제외(첫 조회가 마감 7일 이내):", int((~mature).sum()))

# 검증: 제한 없음 전체 방문자는 02_funnel.py 주 기준과 같아야 한다.
base = result[(result["cohort"] == "전체 방문자(참고)") & (result["scenario"] == "제한 없음")].iloc[0]
funnel_csv = Path("outputs/tables/funnel_conversion.csv")
if funnel_csv.exists():
    ref = pd.read_csv(funnel_csv).iloc[0]
    print("교차검증(02_funnel 주 기준):", [int(ref["view"]), int(ref["addtocart"]), int(ref["transaction"])], "vs", [int(base["view"]), int(base["addtocart"]), int(base["transaction"])])
    assert [int(ref["view"]), int(ref["addtocart"]), int(ref["transaction"])] == [int(base["view"]), int(base["addtocart"]), int(base["transaction"])]
for _, g in result.groupby("cohort"):  # 제한이 강할수록 값이 줄어야 한다
    assert g["addtocart"].is_monotonic_decreasing and g["transaction"].is_monotonic_decreasing

# 그래프 --------------------------------------------------------------
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID = "#1f2933", "#616e7c", "#e8ecf1"
COLORS = {"제한 없음": "#1f4e8c", "7일 이내": "#5b97d3", "24시간 이내": "#b3d0ee"}  # 제한이 짧을수록 연한 파랑
main = result[result["cohort"].str.startswith("성숙")].set_index("scenario")
names = [n for n, _ in SCENARIOS]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(15.5, 7), dpi=160, gridspec_kw={"width_ratios": [1.05, 1]})

# A: 단계별 방문자 수
steps = ["View", "장바구니 담기", "구매"]
cols = ["view", "addtocart", "transaction"]
h = 0.26
for si, step_col in enumerate(cols):
    for ni, name in enumerate(names):
        ypos = (2 - si) + (1 - ni) * h
        val = int(main.loc[name, step_col])
        axa.barh(ypos, val, height=h * 0.9, color=COLORS[name])
        pct = val / int(main.loc[name, "view"]) * 100
        axa.text(val + main["view"].max() * 0.012, ypos, f"{val:,}명 ({pct:.2f}%)" if si else f"{val:,}명", va="center", fontsize=9.5, color=INK)
axa.set_yticks([2, 1, 0], steps, fontsize=12.5, color=INK)
axa.tick_params(axis="y", length=0)
axa.set_xlim(0, main["view"].max() * 1.36)
axa.xaxis.set_major_formatter(lambda v, _: f"{v / 10000:,.0f}만")
axa.tick_params(axis="x", colors=MUTED, length=0)
axa.grid(axis="x", color=GRID, linewidth=0.8)
axa.set_axisbelow(True)
axa.set_title("단계별 방문자 수 (View 대비 %)", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)

# B: 단계별 전환율
rate_cols = [("view_to_cart_pct", "View → 장바구니"), ("cart_to_purchase_pct", "장바구니 → 구매"), ("view_to_purchase_pct", "View → 구매")]
w = 0.26
for gi, (col, _) in enumerate(rate_cols):
    for ni, name in enumerate(names):
        x = gi + (ni - 1) * w
        val = float(main.loc[name, col])
        axb.bar(x, val, width=w * 0.92, color=COLORS[name], label=name if gi == 0 else None)
        axb.text(x, val + 0.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=9.5, color=INK)
axb.set_xticks(range(3), [lab for _, lab in rate_cols], fontsize=12, color=INK)
axb.set_ylim(0, max(float(main[c].max()) for c, _ in rate_cols) * 1.15)
axb.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
axb.tick_params(axis="y", colors=MUTED, length=0)
axb.tick_params(axis="x", length=0)
axb.grid(axis="y", color=GRID, linewidth=0.8)
axb.set_axisbelow(True)
axb.set_title("단계별 전환율", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
axb.legend(loc="upper right", frameon=False, fontsize=11, labelcolor=INK, title="조회 후 소요 시간", title_fontsize=10.5)
for ax in (axa, axb):
    for s in ax.spines.values():
        s.set_visible(False)

b = main.loc["제한 없음", "view_to_purchase_pct"]
fig.suptitle(
    f"View 방문자 중 구매 전환율: 제한 없음 {b:.2f}% · 7일 이내 {main.loc['7일 이내', 'view_to_purchase_pct']:.2f}% · 24시간 이내 {main.loc['24시간 이내', 'view_to_purchase_pct']:.2f}%",
    x=0.02, ha="left", fontsize=16.5, fontweight="bold", color=INK,
)
fig.text(
    0.02, 0.012,
    f"성숙 코호트 {int(main.loc['제한 없음', 'view']):,}명 기준(첫 조회가 2015-09-08 21:00 UTC 이전이라 7일을 끝까지 관찰 가능). 고유 방문자, 순서 충족(View→장바구니→구매), 상품 무관. "
    "시간 제한은 첫 View부터 각 단계까지의 시간이며 모든 단계에 적용. UTC, 완전 중복 제거, 시작 4주 포함.",
    fontsize=9.3, color=MUTED,
)
fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.08, wspace=0.28)
FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE, facecolor="white")
print("저장:", FIGURE, "|", TABLE, "|", TIME_TABLE)
