"""2015년 5월과 6월의 성과를 비교한다(7월 전략 기획서 근거).

실행: 프로젝트 루트에서 python analysis/18_may_june_comparison.py
입력: data/processed/events_with_category.csv, data/processed/funnel_visitors.csv,
      outputs/tables/monthly_metrics.csv, outputs/tables/retention_cohort_daily.csv
산출: outputs/tables/may_june_scale.csv        규모 지표(월 합계·일평균·상위 구매자 제외)
      outputs/tables/may_june_funnel.csv       월 내 퍼널과 첫 조회 24시간 퍼널
      outputs/tables/may_june_root.csv         대분류별 첫 조회 24시간 퍼널
      outputs/tables/may_june_category.csv     표본 충족 카테고리의 첫 조회 24시간 퍼널
      outputs/tables/may_june_item.csv         상품별 이벤트 행 수와 고유 방문자 수(6월 구매자·조회자 상위)
      outputs/tables/may_june_retention.csv    코호트 월별·구매자 재방문율
      outputs/tables/may_june_weekly.csv       온전한 주(월~일)별 지표

기준:
- 7월 이후 이벤트는 근거에서 제외한다. 5월은 05-03 03시 이후만 관측되어 일평균(관측 28.87일)을 함께 본다.
- 월 내 퍼널: 해당 월 이벤트만 사용하고 첫 view → 첫 addtocart → transaction 순서를 지킨 고유 방문자(같은 시각 인정, 상품 무관).
- 첫 조회 24시간 퍼널: funnel_visitors.csv의 전환 값을 첫 조회 월로 나눈다(5월 방문자는 대부분 시작 4주라 신규로 단정하지 않는다).
- 변화 검정: 두 비율 z검정(양측), 대분류·카테고리는 표 안에서 Benjamini-Hochberg 보정 후 q<0.05만 '유의'.
- 카테고리 미확인 비율이 5월·6월에 크게 달라 카테고리별 방문자 수는 월간 비교하지 않고 전환율만 비교한다.
- 구매자 재방문: 해당 월에 첫 구매한 방문자가 첫 구매일로부터 정확히 N일째에 활동한 비율. 분모는 N일째를 관찰할 수 있는 방문자.
"""
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = ["2015-05", "2015-06"]
LAST_DATE = pd.Timestamp("2015-09-15")
OUT = Path("outputs/tables")
MIN_VISITORS, MIN_CARTED = 300, 30  # 카테고리 비교의 월별 최소 표본


def z_test(k1, n1, k2, n2):
    """두 비율의 차이(양측) p값."""
    if min(n1, n2) == 0 or k1 + k2 == 0 or k1 + k2 == n1 + n2:
        return np.nan
    p = (k1 + k2) / (n1 + n2)
    z = (k2 / n2 - k1 / n1) / sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def bh(p):
    """Benjamini-Hochberg q값."""
    s = pd.Series(p, dtype="float64")
    o = s.dropna().sort_values()
    q = (o * len(o) / np.arange(1, len(o) + 1))[::-1].cummin()[::-1].clip(upper=1)
    return q.reindex(s.index)


e = pd.read_csv(
    "data/processed/events_with_category.csv",
    usecols=["timestamp", "visitorid", "event", "itemid", "transactionid"],
)
e["ts"] = pd.to_datetime(e["timestamp"], unit="ms")  # UTC, 시간대 변환 없음
e["date"] = e["ts"].dt.normalize()
e["month"] = e["ts"].dt.strftime("%Y-%m")
m = e[e["month"].isin(MONTHS)]
days = pd.read_csv(OUT / "monthly_metrics.csv").set_index("year_month")["observed_days"]

# ── 규모 ───────────────────────────────────────────────────
scale = []
for mon in MONTHS:
    d = m[m["month"] == mon]
    tx = d[d["event"] == "transaction"]
    per = tx.groupby("visitorid")["transactionid"].nunique()
    top10 = per.nlargest(10)
    rest = per.drop(top10.index)
    scale.append({
        "month": mon, "observed_days": days[mon], "active_users": d["visitorid"].nunique(), "buyers": len(per),
        "purchase_count": tx["transactionid"].nunique(), "addtocart_count": int((d["event"] == "addtocart").sum()),
        "event_count": len(d), "purchases_per_buyer": per.mean(), "repeat_buyers": int((per >= 2).sum()),
        "repeat_buyer_pct": (per >= 2).mean() * 100, "repeat_purchase_share_pct": per[per >= 2].sum() / per.sum() * 100,
        "top10_purchase_share_pct": top10.sum() / per.sum() * 100, "buyers_excl_top10": len(rest),
        "purchases_excl_top10": int(rest.sum()), "repeat_buyer_pct_excl_top10": (rest >= 2).mean() * 100,
    })
scale = pd.DataFrame(scale).set_index("month")
for c in ["active_users", "buyers", "purchase_count", "addtocart_count", "event_count", "purchases_excl_top10"]:
    scale[f"{c}_per_day"] = scale[c] / scale["observed_days"]
scale.T.to_csv(OUT / "may_june_scale.csv", encoding="utf-8-sig")

# ── 퍼널 ───────────────────────────────────────────────────
funnel = []
for mon in MONTHS:
    d = m[m["month"] == mon]
    first_view = d[d["event"] == "view"].groupby("visitorid")["timestamp"].min().rename("v")
    cart = d[d["event"] == "addtocart"].merge(first_view, on="visitorid")
    first_cart = cart[cart["timestamp"] >= cart["v"]].groupby("visitorid")["timestamp"].min().rename("c")
    buy = d[d["event"] == "transaction"].merge(first_cart, on="visitorid")
    buy = buy[buy["timestamp"] >= buy["c"]]
    first_buy = buy.groupby("visitorid")["timestamp"].min().rename("b")
    gap = pd.concat([first_cart, first_buy], axis=1, join="inner")
    gap_h = (gap["b"] - gap["c"]) / 3_600_000
    funnel.append({"basis": "월 내", "month": mon, "view": len(first_view), "carted": len(first_cart), "bought": len(first_buy),
                   "buy_within_1h_of_cart_pct": (gap_h <= 1).mean() * 100})
f = pd.read_csv("data/processed/funnel_visitors.csv", dtype={"root_categoryid": "Int64", "categoryid": "Int64"})
f["month"] = f["first_view_date"].str[:7]
f = f[f["month"].isin(MONTHS) & f["observable_24h"]]
for mon, g in f.groupby("month"):
    funnel.append({"basis": "첫 조회 24시간", "month": mon, "view": len(g), "carted": int(g["converted_cart_24h"].sum()),
                   "bought": int(g["converted_buy_24h"].sum()), "unknown_category_pct": g["categoryid"].isna().mean() * 100})
funnel = pd.DataFrame(funnel)
funnel["view_to_cart_pct"] = funnel["carted"] / funnel["view"] * 100
funnel["cart_to_purchase_pct"] = funnel["bought"] / funnel["carted"] * 100
funnel["view_to_purchase_pct"] = funnel["bought"] / funnel["view"] * 100
funnel["cart_abandoned"] = funnel["carted"] - funnel["bought"]
for basis, g in funnel.groupby("basis"):
    a, b = g.iloc[0], g.iloc[1]
    funnel.loc[g.index[1], "p_view_to_cart"] = z_test(a["carted"], a["view"], b["carted"], b["view"])
    funnel.loc[g.index[1], "p_cart_to_purchase"] = z_test(a.bought, a.carted, b.bought, b.carted)
    funnel.loc[g.index[1], "p_view_to_purchase"] = z_test(a["bought"], a["view"], b["bought"], b["view"])
funnel.to_csv(OUT / "may_june_funnel.csv", index=False, encoding="utf-8-sig")


# ── 대분류·카테고리 ─────────────────────────────────────────
def compare(keys, data):
    t = data.groupby(keys + ["month"]).agg(
        visitors=("visitorid", "size"), carted=("converted_cart_24h", "sum"), bought=("converted_buy_24h", "sum")
    ).unstack("month", fill_value=0)
    t.columns = [f"{a}_{b[-2:]}" for a, b in t.columns]
    t = t[(t["visitors_05"] >= MIN_VISITORS) & (t["visitors_06"] >= MIN_VISITORS) & (t["carted_05"] >= MIN_CARTED) & (t["carted_06"] >= MIN_CARTED)].copy()
    for mm in ["05", "06"]:
        t[f"view_to_purchase_{mm}"] = t[f"bought_{mm}"] / t[f"visitors_{mm}"] * 100
        t[f"cart_to_purchase_{mm}"] = t[f"bought_{mm}"] / t[f"carted_{mm}"] * 100
    t["p_vp"] = [z_test(r.bought_05, r.visitors_05, r.bought_06, r.visitors_06) for r in t.itertuples()]
    t["p_cp"] = [z_test(r.bought_05, r.carted_05, r.bought_06, r.carted_06) for r in t.itertuples()]
    t["q_vp"], t["q_cp"] = bh(t["p_vp"]).values, bh(t["p_cp"]).values
    t["vp_change"] = np.where(t["q_vp"] < 0.05, "유의", "차이 불확실")
    t["cp_change"] = np.where(t["q_cp"] < 0.05, "유의", "차이 불확실")
    return t.sort_values("visitors_06", ascending=False).reset_index()


known = f[f["categoryid"].notna()]
root = compare(["root_categoryid"], known)
root.to_csv(OUT / "may_june_root.csv", index=False, encoding="utf-8-sig")
cat = compare(["root_categoryid", "categoryid"], known)
cat.to_csv(OUT / "may_june_category.csv", index=False, encoding="utf-8-sig")

# ── 상품 ───────────────────────────────────────────────────
it = m.groupby(["itemid", "month", "event"]).size().unstack("event", fill_value=0).unstack("month", fill_value=0)
it.columns = [f"{a}_{b[-2:]}" for a, b in it.columns]
uv = m.groupby(["itemid", "month", "event"])["visitorid"].nunique().unstack("event", fill_value=0).unstack("month", fill_value=0)
uv.columns = [f"{a}_visitors_{b[-2:]}" for a, b in uv.columns]  # 행 수는 한 방문자의 반복 구매에 흔들리므로 고유 방문자를 판단 기준으로 쓴다
it = it.join(uv)
top = pd.concat([it.nlargest(20, "transaction_visitors_06"), it.nlargest(20, "view_visitors_06")]).drop_duplicates()
top.reset_index().to_csv(OUT / "may_june_item.csv", index=False, encoding="utf-8-sig")

# ── 재방문 ─────────────────────────────────────────────────
rows = []
c = pd.read_csv(OUT / "retention_cohort_daily.csv")
c["month"] = c["cohort_date"].str[:7]
for (mon, n), g in c[c["month"].isin(MONTHS) & c["n_day"].isin([1, 7, 30])].groupby(["month", "n_day"]):
    rows.append({"group": "전체 코호트", "month": mon, "n_day": n, "denominator": int(g["cohort_visitors"].sum()), "returned": int(g["returned_visitors"].sum())})
active = e[["visitorid", "date"]].drop_duplicates()
first_buy = e[e["event"] == "transaction"].groupby("visitorid")["date"].min().rename("first_buy").reset_index()
first_buy["month"] = first_buy["first_buy"].dt.strftime("%Y-%m")
first_buy = first_buy[first_buy["month"].isin(MONTHS)]
keys = pd.MultiIndex.from_frame(active)
for n in [1, 7, 30]:
    target = first_buy["first_buy"] + pd.Timedelta(days=n)
    ok = target <= LAST_DATE
    hit = pd.MultiIndex.from_arrays([first_buy["visitorid"], target], names=["visitorid", "date"]).isin(keys)
    for mon in MONTHS:
        s = first_buy["month"] == mon
        rows.append({"group": "첫 구매자", "month": mon, "n_day": n, "denominator": int((s & ok).sum()), "returned": int((s & ok & hit).sum())})
ret = pd.DataFrame(rows)
ret["retention_pct"] = ret["returned"] / ret["denominator"] * 100
for (grp, n), g in ret.groupby(["group", "n_day"]):
    a, b = g.iloc[0], g.iloc[1]
    ret.loc[g.index[1], "p_vs_may"] = z_test(a.returned, a.denominator, b.returned, b.denominator)
ret.to_csv(OUT / "may_june_retention.csv", index=False, encoding="utf-8-sig")

# ── 주간 ───────────────────────────────────────────────────
w = e[(e["date"] >= "2015-05-04") & (e["date"] <= "2015-06-28")].copy()
w["week_start"] = (w["date"] - pd.to_timedelta(w["date"].dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d")
tx = w[w["event"] == "transaction"]
weekly = pd.DataFrame({
    "active_users": w.groupby("week_start")["visitorid"].nunique(), "view_count": w[w["event"] == "view"].groupby("week_start").size(),
    "addtocart_count": w[w["event"] == "addtocart"].groupby("week_start").size(), "buyers": tx.groupby("week_start")["visitorid"].nunique(),
    "purchase_count": tx.groupby("week_start")["transactionid"].nunique(),
})
tx_month = tx.copy()
tx_month["month"] = tx_month["date"].dt.strftime("%Y-%m")
top10_tx = []
for mon, g in tx_month.groupby("month"):  # 각 월의 상위 구매자 10명은 그 월 전체(주 경계 밖 포함) 구매 횟수로 고른다
    month_tx = m[(m["month"] == mon) & (m["event"] == "transaction")]
    top = month_tx.groupby("visitorid")["transactionid"].nunique().nlargest(10).index
    top10_tx.append(g[g["visitorid"].isin(top)])
top10_tx = pd.concat(top10_tx)
weekly["purchases_month_top10_buyers"] = top10_tx.groupby("week_start")["transactionid"].nunique()
weekly["purchases_other_buyers"] = weekly["purchase_count"] - weekly["purchases_month_top10_buyers"]
weekly = weekly.fillna(0).reset_index()
weekly.to_csv(OUT / "may_june_weekly.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 80)
print(scale.T.round(2)); print(funnel.round(4).T)
show = ["visitors_05", "visitors_06", "bought_05", "bought_06", "view_to_purchase_05", "view_to_purchase_06", "q_vp", "vp_change", "cart_to_purchase_05", "cart_to_purchase_06", "q_cp", "cp_change"]
print(root[["root_categoryid"] + show].round(3)); print(len(cat)); print(cat[(cat["q_vp"] < .05) | (cat["q_cp"] < .05)][["root_categoryid", "categoryid"] + show].round(3))
print(ret.round(4)); print(weekly)
