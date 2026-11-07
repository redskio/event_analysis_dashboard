"""대분류·카테고리별 24시간 퍼널 성과를 집계하고 비교한다.

실행: 프로젝트 루트에서 python analysis/13_funnel_by_category.py
입력: data/processed/funnel_visitors.csv (12_funnel_visitors.py)
산출: outputs/tables/funnel24h_by_root.csv, outputs/tables/funnel24h_by_category.csv

집계 기준: 첫 조회 시점의 카테고리로 방문자를 나누고(카테고리 미확인은 별도), 24시간 이내 전환율을 계산한다.
관찰 가능 방문자(첫 조회가 종료 24시간 전 이전)만 사용한다. 전환은 방문자 단위로 상품·카테고리와 무관하게 센다.
전환율 격차는 Wilson 95% 신뢰구간으로 전체 평균과 비교한다(구간이 전체 평균을 벗어나면 '유의').
"""
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/processed/funnel_visitors.csv")
OUT_ROOT = Path("outputs/tables/funnel24h_by_root.csv")
OUT_CAT = Path("outputs/tables/funnel24h_by_category.csv")
MIN_MS = 60_000

df = pd.read_csv(SRC, dtype={"root_categoryid": "Int64", "categoryid": "Int64"})
d = df[df["observable_24h"]].copy()
d["v2c_min"] = d["view_to_cart_ms"].where(d["converted_cart_24h"]) / MIN_MS
d["c2b_min"] = d["cart_to_buy_ms"].where(d["converted_buy_24h"]) / MIN_MS
d["views_before_cart_c"] = d["views_before_cart"].astype("float").where(d["converted_cart_24h"])
d["carts_before_buy_c"] = d["carts_before_buy"].astype("float").where(d["converted_buy_24h"])


def wilson(k, n, z=1.96):
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    p = np.divide(k, n, out=np.zeros_like(n), where=n > 0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def summarize(g, keys):
    s = g.groupby(keys).agg(
        visitors=("visitorid", "size"),
        carted=("converted_cart_24h", "sum"),
        bought=("converted_buy_24h", "sum"),
        median_view_to_cart_min=("v2c_min", "median"),
        median_cart_to_buy_min=("c2b_min", "median"),
        mean_views_before_cart=("views_before_cart_c", "mean"),
        median_views_before_cart=("views_before_cart_c", "median"),
        mean_carts_before_buy=("carts_before_buy_c", "mean"),
    ).reset_index()
    s["view_to_cart_pct"] = s["carted"] / s["visitors"] * 100
    s["cart_to_purchase_pct"] = np.where(s["carted"] > 0, s["bought"] / s["carted"].where(s["carted"] > 0) * 100, np.nan)
    s["view_to_purchase_pct"] = s["bought"] / s["visitors"] * 100
    return s


def add_flags(s, overall):
    """전체 평균과의 격차(%p)와 Wilson 95% 구간 기준 유의 여부."""
    for name, k, n, ov in [("view_to_cart", "carted", "visitors", overall["v2c"]), ("cart_to_purchase", "bought", "carted", overall["c2b"]), ("view_to_purchase", "bought", "visitors", overall["v2b"])]:
        lo, hi = wilson(s[k], s[n].where(s[n] > 0, 1))
        s[f"{name}_ci_low"] = lo * 100
        s[f"{name}_ci_high"] = hi * 100
        s[f"{name}_gap_pp"] = s[f"{name}_pct"] - ov * 100
        s[f"{name}_vs_overall"] = np.select([s[f"{name}_ci_low"] > ov * 100, s[f"{name}_ci_high"] < ov * 100], ["높음", "낮음"], "차이 불확실")
    return s


tot = {"v": len(d), "c": int(d["converted_cart_24h"].sum()), "b": int(d["converted_buy_24h"].sum())}
overall = {"v2c": tot["c"] / tot["v"], "c2b": tot["b"] / tot["c"], "v2b": tot["b"] / tot["v"]}
print(f"전체(관찰 가능 방문자 {tot['v']:,}명): 장바구니 {tot['c']:,}, 구매 {tot['b']:,} | View→장바구니 {overall['v2c'] * 100:.2f}% · 장바구니→구매 {overall['c2b'] * 100:.2f}% · View→구매 {overall['v2b'] * 100:.2f}%")
print(f"전체 중앙값: View→장바구니 {d['v2c_min'].median():.1f}분, 장바구니→구매 {d['c2b_min'].median():.1f}분 | 평균 조회 {d['views_before_cart_c'].mean():.2f}회, 평균 장바구니 {d['carts_before_buy_c'].mean():.2f}회")

d["root"] = d["root_categoryid"].fillna(-1).astype(int)  # -1: 카테고리 미확인
root = add_flags(summarize(d, ["root"]), overall).sort_values("visitors", ascending=False)
root["visitor_share_pct"] = root["visitors"] / tot["v"] * 100
cat = add_flags(summarize(d[d["categoryid"].notna()].assign(cat=lambda x: x["categoryid"].astype(int), root=lambda x: x["root_categoryid"].astype(int)), ["root", "cat"]), overall)
cat = cat.sort_values("visitors", ascending=False)
cat["visitor_share_pct"] = cat["visitors"] / tot["v"] * 100

OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
root.rename(columns={"root": "root_categoryid"}).round(3).to_csv(OUT_ROOT, index=False, encoding="utf-8-sig")
cat.rename(columns={"root": "root_categoryid", "cat": "categoryid"}).round(3).to_csv(OUT_CAT, index=False, encoding="utf-8-sig")
assert root["visitors"].sum() == tot["v"] and root["carted"].sum() == tot["c"] and root["bought"].sum() == tot["b"]
assert cat["visitors"].sum() == d["categoryid"].notna().sum()

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 100)
show = ["root", "visitors", "visitor_share_pct", "view_to_cart_pct", "view_to_cart_vs_overall", "cart_to_purchase_pct", "cart_to_purchase_vs_overall", "view_to_purchase_pct", "view_to_purchase_vs_overall"]
print("\n[대분류별 24시간 퍼널] (root=-1: 카테고리 미확인)")
print(root[show].round(2).to_string(index=False))
print("\n[대분류별 소요 시간·횟수]")
print(root[["root", "visitors", "carted", "bought", "median_view_to_cart_min", "median_cart_to_buy_min", "mean_views_before_cart", "median_views_before_cart", "mean_carts_before_buy"]].round(2).to_string(index=False))

big = cat[cat["visitors"] >= 1000]
print(f"\n[카테고리] 전체 {len(cat)}개, 방문자 1,000명 이상 {len(big)}개")
cols = ["root", "cat", "visitors", "view_to_cart_pct", "cart_to_purchase_pct", "view_to_purchase_pct", "view_to_purchase_vs_overall", "carted", "bought"]
print("View→구매 상위 8:\n", big.sort_values("view_to_purchase_pct", ascending=False)[cols].head(8).round(2).to_string(index=False))
print("View→구매 하위 8:\n", big.sort_values("view_to_purchase_pct")[cols].head(8).round(2).to_string(index=False))
print("View→장바구니 상위/하위 5:\n", big.sort_values("view_to_cart_pct", ascending=False)[cols].head(5).round(2).to_string(index=False), "\n", big.sort_values("view_to_cart_pct")[cols].head(5).round(2).to_string(index=False))
bc = big[big["carted"] >= 100]
print("장바구니→구매 상위/하위 5 (장바구니 100명 이상):\n", bc.sort_values("cart_to_purchase_pct", ascending=False)[cols].head(5).round(2).to_string(index=False), "\n", bc.sort_values("cart_to_purchase_pct")[cols].head(5).round(2).to_string(index=False))
k = root[(root["root"] != -1) & (root["visitors"] >= 1000)]
print("\n대분류 간 상관(View→장바구니 vs 장바구니→구매):", round(k["view_to_cart_pct"].corr(k["cart_to_purchase_pct"]), 2))
print("카테고리 수 중 구매 전환 '높음'/'낮음'(1,000명 이상):", big["view_to_purchase_vs_overall"].value_counts().to_dict())
print("시작 4주 첫 조회 방문자 비중(대분류 미확인 vs 확인):", round(d.loc[d["root"] == -1, "first_view_in_first_4weeks"].mean() * 100, 1), round(d.loc[d["root"] != -1, "first_view_in_first_4weeks"].mean() * 100, 1))
