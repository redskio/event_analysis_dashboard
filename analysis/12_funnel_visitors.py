"""방문자별 퍼널 테이블(24시간 전환 기준)을 만든다. 대시보드 '퍼널' 탭과 13번 분석의 입력이다.

실행: 프로젝트 루트에서 python analysis/12_funnel_visitors.py
입력: data/processed/events_clean.csv, data/processed/first_view_visitors.csv(첫 조회 카테고리)
산출: data/processed/funnel_visitors.csv (view가 있는 방문자 1명당 1행)

퍼널 정의는 02_funnel.py와 같다(고유 visitorid, 순서 충족, 동일 시각은 이후로 인정).
  t_view: 첫 view / t_cart: t_view 이후 첫 addtocart / t_buy: t_cart 이후 첫 transaction
24시간 기준: 첫 view로부터 각 단계까지 24시간 이내여야 전환으로 본다(11_funnel_time_window.py와 같은 기준).
  converted_cart_24h : t_cart - t_view <= 24h
  converted_buy_24h  : t_buy  - t_view <= 24h (t_cart 이후 구매, 장바구니도 24h 이내)
  observable_24h     : t_view <= 데이터 종료(2015-09-15 21:00 UTC) - 24h. 이후 첫 조회자는 24시간을 끝까지 관찰할 수 없다.
시간·횟수:
  view_to_cart_ms, cart_to_buy_ms  각 전환까지 걸린 시간(밀리초)
  views_before_cart  첫 view부터 t_cart까지(양끝 포함) view 이벤트 수
  carts_before_buy   t_cart부터 t_buy까지(양끝 포함) addtocart 이벤트 수
카테고리는 첫 조회 시점에 적용된 값이며(결측 가능) 방문자 단위로 상품·카테고리 무관하게 전환을 센다.
"""
from pathlib import Path

import pandas as pd

EVENTS = Path("data/processed/events_clean.csv")
FIRST_VIEW = Path("data/processed/first_view_visitors.csv")
OUT = Path("data/processed/funnel_visitors.csv")
DAY_MS = 86_400_000
DATA_END_MS = int(pd.Timestamp("2015-09-15 21:00", tz="UTC").timestamp() * 1000)

df = pd.read_csv(EVENTS, usecols=["timestamp", "visitorid", "event"], dtype={"visitorid": "int64", "event": "str"})
views = df[df["event"] == "view"]
per = views.groupby("visitorid", as_index=False)["timestamp"].min().rename(columns={"timestamp": "t_view"})

cart_ev = df[df["event"] == "addtocart"][["visitorid", "timestamp"]].merge(per, on="visitorid")
cart_ev = cart_ev[cart_ev["timestamp"] >= cart_ev["t_view"]][["visitorid", "timestamp"]]
per = per.merge(cart_ev.groupby("visitorid", as_index=False)["timestamp"].min().rename(columns={"timestamp": "t_cart"}), on="visitorid", how="left")

buy_ev = df[df["event"] == "transaction"][["visitorid", "timestamp"]].merge(per.loc[per["t_cart"].notna(), ["visitorid", "t_cart"]], on="visitorid")
buy_ev = buy_ev[buy_ev["timestamp"] >= buy_ev["t_cart"]]
per = per.merge(buy_ev.groupby("visitorid", as_index=False)["timestamp"].min().rename(columns={"timestamp": "t_buy"}), on="visitorid", how="left")

# 단계 직전까지의 이전 단계 수행 횟수
v = views[["visitorid", "timestamp"]].merge(per.loc[per["t_cart"].notna(), ["visitorid", "t_cart"]], on="visitorid")
views_before_cart = v[v["timestamp"] <= v["t_cart"]].groupby("visitorid").size().rename("views_before_cart")
c = cart_ev.merge(per.loc[per["t_buy"].notna(), ["visitorid", "t_cart", "t_buy"]], on="visitorid")
carts_before_buy = c[(c["timestamp"] >= c["t_cart"]) & (c["timestamp"] <= c["t_buy"])].groupby("visitorid").size().rename("carts_before_buy")
per = per.merge(views_before_cart, on="visitorid", how="left").merge(carts_before_buy, on="visitorid", how="left")

per["view_to_cart_ms"] = per["t_cart"] - per["t_view"]
per["cart_to_buy_ms"] = per["t_buy"] - per["t_cart"]
per["converted_cart_24h"] = per["view_to_cart_ms"] <= DAY_MS
per["converted_buy_24h"] = (per["t_buy"] - per["t_view"]) <= DAY_MS
per["observable_24h"] = per["t_view"] <= DATA_END_MS - DAY_MS

fv = pd.read_csv(FIRST_VIEW, usecols=["visitorid", "first_view_timestamp", "first_view_date", "first_view_in_first_4weeks",
                                      "categoryid", "root_categoryid", "category_match_status"], dtype={"categoryid": "Int64", "root_categoryid": "Int64"})
per = per.merge(fv, on="visitorid", how="left", validate="one_to_one")
assert per["first_view_timestamp"].eq(per["t_view"]).all()  # 첫 조회 테이블과 같은 첫 view

out = per[["visitorid", "first_view_timestamp", "first_view_date", "first_view_in_first_4weeks", "root_categoryid", "categoryid",
           "category_match_status", "view_to_cart_ms", "cart_to_buy_ms", "views_before_cart", "carts_before_buy",
           "converted_cart_24h", "converted_buy_24h", "observable_24h"]].copy()
for col in ["view_to_cart_ms", "cart_to_buy_ms", "views_before_cart", "carts_before_buy"]:
    out[col] = out[col].astype("Int64")

# 검증
assert out["visitorid"].is_unique and len(out) == views["visitorid"].nunique()
assert (out["converted_buy_24h"] <= out["converted_cart_24h"]).all()
assert out["views_before_cart"].notna().eq(per["t_cart"].notna()).all() and out["carts_before_buy"].notna().eq(per["t_buy"].notna()).all()
assert (out["views_before_cart"].dropna() >= 1).all() and (out["carts_before_buy"].dropna() >= 1).all()

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
o = out[out["observable_24h"]]
print("행:", len(out), "| 24시간 관찰 가능:", len(o), f"| 제외(마지막 24시간 첫 조회): {len(out) - len(o)}")
print("제한 없음 퍼널:", len(out), int(per["t_cart"].notna().sum()), int(per["t_buy"].notna().sum()))
print("24시간 퍼널(관찰 가능 방문자):", len(o), int(o["converted_cart_24h"].sum()), int(o["converted_buy_24h"].sum()),
      f"| 전환율 {o['converted_cart_24h'].mean() * 100:.2f}% / {o['converted_buy_24h'].sum() / o['converted_cart_24h'].sum() * 100:.2f}% / {o['converted_buy_24h'].mean() * 100:.2f}%")
print("조회 횟수(장바구니 전환자) 평균/중앙값:", round(out["views_before_cart"].mean(), 2), out["views_before_cart"].median(),
      "| 장바구니 횟수(구매 전환자) 평균/중앙값:", round(out["carts_before_buy"].mean(), 2), out["carts_before_buy"].median())
