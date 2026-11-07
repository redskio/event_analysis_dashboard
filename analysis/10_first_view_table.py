"""방문자별 첫 상품 조회 테이블(대시보드 '카테고리별 첫 조회 성과' 탭의 입력)을 만든다.

실행: 프로젝트 루트에서 python analysis/10_first_view_table.py
입력: data/processed/events_with_category.csv
산출: data/processed/first_view_visitors.csv (view가 있는 방문자 1명당 1행)

첫 상품 조회: 방문자의 가장 이른 view 1건(같은 시각이면 파일 순서의 첫 행). 카테고리는 그 시점에 적용된 값.
성과 지표(방문자 단위, 카테고리·상품과 무관하게 계산):
  carted_after_first_view     첫 조회 시각 이후(같은 시각 포함) addtocart가 1건 이상 있는 방문자
  purchased_after_first_view  첫 조회 시각 이후(같은 시각 포함) transaction이 1건 이상 있는 방문자
"""
from pathlib import Path

import pandas as pd

SRC = Path("data/processed/events_with_category.csv")
OUT = Path("data/processed/first_view_visitors.csv")
FIRST_4WEEKS_END_MS = int(pd.Timestamp("2015-05-31", tz="UTC").timestamp() * 1000)

df = pd.read_csv(
    SRC,
    usecols=["timestamp", "visitorid", "event", "itemid", "categoryid", "root_categoryid", "category_match_status"],
    dtype={"categoryid": "Int64", "root_categoryid": "Int64"},
)
assert df["timestamp"].is_monotonic_increasing

first = (
    df[df["event"] == "view"]
    .drop_duplicates("visitorid", keep="first")
    .rename(columns={"timestamp": "first_view_timestamp"})
    .drop(columns="event")
    .reset_index(drop=True)
)
first_ts = first.set_index("visitorid")["first_view_timestamp"]


def visitors_after_first_view(event):
    d = df[df["event"] == event]
    return set(d.loc[d["timestamp"] >= d["visitorid"].map(first_ts), "visitorid"])


first["carted_after_first_view"] = first["visitorid"].isin(visitors_after_first_view("addtocart"))
first["purchased_after_first_view"] = first["visitorid"].isin(visitors_after_first_view("transaction"))
first["first_view_date"] = pd.to_datetime(first["first_view_timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
first["first_view_in_first_4weeks"] = first["first_view_timestamp"] < FIRST_4WEEKS_END_MS
first = first[
    ["visitorid", "first_view_timestamp", "first_view_date", "first_view_in_first_4weeks", "itemid", "categoryid",
     "root_categoryid", "category_match_status", "carted_after_first_view", "purchased_after_first_view"]
]

# 검증
assert first["visitorid"].is_unique and len(first) == df.loc[df["event"] == "view", "visitorid"].nunique()
assert first["categoryid"].notna().eq(first["category_match_status"].eq("matched")).all()
assert first["root_categoryid"].notna().eq(first["categoryid"].notna()).all()
assert (first["purchased_after_first_view"] <= first["visitorid"].isin(df.loc[df["event"] == "transaction", "visitorid"])).all()

OUT.parent.mkdir(parents=True, exist_ok=True)
first.to_csv(OUT, index=False)
print("행:", len(first), "| 열:", list(first.columns))
print("카테고리 상태:", first["category_match_status"].value_counts().to_dict())
print("첫 조회 이후 장바구니:", int(first["carted_after_first_view"].sum()), f"({first['carted_after_first_view'].mean() * 100:.2f}%)",
      "| 첫 조회 이후 구매:", int(first["purchased_after_first_view"].sum()), f"({first['purchased_after_first_view'].mean() * 100:.2f}%)")
print("고유 대분류:", first["root_categoryid"].nunique(), "| 고유 카테고리:", first["categoryid"].nunique(), "| 고유 상품:", first["itemid"].nunique())
print("시작 4주 첫 조회:", int(first["first_view_in_first_4weeks"].sum()))
