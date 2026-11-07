"""events_clean.csv를 방문자(visitorid) 1행으로 집계한다.

실행: 프로젝트 루트에서 python analysis/04_visitor_table.py
산출: data/processed/visitor_summary.csv

금액 정보가 데이터에 없으므로 구매 금액 칼럼은 만들지 않는다.
"""
from pathlib import Path

import pandas as pd

CLEAN = Path("data/processed/events_clean.csv")
OUT = Path("data/processed/visitor_summary.csv")
FIRST_4WEEKS_END = pd.Timestamp("2015-05-31", tz="UTC")

df = pd.read_csv(CLEAN, usecols=["timestamp", "visitorid", "event", "transactionid"], dtype={"transactionid": "Int64"})


def to_utc_date(series):
    return pd.to_datetime(series, unit="ms", utc=True).dt.strftime("%Y-%m-%d")


tx = df[df["event"] == "transaction"]
first_ts = df.groupby("visitorid")["timestamp"].min()

visitors = pd.DataFrame(index=first_ts.index)
visitors["purchase_count"] = tx.groupby("visitorid")["transactionid"].nunique()
visitors["first_visit_date"] = to_utc_date(first_ts)
visitors["last_purchase_date"] = to_utc_date(tx.groupby("visitorid")["timestamp"].max())
visitors["view_count"] = df[df["event"] == "view"].groupby("visitorid").size()
visitors["addtocart_count"] = df[df["event"] == "addtocart"].groupby("visitorid").size()
visitors["first_visit_in_first_4weeks"] = pd.to_datetime(first_ts, unit="ms", utc=True) < FIRST_4WEEKS_END

count_cols = ["purchase_count", "view_count", "addtocart_count"]
visitors[count_cols] = visitors[count_cols].fillna(0).astype("int64")

# 검증
assert visitors.index.is_unique
assert len(visitors) == df["visitorid"].nunique()
assert visitors["purchase_count"].sum() == tx["transactionid"].nunique()
assert visitors["view_count"].sum() == (df["event"] == "view").sum()
assert visitors["addtocart_count"].sum() == (df["event"] == "addtocart").sum()
assert (visitors["purchase_count"] > 0).eq(visitors["last_purchase_date"].notna()).all()

OUT.parent.mkdir(parents=True, exist_ok=True)
visitors.reset_index().to_csv(OUT, index=False)

print("행:", len(visitors), "| 열:", ["visitorid"] + list(visitors.columns))
print("결측:", visitors.isna().sum().to_dict())
print("합계:", visitors[count_cols].sum().to_dict())
print("구매자:", int((visitors["purchase_count"] > 0).sum()))
print("시작 4주 내 최초 관측 방문자:", int(visitors["first_visit_in_first_4weeks"].sum()))
print(visitors[count_cols].describe(percentiles=[0.5, 0.99]).round(2))
print(visitors.sort_values("purchase_count", ascending=False).head(3).reset_index().to_string())
