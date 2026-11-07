"""events_clean.csv에 상품 카테고리를 시점 기준으로 left join 한다.

실행: 프로젝트 루트에서 python analysis/08_join_category.py
입력: data/processed/events_clean.csv (기준 테이블), data/raw/item_category_history.csv
산출: data/processed/events_with_category.csv, outputs/tables/category_join_log.csv

조인 기준 (data/raw의 SOURCE 안내): 같은 itemid의 이력 중 effective_timestamp <= 이벤트 timestamp 인
기록만 사용하고, 그중 가장 늦은 기록을 붙인다. 이벤트 시각 이전 기록이 없으면 카테고리는 결측으로 두고
이후 시점의 카테고리로 채우지 않는다(예측 시점 이후 정보 사용 금지 원칙과 같은 이유).
"""
from pathlib import Path

import pandas as pd

EVENTS = Path("data/processed/events_clean.csv")
HISTORY = Path("data/raw/item_category_history.csv")
OUT = Path("data/processed/events_with_category.csv")
LOG = Path("outputs/tables/category_join_log.csv")

ev = pd.read_csv(EVENTS, dtype={"transactionid": "Int64"})
hist = pd.read_csv(HISTORY).rename(columns={"effective_timestamp": "category_effective_timestamp"})
n_events = len(ev)
assert ev["timestamp"].is_monotonic_increasing  # merge_asof 전제: 기준 키 정렬

# 이력 키 점검: (itemid, 시각)이 유일해야 as-of 결과가 하나로 정해진다.
assert not hist.duplicated(["itemid", "category_effective_timestamp"]).any()
hist = hist.sort_values("category_effective_timestamp", kind="stable")

joined = pd.merge_asof(
    ev,
    hist[["category_effective_timestamp", "itemid", "categoryid", "root_categoryid"]],
    left_on="timestamp",
    right_on="category_effective_timestamp",
    by="itemid",
    direction="backward",
    allow_exact_matches=True,
)

# 매칭 상태: 왜 카테고리가 비었는지 구분해 남긴다.
item_in_history = ev["itemid"].isin(hist["itemid"]).to_numpy()
matched = joined["categoryid"].notna().to_numpy()
status = pd.Series("matched", index=joined.index)
status[~matched & ~item_in_history] = "item_not_in_history"
status[~matched & item_in_history] = "before_first_history"
joined["category_match_status"] = status
for col in ["categoryid", "root_categoryid", "category_effective_timestamp"]:
    joined[col] = joined[col].astype("Int64")

# 검증
assert len(joined) == n_events, "left join 후 행 수가 달라짐"
assert joined["timestamp"].equals(ev["timestamp"]) and joined["visitorid"].equals(ev["visitorid"])
assert (joined["category_effective_timestamp"].dropna() <= joined.loc[joined["category_effective_timestamp"].notna(), "timestamp"]).all()
assert joined["categoryid"].notna().eq(joined["category_match_status"].eq("matched")).all()

OUT.parent.mkdir(parents=True, exist_ok=True)
joined.to_csv(OUT, index=False)

# 기록
by_status = joined["category_match_status"].value_counts()
rows = [
    ("기준 테이블 행 수", EVENTS.name, n_events, ""),
    ("조인 후 행 수", OUT.name, len(joined), "left join 이므로 기준 행 수와 같아야 함"),
    ("매칭(카테고리 있음)", "category_match_status=matched", int(by_status.get("matched", 0)), f"{by_status.get('matched', 0) / n_events * 100:.2f}%"),
    ("이력에 없는 상품", "category_match_status=item_not_in_history", int(by_status.get("item_not_in_history", 0)), f"{by_status.get('item_not_in_history', 0) / n_events * 100:.2f}%"),
    ("상품 최초 이력 이전 이벤트", "category_match_status=before_first_history", int(by_status.get("before_first_history", 0)), f"{by_status.get('before_first_history', 0) / n_events * 100:.2f}%"),
    ("매칭됐지만 root 카테고리 결측", "root_categoryid is NA & matched", int((joined["root_categoryid"].isna() & matched).sum()), "category_tree에서 root를 찾지 못한 경우"),
    ("조인 시 사용한 이력 행", HISTORY.name, len(hist), "완전 중복 0, (itemid, 시각) 중복 0"),
]
log = pd.DataFrame(rows, columns=["item", "criterion", "rows", "note"])
LOG.parent.mkdir(parents=True, exist_ok=True)
log.to_csv(LOG, index=False, encoding="utf-8-sig")
print(log.to_string())

print("\n이벤트별 매칭률(%):")
print((joined.groupby("event")["category_match_status"].apply(lambda s: s.eq("matched").mean() * 100)).round(2).to_string())
joined["_m"] = pd.to_datetime(joined["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m")
print("\n월별 매칭률(%):")
print(joined.groupby("_m")["category_match_status"].apply(lambda s: s.eq("matched").mean() * 100).round(2).to_string())
print("\n구매(transaction) 행 상태:", joined.loc[joined.event == "transaction", "category_match_status"].value_counts().to_dict())
print("고유 카테고리(매칭 행):", joined["categoryid"].nunique(), "| 고유 root:", joined["root_categoryid"].nunique())
print("열:", list(joined.columns.drop("_m")))
