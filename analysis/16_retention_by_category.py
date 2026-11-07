"""대시보드 '리텐션' 탭의 입력인 방문자-날짜 테이블을 만들고, 전체 기간 곡선을 앞 단계 표와 대조한다.

실행: 프로젝트 루트에서 python analysis/16_retention_by_category.py
입력: data/processed/events_with_category.csv (08_join_category.py, events_clean.csv에 시점 기준 카테고리를 붙인 파일)
비교: outputs/tables/retention_nday.csv, outputs/tables/retention_by_behavior.csv (전체 값이 같아야 한다)
산출: data/processed/retention_visitor_days.csv (방문자·활동 UTC 날짜당 1행)

열
- root_categoryid, categoryid       그날 첫 이벤트의 상품 카테고리(그 시각에 적용된 값, 같은 시각이면 파일 순서의 첫 행, 미확인 -1)
- buy                               그날 구매 여부
- buy_root_categoryid, buy_categoryid  그날 첫 구매의 상품 카테고리(구매가 없으면 -1)

대시보드는 이 테이블로 선택 기간에 방문한 방문자의 Day 0(선택 기간의 첫 활동일)을 그때그때 계산한다. 어떤 행도 제외하지 않는다.

지표
- revisit        재방문율:        Day 0 = 첫 이벤트일,  재방문 = 전체 이벤트
- repurchase     재구매율:        Day 0 = 첫 구매일,    재방문 = transaction
- buyer_revisit  구매자의 재방문율: Day 0 = 첫 구매일,    재방문 = 전체 이벤트

기준(grain): day는 정확히 N일째, week는 Day 0이 속한 달력 주(월요일 시작)로부터 N번째 주, month는 달력 월로부터 N번째 달.
Day 0 이전의 활동은 세지 않고, 한 주·달에 여러 번 활동해도 1명으로 센다.
아래 대조는 데이터 전체 기간을 선택했을 때의 값이다.
"""
from pathlib import Path

import pandas as pd

SRC = Path("data/processed/events_with_category.csv")
OUT = Path("data/processed/retention_visitor_days.csv")
OVERALL = Path("outputs/tables/retention_nday.csv")
BY_BEHAVIOR = Path("outputs/tables/retention_by_behavior.csv")
DAY_MS = 86_400_000
UNKNOWN = -1

df = pd.read_csv(
    SRC,
    usecols=["timestamp", "visitorid", "event", "categoryid", "root_categoryid"],
    dtype={"timestamp": "int64", "visitorid": "int64", "event": "str", "categoryid": "Int64", "root_categoryid": "Int64"},
)
assert df["timestamp"].is_monotonic_increasing  # 첫 행 = 가장 이른 이벤트
df["day"] = df["timestamp"] // DAY_MS  # UTC 날짜 번호. 시간대 변환 없음
df["categoryid"] = df["categoryid"].fillna(UNKNOWN).astype("int64")
df["root_categoryid"] = df["root_categoryid"].fillna(UNKNOWN).astype("int64")
buy = df[df["event"] == "transaction"]

anchor_any = df.drop_duplicates("visitorid")[["visitorid", "day", "root_categoryid", "categoryid"]].rename(columns={"day": "day0"})
anchor_buy = buy.drop_duplicates("visitorid")[["visitorid", "day", "root_categoryid", "categoryid"]].rename(columns={"day": "day0"})
active_days = df[["visitorid", "day"]].drop_duplicates()
buy_days = buy[["visitorid", "day"]].drop_duplicates()

METRICS = {"revisit": (anchor_any, active_days), "repurchase": (anchor_buy, buy_days), "buyer_revisit": (anchor_buy, active_days)}


def period_index(days, grain):
    """UTC 날짜 번호를 기준별 기간 번호로 바꾼다. 1970-01-01은 목요일이므로 3을 더하면 월요일에 주가 바뀐다."""
    if grain == "day":
        return days
    if grain == "week":
        return (days + 3) // 7
    dates = pd.to_datetime(days * DAY_MS, unit="ms")
    return dates.dt.year * 12 + dates.dt.month - 1


def build(anchor, activity, grain):
    joined = activity.merge(anchor, on="visitorid")
    joined = joined[joined["day"] >= joined["day0"]].copy()  # Day 0 이전 활동은 세지 않는다
    joined["period"] = period_index(joined["day"], grain)
    joined["n"] = joined["period"] - period_index(joined["day0"], grain)
    joined = joined.drop_duplicates(["visitorid", "period"])
    return joined.groupby(["day0", "root_categoryid", "categoryid", "n"], as_index=False).size().rename(columns={"size": "returned"})


GRAINS = ("day", "week", "month")
result = pd.concat(
    [build(*pair, grain).assign(metric=name, grain=grain) for name, pair in METRICS.items() for grain in GRAINS],
    ignore_index=True,
)
last_day = int(df["day"].max())


def overall_curve(metric):
    r = result[(result["metric"] == metric) & (result["grain"] == "day")]
    size = r[r["n"] == 0].groupby("day0")["returned"].sum()
    returned = r.groupby("n")["returned"].sum()
    eligible = pd.Series({n: int(size[size.index <= last_day - n].sum()) for n in returned.index})
    return eligible, returned


# 검증: 인원과 전체 곡선이 앞 단계 표와 같아야 한다
for name, (anchor, _) in METRICS.items():
    for grain in GRAINS:
        assert result[(result["metric"] == name) & (result["grain"] == grain) & (result["n"] == 0)]["returned"].sum() == len(anchor)
reference = pd.read_csv(OVERALL, encoding="utf-8-sig").set_index("n_day")
behavior = pd.read_csv(BY_BEHAVIOR, encoding="utf-8-sig")
behavior = behavior[behavior["group"] == "구매"].set_index("n_day")
for metric, ref in (("revisit", reference), ("buyer_revisit", behavior)):
    eligible, returned = overall_curve(metric)
    assert (eligible == ref["observable_visitors"].reindex(eligible.index)).all(), metric
    assert (returned == ref["returned_visitors"].reindex(returned.index)).all(), metric

# 방문자-날짜 테이블: df가 시각순이므로 첫 행이 그날의 첫 이벤트다
day_first = df.drop_duplicates(["visitorid", "day"])[["visitorid", "day", "root_categoryid", "categoryid"]]
buy_first = buy.drop_duplicates(["visitorid", "day"])[["visitorid", "day", "root_categoryid", "categoryid"]].rename(
    columns={"root_categoryid": "buy_root_categoryid", "categoryid": "buy_categoryid"}
)
out = day_first.merge(buy_first, on=["visitorid", "day"], how="left", validate="one_to_one")
out["buy"] = out["buy_categoryid"].notna()
out[["buy_root_categoryid", "buy_categoryid"]] = out[["buy_root_categoryid", "buy_categoryid"]].fillna(UNKNOWN).astype("int64")
assert len(out) == len(active_days) and int(out["buy"].sum()) == len(buy_days)
out["date"] = pd.to_datetime(out["day"] * DAY_MS, unit="ms").dt.strftime("%Y-%m-%d")
out = out.sort_values(["visitorid", "day"], kind="stable")[["visitorid", "date", "root_categoryid", "categoryid", "buy", "buy_root_categoryid", "buy_categoryid"]]
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

print("저장:", OUT, "| 행:", len(out))
for name, (anchor, _) in METRICS.items():
    eligible, returned = overall_curve(name)
    unknown = (anchor["categoryid"] == UNKNOWN).mean() * 100
    rates = {f"D{n}": round(returned.get(n, 0) / eligible[n] * 100, 3) for n in (1, 7, 14, 30, 60, 90)}
    print(f"{name}: 대상 {len(anchor):,}명, 카테고리 미확인 {unknown:.1f}%, {rates}, D30 재방문자 {int(returned.get(30, 0))}명")

first_day = buy.merge(anchor_buy[["visitorid", "day0"]], on="visitorid")
first_day = first_day[first_day["day"] == first_day["day0"]]
multi = first_day.groupby("visitorid")["categoryid"].nunique()
print("첫 구매일에 2개 이상 카테고리를 구매한 방문자:", int((multi > 1).sum()), f"({(multi > 1).mean() * 100:.1f}%)")
print("대분류 수:", anchor_any.loc[anchor_any["root_categoryid"] != UNKNOWN, "root_categoryid"].nunique(), "| 카테고리 수:", anchor_any.loc[anchor_any["categoryid"] != UNKNOWN, "categoryid"].nunique())

# 첫 조회 기준(카테고리 탭)과 첫 이벤트 기준(리텐션 탭)의 차이
view_day = df[df["event"] == "view"].drop_duplicates("visitorid").set_index("visitorid")["day"]
compare = anchor_any.set_index("visitorid")["day0"]
print("view 없는 방문자:", int((~compare.index.isin(view_day.index)).sum()), "| 첫 조회일과 첫 이벤트일이 다른 방문자:", int((view_day != compare.reindex(view_day.index)).sum()))
