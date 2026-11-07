"""events.csv 데이터 품질 점검 (원본은 읽기만 하고 수정하지 않는다).

실행: 프로젝트 루트에서 python analysis/01_data_quality.py
산출: outputs/tables/dq_*.csv
"""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/events.csv")
OUT = Path("outputs/tables")
ID_COLS = ["visitorid", "itemid", "transactionid"]
KEY_COLS = ["timestamp", "visitorid", "event", "itemid", "transactionid"]


def section(title):
    print(f"\n=== {title} ===")


# 1) 형식 오류까지 잡기 위해 전 열을 문자열로 읽는다.
raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
n = len(raw)
section("기본")
print("행, 열:", raw.shape, "| 열:", list(raw.columns))

# 2) 이벤트 종류별 건수 (표기 변형 확인을 위해 원문 그대로 집계)
section("이벤트별 건수")
event_counts = raw["event"].value_counts().rename("rows").to_frame()
event_counts["share_pct"] = (event_counts["rows"] / n * 100).round(3)
print(event_counts)
event_counts.to_csv(OUT / "dq_event_counts.csv")

# 3) 열별 결측: 빈 문자열, 공백뿐인 값, 결측 대체 표기
section("열별 결측")
NA_TOKENS = {"na", "n/a", "nan", "null", "none", "-", "?", "undefined"}
missing = pd.DataFrame(
    {
        "empty": (raw == "").sum(),
        "whitespace_only": raw.apply(lambda s: (s.str.strip().eq("") & s.ne("")).sum()),
        "na_token": raw.apply(lambda s: s.str.strip().str.lower().isin(NA_TOKENS).sum()),
        "padded": raw.apply(lambda s: (s != s.str.strip()).sum()),
    }
)
missing["missing_pct"] = (missing["empty"] / n * 100).round(3)
print(missing)
missing.to_csv(OUT / "dq_missing_by_column.csv", index_label="column")

# 4) 형식 검증: 모든 ID, timestamp는 숫자만이어야 한다.
section("형식 검증 (비어 있지 않은 값 중 숫자 형식이 아닌 건수)")
for col in ["timestamp"] + ID_COLS:
    filled = raw.loc[raw[col] != "", col]
    print(f"{col}: {(~filled.str.fullmatch(r'[0-9]+')).sum()} | 앞자리 0: {filled.str.fullmatch(r'0[0-9]+').sum()}")
print("timestamp 자릿수:", raw["timestamp"].str.len().value_counts().to_dict())

# 5) 이벤트 × transactionid 유무
section("event × transactionid 유무")
raw["has_txn"] = raw["transactionid"] != ""
crosstab = pd.crosstab(raw["event"], raw["has_txn"]).rename(columns={False: "txn_missing", True: "txn_present"})
print(crosstab)
crosstab.to_csv(OUT / "dq_event_by_transactionid.csv")

# 이후 점검은 숫자형으로 변환해 수행한다.
df = raw.drop(columns="has_txn").copy()
df["timestamp"] = df["timestamp"].astype("int64")
df["visitorid"] = df["visitorid"].astype("int64")
df["itemid"] = df["itemid"].astype("int64")
df["transactionid"] = pd.to_numeric(df["transactionid"]).astype("Int64")
df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
tx = df[df["event"] == "transaction"]

# 6) 중복
section("중복")
dup_mask = df.duplicated(KEY_COLS, keep="first")
print("완전 중복(첫 행 제외):", int(dup_mask.sum()), f"({dup_mask.mean() * 100:.3f}%)")
print("이벤트별 중복:", df.loc[dup_mask, "event"].value_counts().to_dict())
print("완전 중복이 속한 전체 행:", int(df.duplicated(KEY_COLS, keep=False).sum()))
print("중복 그룹 크기 상위:", df[df.duplicated(KEY_COLS, keep=False)].groupby(KEY_COLS, dropna=False).size().value_counts().head(5).to_dict())
near = df[~dup_mask].duplicated(["visitorid", "event", "itemid"], keep=False)
print("(visitor,event,item) 반복 행 — 다른 시각 포함:", int(near.sum()))
srt = df[~dup_mask].sort_values(["visitorid", "itemid", "event", "timestamp"])
gap = srt.groupby(["visitorid", "itemid", "event"])["timestamp"].diff()
for sec in [1, 5, 60]:
    m = gap.notna() & (gap < sec * 1000)
    print(f"같은 visitor·item·event가 {sec}초 미만 간격으로 재발생:", srt.loc[m, "event"].value_counts().to_dict())

# 7) 시간 범위와 일별 추이
section("시간")
print("최소/최대 (UTC):", df["ts"].min(), df["ts"].max())
print("기간(일):", (df["ts"].max() - df["ts"].min()).days)
print("파일이 시간순 정렬?:", df["timestamp"].is_monotonic_increasing)
daily = df.groupby([df["ts"].dt.date.rename("date"), "event"]).size().unstack(fill_value=0)
daily["total"] = daily.sum(axis=1)
daily["visitors"] = df.groupby(df["ts"].dt.date)["visitorid"].nunique()
daily.to_csv(OUT / "dq_daily_events.csv")
print("일수:", len(daily), "| 첫 3일\n", daily.head(3), "\n마지막 3일\n", daily.tail(3))
print("일별 total 요약:\n", daily["total"].describe().round(0))
all_days = pd.date_range(daily.index.min(), daily.index.max()).date
print("이벤트가 없는 날:", sorted(set(all_days) - set(daily.index)))
median = daily["total"].median()
print("중앙값 50% 미만/150% 초과인 날:\n", daily[(daily["total"] < median * 0.5) | (daily["total"] > median * 1.5)])
rolling = daily["transaction"] / daily["view"] * 100
print("일별 transaction/view(%):\n", rolling.describe().round(3))
hourly = df.groupby(df["ts"].dt.floor("h")).size()
hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h"), fill_value=0)
print("이벤트 0건 시간대 수:", int((hourly == 0).sum()), "| 예:", [str(x) for x in hourly[hourly == 0].index[:10]])
print("UTC 시간대별 비중(%):", (df["ts"].dt.hour.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())

# 8) 식별자 범위, 카디널리티
section("식별자")
print(df[["visitorid", "itemid", "transactionid"]].agg(["nunique", "min", "max"]))
print("이벤트별 고유 방문자:", df.groupby("event")["visitorid"].nunique().to_dict())
print("이벤트별 고유 아이템:", df.groupby("event")["itemid"].nunique().to_dict())

# 9) transaction 무결성
section("transactionid 무결성")
per_txn = tx.groupby("transactionid").agg(
    rows=("itemid", "size"),
    items=("itemid", "nunique"),
    visitors=("visitorid", "nunique"),
    timestamps=("timestamp", "nunique"),
    span_ms=("timestamp", lambda s: s.max() - s.min()),
)
print("고유 transactionid:", len(per_txn))
print("주문당 행 수 분포:\n", per_txn["rows"].describe().round(2))
print("여러 방문자에 걸친 ID:", int((per_txn["visitors"] > 1).sum()))
print("여러 시각에 걸친 ID:", int((per_txn["timestamps"] > 1).sum()))
print("여러 시각 ID의 시간차(초):\n", (per_txn.loc[per_txn["timestamps"] > 1, "span_ms"] / 1000).describe().round(1))
print("같은 주문에 같은 아이템 반복된 ID:", int((per_txn["rows"] > per_txn["items"]).sum()))
print("행 수 상위 5:\n", per_txn.sort_values("rows", ascending=False).head(5))

# 10) 퍼널 선후관계: 같은 방문자·아이템 기준, 동일 시각은 '이전'으로 인정
section("퍼널 선후관계 (visitor·item 단위)")
first = df.groupby(["visitorid", "itemid", "event"])["timestamp"].min().unstack()
for col in ["view", "addtocart", "transaction"]:
    if col not in first:
        first[col] = pd.NA
tx_pairs = first[first["transaction"].notna()]
atc_pairs = first[first["addtocart"].notna()]
rows = {
    "구매 쌍(visitor-item)": len(tx_pairs),
    "구매 쌍 중 view 기록 없음": int(tx_pairs["view"].isna().sum()),
    "구매 쌍 중 addtocart 기록 없음": int(tx_pairs["addtocart"].isna().sum()),
    "구매 쌍 중 view·addtocart 모두 없음": int((tx_pairs["view"].isna() & tx_pairs["addtocart"].isna()).sum()),
    "구매 쌍 중 최초 view가 최초 구매보다 뒤": int((tx_pairs["view"] > tx_pairs["transaction"]).sum()),
    "구매 쌍 중 최초 addtocart가 최초 구매보다 뒤": int((tx_pairs["addtocart"] > tx_pairs["transaction"]).sum()),
    "장바구니 쌍": len(atc_pairs),
    "장바구니 쌍 중 view 기록 없음": int(atc_pairs["view"].isna().sum()),
    "장바구니 쌍 중 최초 view가 최초 장바구니보다 뒤": int((atc_pairs["view"] > atc_pairs["addtocart"]).sum()),
}
funnel_order = pd.Series(rows, name="count")
print(funnel_order)
funnel_order.to_csv(OUT / "dq_funnel_order.csv", index_label="check")

# 방문자 단위로 봐도 이전 행동이 없는 구매자
vis_first = df.groupby(["visitorid", "event"])["timestamp"].min().unstack()
buyers = vis_first[vis_first["transaction"].notna()]
print("구매 방문자:", len(buyers), "| 그중 view 전혀 없음:", int(buyers["view"].isna().sum()), "| 구매 시점 이전 이벤트 없음:", int((buyers[["view", "addtocart"]].min(axis=1).isna() | (buyers[["view", "addtocart"]].min(axis=1) > buyers["transaction"])).sum()))

# 첫날 이전 행동 절단 영향: 데이터 시작 7일 내 구매에서 view 누락 비중 비교
start = df["timestamp"].min()
early = tx_pairs["transaction"] < start + 7 * 86_400_000
print("view 없는 구매 쌍 비중 — 시작 7일 내:", round(tx_pairs.loc[early, "view"].isna().mean() * 100, 1), "% | 이후:", round(tx_pairs.loc[~early, "view"].isna().mean() * 100, 1), "%")

# 11) 방문자 활동 분포, 봇 의심
section("방문자 활동 집중도")
per_vis = df.groupby("visitorid").agg(
    rows=("event", "size"),
    days=("ts", lambda s: s.dt.date.nunique()),
)
per_vis = per_vis.join(df.pivot_table(index="visitorid", columns="event", values="itemid", aggfunc="size", fill_value=0))
print("방문자당 이벤트:\n", per_vis["rows"].describe(percentiles=[0.5, 0.9, 0.99, 0.999]).round(1))
print("이벤트 1건 방문자 비중(%):", round((per_vis["rows"] == 1).mean() * 100, 2))
print("활동일 1일 방문자 비중(%):", round((per_vis["days"] == 1).mean() * 100, 2))
ranked = per_vis["rows"].sort_values(ascending=False)
for k in [10, 100, 1000]:
    print(f"상위 {k}명 이벤트 비중(%):", round(ranked.head(k).sum() / n * 100, 2))
top_buyers = per_vis["transaction"].sort_values(ascending=False)
for k in [10, 100]:
    print(f"구매 상위 {k}명의 transaction 비중(%):", round(top_buyers.head(k).sum() / len(tx) * 100, 2))
print("방문자당 구매 행 수 상위 10:", top_buyers.head(10).to_dict())
top = per_vis.sort_values("rows", ascending=False).head(20)
print("이벤트 상위 20명:\n", top)
top.to_csv(OUT / "dq_top_visitors.csv")

# 초고빈도 방문자(분당 이벤트, 이벤트 간 간격)
bursty = df.sort_values(["visitorid", "timestamp"])
gap_v = bursty.groupby("visitorid")["timestamp"].diff()
stats = pd.DataFrame({"visitorid": bursty["visitorid"], "gap": gap_v}).dropna().groupby("visitorid")["gap"].median()
top_ids = ranked.head(1000).index
print("상위 1000명의 이벤트 간격 중앙값(초) 요약:\n", (stats.reindex(top_ids) / 1000).describe().round(1))

per_vis["view_only_heavy"] = (per_vis["rows"] >= 500) & (per_vis["transaction"] == 0)
print("이벤트 500건 이상 방문자:", int((per_vis["rows"] >= 500).sum()), "| 그중 구매 0건:", int(per_vis["view_only_heavy"].sum()))

# 코호트 관련: 방문자당 활동 기간
span = df.groupby("visitorid")["ts"].agg(["min", "max"])
span["days"] = (span["max"] - span["min"]).dt.days
print("재방문(최초~최종 1일 이상) 방문자 비중(%):", round((span["days"] >= 1).mean() * 100, 2))

# 아이템 집중도
section("아이템 집중도")
per_item = df.groupby("itemid").size().sort_values(ascending=False)
print("상위 10 아이템 이벤트 비중(%):", round(per_item.head(10).sum() / n * 100, 2), "| 상위 1건:", per_item.index[0], int(per_item.iloc[0]))
print("이벤트 1건 아이템 비중(%):", round((per_item == 1).mean() * 100, 2))

# 12) 경계 시각: 시작·종료가 03:00 UTC에 맞물리므로 UTC-3 하루 기준과 비교
section("하루 경계 비교 (UTC vs UTC-3)")
local = df["ts"] - pd.Timedelta(hours=3)
local_daily = df.groupby(local.dt.date).size()
print("UTC-3 기준 일수:", len(local_daily), "| 첫날/마지막날:", int(local_daily.iloc[0]), int(local_daily.iloc[-1]))
print("UTC-3 기준 마지막 7일:", local_daily.tail(7).to_dict())
print("각 경계 일의 시간 범위:", local.min(), local.max())

# 13) 이벤트 행 단위 선후관계: 각 행 시점 이전(동일 시각 포함) 선행 행동 존재 여부
section("행 단위 선행 행동 (중복 제거 후)")
dedup = df[~dup_mask].sort_values("timestamp")


def prior_share(target, prior):
    left = dedup[dedup["event"] == target][["timestamp", "visitorid", "itemid", "ts"]]
    right = dedup[dedup["event"] == prior][["timestamp", "visitorid", "itemid"]].rename(columns={"timestamp": "prior_ts"})
    merged = pd.merge_asof(left, right, left_on="timestamp", right_on="prior_ts", by=["visitorid", "itemid"])
    merged["no_prior"] = merged["prior_ts"].isna()
    weekly = merged.groupby(merged["ts"].dt.to_period("M").astype(str))["no_prior"].mean().mul(100).round(1)
    print(f"{target} 중 같은 visitor·item의 선행 {prior} 없음: {int(merged['no_prior'].sum())} / {len(merged)} ({merged['no_prior'].mean() * 100:.1f}%)")
    print("  월별(%):", weekly.to_dict())
    return merged


prior_share("addtocart", "view")
tx_no_atc = prior_share("transaction", "addtocart")
prior_share("transaction", "view")
same_ms = dedup.duplicated(["visitorid", "timestamp"], keep=False)
print("같은 방문자·같은 ms에 복수 이벤트:", int(same_ms.sum()), "행 | 이벤트 조합 상위:",
      dedup[same_ms].groupby(["visitorid", "timestamp"])["event"].agg(lambda s: "+".join(sorted(s))).value_counts().head(5).to_dict())

# 14) 재발생 비중과 transaction 세부
section("이중 기록 의심")
ev_total = df[~dup_mask]["event"].value_counts()
for sec in [1, 5]:
    m = gap.notna() & (gap < sec * 1000)
    share = (srt.loc[m, "event"].value_counts() / ev_total * 100).round(2)
    print(f"{sec}초 미만 재발생 비중(%):", share.to_dict())
print("주문 내 시간차 60초 이상 ID:", int((per_txn["span_ms"] >= 60_000).sum()), "| 1초 이상:", int((per_txn["span_ms"] >= 1000).sum()))
print("주문 내 같은 아이템 반복 초과 행:", int((per_txn["rows"] - per_txn["items"]).sum()))
pair_txn = tx.groupby(["visitorid", "itemid"])["transactionid"].nunique()
print("같은 visitor·item이 2개 이상 주문:", int((pair_txn > 1).sum()), "/", len(pair_txn))
tx_sorted = tx.sort_values(["visitorid", "itemid", "timestamp"])
tx_gap = tx_sorted.groupby(["visitorid", "itemid"])["timestamp"].diff()
same_id = tx_sorted.groupby(["visitorid", "itemid"])["transactionid"].shift() == tx_sorted["transactionid"]
near = tx_gap.notna() & (tx_gap < 60_000)
print("60초 내 재구매 행 중 같은 transactionid:", int((near & same_id).sum()), "| 다른 transactionid:", int((near & ~same_id).sum()))
print("주문 8351 예시:\n", tx[tx["transactionid"] == 8351].sort_values("timestamp")[["ts", "itemid"]].head(8))

# 15) 7/26 급증 원인, 월별 구조 변화
section("7/26 급증과 기간별 추이")
day = df[df["ts"].dt.date == pd.Timestamp("2015-07-26").date()]
print("시간대별:", day.groupby(day["ts"].dt.hour).size().to_dict())
print("상위 방문자 집중:", day["visitorid"].value_counts().head(5).to_dict())
print("상위 아이템 집중:", day["itemid"].value_counts().head(5).to_dict())
monthly = df.groupby(df["ts"].dt.to_period("M"))
month_tbl = monthly["event"].value_counts().unstack()
month_tbl["visitors"] = monthly["visitorid"].nunique()
month_tbl["atc_per_view_pct"] = (month_tbl["addtocart"] / month_tbl["view"] * 100).round(2)
month_tbl["tx_per_view_pct"] = (month_tbl["transaction"] / month_tbl["view"] * 100).round(2)
print(month_tbl)
week = df.groupby(df["ts"].dt.to_period("W"))
print("주별 view:", {str(k): int(v) for k, v in (df[df["event"] == "view"].groupby(df["ts"].dt.to_period("W")).size()).items()})

# 16) 코호트 관점: 첫 관측일과 관측 가능 기간
section("코호트 관점")
first_seen = df.groupby("visitorid")["ts"].min().dt.to_period("W")
new_by_week = first_seen.value_counts().sort_index()
active_by_week = week["visitorid"].nunique()
share_new = (new_by_week / active_by_week * 100).round(1)
print("주별 신규 비중(%), 앞 4주:", {str(k): v for k, v in share_new.head(4).items()}, "| 뒤 4주:", {str(k): v for k, v in share_new.tail(4).items()})
print("주별 신규/활성 비중(%) 표:\n", pd.DataFrame({"new": new_by_week, "active": active_by_week, "new_pct": share_new}).head(5))

# 17) ID 부여 방식
section("ID 부여 방식")
print("visitorid 연속(0~max):", df["visitorid"].nunique() == df["visitorid"].max() + 1, "| transactionid 연속:", tx["transactionid"].nunique() == tx["transactionid"].max() + 1)
print("itemid 연속:", df["itemid"].nunique() == df["itemid"].max() + 1)
print("visitorid-최초관측 상관(스피어만):", round(df.groupby("visitorid")["timestamp"].min().reset_index().corr(method="spearman").iloc[0, 1], 4))
print("transactionid-시각 상관(스피어만):", round(tx.groupby("transactionid")["timestamp"].min().reset_index().astype("int64").corr(method="spearman").iloc[0, 1], 4))

# 18) 봇·비정상 방문자 후보 규모
section("비정상 방문자 후보")
per_vis["distinct_items"] = df.groupby("visitorid")["itemid"].nunique()
per_vis["median_gap_s"] = stats / 1000
per_min = df.groupby(["visitorid", df["ts"].dt.floor("min")]).size().groupby("visitorid").max()
per_vis["peak_per_min"] = per_min
per_vis["hours_active"] = df.groupby("visitorid")["ts"].apply(lambda s: s.dt.hour.nunique())
for label, m in {
    "이벤트 500건+": per_vis["rows"] >= 500,
    "분당 최대 30건+": per_vis["peak_per_min"] >= 30,
    "간격 중앙값 3초 미만 & 100건+": (per_vis["median_gap_s"] < 3) & (per_vis["rows"] >= 100),
    "24시간 전 시간대 활동 & 구매 0": (per_vis["hours_active"] == 24) & (per_vis["transaction"] == 0),
    "구매 행 50건+": per_vis["transaction"] >= 50,
}.items():
    print(f"{label}: 방문자 {int(m.sum())}, 이벤트 {int(per_vis.loc[m, 'rows'].sum())} ({per_vis.loc[m, 'rows'].sum() / n * 100:.2f}%), transaction {int(per_vis.loc[m, 'transaction'].sum())} ({per_vis.loc[m, 'transaction'].sum() / len(tx) * 100:.2f}%)")
print("분당 이벤트 상위 10:\n", per_vis.sort_values("peak_per_min", ascending=False).head(10)[["rows", "view", "addtocart", "transaction", "distinct_items", "median_gap_s", "peak_per_min", "days"]])
print("view/item 비율 낮은 100+ 구매 방문자 예:\n", per_vis[per_vis["transaction"] >= 100][["rows", "transaction", "distinct_items", "days"]].sort_values("transaction", ascending=False).head(5))

# 19) 종료 직전 2일(UTC-3 9/16~9/17) 감소 여부: 직전 4주 같은 요일과 비교
section("종료 직전 2일 감소")
local_ts = df["ts"] - pd.Timedelta(hours=3)
local_day = local_ts.dt.normalize()
day_event = df.groupby([local_day.rename("day"), "event"]).size().unstack(fill_value=0)
day_event["total"] = day_event.sum(axis=1)
for target in ["2015-09-14", "2015-09-15", "2015-09-16", "2015-09-17"]:
    t = pd.Timestamp(target, tz="UTC")
    base = day_event.loc[[t - pd.Timedelta(weeks=w) for w in range(1, 5)]].mean().round(0)
    print(target, t.day_name(), "실제:", day_event.loc[t].to_dict(), "| 직전 4주 같은 요일 평균:", base.to_dict())
last = local_ts[local_day >= pd.Timestamp("2015-09-14", tz="UTC")]
print(last.groupby([last.dt.date, last.dt.hour]).size().unstack(0).fillna(0).astype(int))
