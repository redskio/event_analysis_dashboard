"""CLAUDE.md '이번 분석 기준'에 따라 분석용 이벤트 데이터를 만든다.

실행: 프로젝트 루트에서 python analysis/03_clean_events.py
산출: data/processed/events_clean.csv, outputs/tables/cleaning_log.csv
"""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/events.csv")
CLEAN = Path("data/processed/events_clean.csv")
LOG = Path("outputs/tables/cleaning_log.csv")
CUTOFF = pd.Timestamp("2015-09-15 21:00:00", tz="UTC")
OBSERVATION_WEEKS = 4

df = pd.read_csv(RAW, dtype={"visitorid": "int64", "itemid": "int64", "event": "str", "transactionid": "Int64"})
n_raw = len(df)

# 1) 모든 열이 같은 완전 중복 제거 (첫 행 유지)
dup_mask = df.duplicated(keep="first")
dup_by_event = df.loc[dup_mask, "event"].value_counts()
df = df[~dup_mask]
n_after_dup = len(df)

# 2) 수집량 급감 구간 제외 (기준 시각 미만만 유지)
time_utc = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
keep = time_utc < CUTOFF
cut_by_event = df.loc[~keep, "event"].value_counts()
df, time_utc = df[keep].copy(), time_utc[keep]
n_final = len(df)

# 3) 시작 4주 표시: 제외하지 않고 이전 데이터가 관측되지 않는 구간임을 표시한다.
first_day = time_utc.min().normalize()
window_end = first_day + pd.Timedelta(weeks=OBSERVATION_WEEKS)
df["is_first_4weeks"] = time_utc < window_end

# 4) UTC 기준 날짜 칼럼. timestamp 원값은 그대로 둔다.
WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
df["year"] = time_utc.dt.year
df["month"] = time_utc.dt.month
df["day"] = time_utc.dt.day
df["weekday"] = time_utc.dt.dayofweek.map(dict(enumerate(WEEKDAYS)))
df["day_type"] = time_utc.dt.dayofweek.map(lambda d: "주말" if d >= 5 else "평일")

df = df.sort_values("timestamp", kind="stable")

# 검증
assert not df.duplicated(["timestamp", "visitorid", "event", "itemid", "transactionid"]).any()
assert pd.to_datetime(df["timestamp"].max(), unit="ms", utc=True) < CUTOFF
assert n_raw - dup_mask.sum() - (~keep).sum() == n_final
assert (df["event"].eq("transaction") == df["transactionid"].notna()).all()

CLEAN.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN, index=False)

log = pd.DataFrame(
    [
        {"step": "원본", "criterion": str(RAW), "rows_removed": 0, "rows_after": n_raw, "detail": ""},
        {"step": "완전 중복 제거", "criterion": "모든 열 동일, 첫 행 유지", "rows_removed": int(dup_mask.sum()), "rows_after": n_after_dup, "detail": str(dup_by_event.to_dict())},
        {"step": "기간 제외", "criterion": f"timestamp >= {CUTOFF}", "rows_removed": int((~keep).sum()), "rows_after": n_final, "detail": str(cut_by_event.to_dict())},
    ]
)
LOG.parent.mkdir(parents=True, exist_ok=True)
log.to_csv(LOG, index=False, encoding="utf-8-sig")
print(log.to_string())

print("\n이벤트별 행:", df["event"].value_counts().to_dict())
print("기간(UTC):", time_utc.min(), "~", time_utc.max())
print("시작 4주 구간(UTC):", first_day, "이상", window_end, "미만")
print("시작 4주 행:", int(df["is_first_4weeks"].sum()), f"({df['is_first_4weeks'].mean() * 100:.2f}%)")
print("고유 방문자:", df["visitorid"].nunique(), "| 고유 주문:", df["transactionid"].nunique())
print("첫날/마지막 날 포함 시간(UTC):", time_utc.min().time(), "부터 /", time_utc.max().time(), "까지")
