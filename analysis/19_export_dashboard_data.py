"""대시보드(Streamlit Cloud 배포)용 데이터를 parquet로 내보낸다.

실행: 프로젝트 루트에서 python analysis/19_export_dashboard_data.py
입력: data/processed/ 의 events_clean.csv, first_view_visitors.csv, funnel_visitors.csv, retention_visitor_days.csv
산출: data/processed/dashboard/*.parquet, meta.json (CSV 436MB → parquet 약 33MB, GitHub 100MB 파일 제한 이내)

값은 바꾸지 않고 저장 방식만 줄인다. 대시보드가 읽는 열만 남기고, 자료형을 작게 바꾸며(날짜는 2015-05-03부터의 일수,
카테고리 ID는 int16, 결측은 -1), 대시보드 로더가 CSV 로더와 같은 DataFrame을 다시 만든다.
  - 날짜: DAY0(2015-05-03)부터의 일수(int16). 이벤트 시각의 시·분·초는 대시보드가 쓰지 않으므로 일 단위만 남긴다.
  - 데이터 시작 시각(첫 이벤트의 밀리초)은 meta.json에 따로 저장한다.
  - 소요 시간은 밀리초 정수(int32), 횟수는 int16로 저장하고 전환이 없으면 -1이다.
  - first_view.parquet에는 30일 재방문(ret30)과 관찰 가능 여부(obs30)를 미리 계산해 넣는다. 대시보드가 실행할 때
    이벤트 전체와 (방문자, 날짜) 쌍을 맞추면 메모리를 많이 쓰기 때문이다.
      obs30: 첫 조회일 + 30일이 마지막 관측일(2015-09-15) 이내
      ret30: 첫 조회일로부터 정확히 30일째 날짜에 이벤트(종류·상품 무관)가 있는 방문자
행 순서는 원본 CSV와 같게 유지한다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "dashboard"
DAY0 = pd.Timestamp("2015-05-03")
DAY_MS = 86_400_000
DAY0_MS = DAY0.value // 1_000_000
LAST_DAY = (pd.Timestamp("2015-09-15") - DAY0).days  # 마지막 관측일(UTC)의 DAY0 기준 일수


def to_days_from_date_string(s):
    return (pd.to_datetime(s) - DAY0).dt.days.astype("int16")


def to_int16(s, label):
    assert s.min() >= -1 and s.max() < 2**15, f"{label}: int16 범위를 벗어남"
    return s.astype("int16")


def write(df, name):
    path = OUT / f"{name}.parquet"
    df.to_parquet(path, compression="zstd", index=False)
    print(f"{name:14s} {len(df):>9,}행 {df.shape[1]:>2}열 | {path.stat().st_size / 1048576:5.1f}MB | 메모리 {df.memory_usage(deep=True).sum() / 1048576:6.1f}MB")
    return path.stat().st_size


OUT.mkdir(parents=True, exist_ok=True)
total = 0

# 1) 이벤트 (탭 1, 첫 조회 재방문 계산)
e = pd.read_csv(PROCESSED / "events_clean.csv", usecols=["timestamp", "visitorid", "event", "transactionid"],
                dtype={"visitorid": "int32", "event": "str", "transactionid": "float64"})
assert e["timestamp"].is_monotonic_increasing
data_start_ms = int(e["timestamp"].min())
day = ((e["timestamp"] - DAY0_MS) // DAY_MS).astype("int64")
assert day.min() >= 0 and day.max() < 2**15
events = pd.DataFrame(
    {
        "day": day.astype("int16"),
        "visitorid": e["visitorid"].astype("int32"),
        "event": pd.Categorical(e["event"], categories=sorted(e["event"].unique())),
        "transactionid": e["transactionid"].fillna(-1).astype("int32"),
    }
)
total += write(events, "events")
(OUT / "meta.json").write_text(json.dumps({"data_start_ms": data_start_ms, "day0": "2015-05-03"}, indent=2), encoding="utf-8")

# 2) 첫 조회 (탭 2, 산포도)
f = pd.read_csv(PROCESSED / "first_view_visitors.csv",
                usecols=["visitorid", "first_view_date", "first_view_in_first_4weeks", "itemid", "categoryid", "root_categoryid",
                         "carted_after_first_view", "purchased_after_first_view"],
                dtype={"visitorid": "int32", "itemid": "int32", "categoryid": "Int64", "root_categoryid": "Int64"})
fv_day = to_days_from_date_string(f["first_view_date"])
# (방문자, 날짜) 쌍을 정수 하나로 합쳐 비교한다. day는 0~135이므로 65536을 곱해도 겹치지 않는다.
active_keys = np.unique(events["visitorid"].to_numpy(dtype="int64") * 65536 + events["day"].to_numpy(dtype="int64"))
ret_keys = f["visitorid"].to_numpy(dtype="int64") * 65536 + (fv_day.to_numpy(dtype="int64") + 30)
first_view = pd.DataFrame(
    {
        "visitorid": f["visitorid"],
        "day": fv_day,
        "early": f["first_view_in_first_4weeks"].astype(bool),
        "itemid": f["itemid"],
        "categoryid": to_int16(f["categoryid"].fillna(-1).astype("int64"), "categoryid"),
        "root_categoryid": to_int16(f["root_categoryid"].fillna(-1).astype("int64"), "root_categoryid"),
        "carted": f["carted_after_first_view"].astype(bool),
        "purchased": f["purchased_after_first_view"].astype(bool),
        "obs30": (fv_day.to_numpy(dtype="int64") + 30) <= LAST_DAY,
        "ret30": np.isin(ret_keys, active_keys),
    }
)
assert not (first_view["ret30"] & ~first_view["obs30"]).any()  # 관찰할 수 없는 날짜에는 재방문이 있을 수 없다
total += write(first_view, "first_view")

# 3) 퍼널 (탭 3)
g = pd.read_csv(PROCESSED / "funnel_visitors.csv",
                usecols=["first_view_date", "first_view_in_first_4weeks", "root_categoryid", "categoryid", "view_to_cart_ms", "cart_to_buy_ms",
                         "views_before_cart", "carts_before_buy", "converted_cart_24h", "converted_buy_24h", "observable_24h"],
                dtype={"root_categoryid": "Int64", "categoryid": "Int64", "view_to_cart_ms": "float64", "cart_to_buy_ms": "float64",
                       "views_before_cart": "float64", "carts_before_buy": "float64"})
c24, b24 = g["converted_cart_24h"].astype(bool), g["converted_buy_24h"].astype(bool)
for col in ["view_to_cart_ms", "cart_to_buy_ms", "views_before_cart", "carts_before_buy"]:  # 정수값인지 확인(소수가 있으면 잘리므로)
    v = g[col].dropna()
    assert (v == v.round()).all(), col
funnel = pd.DataFrame(
    {
        "day": to_days_from_date_string(g["first_view_date"]),
        "early": g["first_view_in_first_4weeks"].astype(bool),
        "obs": g["observable_24h"].astype(bool),
        "root_categoryid": to_int16(g["root_categoryid"].fillna(-1).astype("int64"), "root_categoryid"),
        "categoryid": to_int16(g["categoryid"].fillna(-1).astype("int64"), "categoryid"),
        "c24": c24,
        "b24": b24,
        # 대시보드는 24시간 이내 전환한 방문자의 값만 쓰므로 그 값만 남기고 나머지는 -1(결측)
        "v2c_ms": g["view_to_cart_ms"].where(c24).fillna(-1).astype("int32"),
        "c2b_ms": g["cart_to_buy_ms"].where(b24).fillna(-1).astype("int32"),
        "views_n": g["views_before_cart"].where(c24).fillna(-1).astype("int16"),
        "carts_n": g["carts_before_buy"].where(b24).fillna(-1).astype("int16"),
    }
)
total += write(funnel, "funnel")

# 4) 리텐션 (탭 4)
r = pd.read_csv(PROCESSED / "retention_visitor_days.csv",
                dtype={"visitorid": "int32", "root_categoryid": "int32", "categoryid": "int32", "buy": "bool",
                       "buy_root_categoryid": "int32", "buy_categoryid": "int32"})
r = r.assign(date=pd.to_datetime(r["date"].astype("str"))).sort_values(["visitorid", "date"], kind="stable").reset_index(drop=True)
retention = pd.DataFrame(
    {
        "visitorid": r["visitorid"],
        "day": (r["date"] - DAY0).dt.days.astype("int16"),
        "root_categoryid": to_int16(r["root_categoryid"].astype("int64"), "root_categoryid"),
        "categoryid": to_int16(r["categoryid"].astype("int64"), "categoryid"),
        "buy": r["buy"].astype(bool),
        "buy_root_categoryid": to_int16(r["buy_root_categoryid"].astype("int64"), "buy_root_categoryid"),
        "buy_categoryid": to_int16(r["buy_categoryid"].astype("int64"), "buy_categoryid"),
    }
)
total += write(retention, "retention")

print(f"\n합계 {total / 1048576:.1f}MB | 최대 파일 {max(p.stat().st_size for p in OUT.glob('*.parquet')) / 1048576:.1f}MB")

# 검증: 다시 읽어 행 수와 집계가 원본 CSV와 같은지 확인한다.
back = {n: pd.read_parquet(OUT / f"{n}.parquet") for n in ["events", "first_view", "funnel", "retention"]}
assert len(back["events"]) == len(e) and back["events"]["day"].between(0, 135).all()
assert (back["events"]["transactionid"] >= 0).sum() == e["transactionid"].notna().sum()
assert back["first_view"]["carted"].sum() == f["carted_after_first_view"].sum()
assert back["funnel"]["c24"].sum() == c24.sum() and (back["funnel"]["v2c_ms"] >= 0).sum() == c24.sum()
assert back["retention"]["buy"].sum() == r["buy"].sum() and len(back["retention"]) == len(r)
print("검증 통과: 행 수·플래그·전환 수가 원본 CSV와 같음")
