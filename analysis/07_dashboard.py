"""퍼널 주요 지표·추이, 카테고리별 첫 조회 성과, 퍼널, 리텐션 Streamlit 대시보드.

실행: 프로젝트 루트에서 streamlit run analysis/07_dashboard.py
선택한 탭만 계산한다(st.tabs on_change="rerun"). 닫힌 탭의 위젯 값은 session_state에 다시 써서 유지하므로 탭 안의 모든 위젯에는 key가 있어야 한다.
입력: data/processed/dashboard/*.parquet (analysis/19_export_dashboard_data.py가 data/processed의 CSV에서 만든 배포용 파일, 약 33MB)
  events.parquet     탭 1: 주요 지표·추이 (UTC 기준, 완전 중복 제거, 2015-09-15 21:00 UTC 미만)
  first_view.parquet 탭 2: 카테고리별 첫 조회 성과
  funnel.parquet     탭 3: 퍼널(24시간 전환 기준)
  retention.parquet  탭 4: 리텐션
  meta.json          데이터 시작 시각
지표 정의는 05_key_metrics.py, 06_monthly_metrics.py, 09_first_view_category.py와 같다.
N-day 재방문은 14_retention_curve.py, 15_retention_by_behavior.py와 같다(정확히 N일째 날짜의 활동, 분모는 N일째를 관찰할 수 있는 방문자).
코호트 히트맵의 일별 값은 17_retention_cohort_heatmap.py와 같다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DASH_DATA = ROOT / "data" / "processed" / "dashboard"
EVENTS = DASH_DATA / "events.parquet"
FIRST_VIEW = DASH_DATA / "first_view.parquet"
FUNNEL = DASH_DATA / "funnel.parquet"
RETENTION = DASH_DATA / "retention.parquet"
META = DASH_DATA / "meta.json"
DAY0_MS = pd.Timestamp("2015-05-03").value // 1_000_000  # parquet의 day 열은 2015-05-03 00:00 UTC부터의 일수
# CLAUDE.md의 분석 기준: 이 시각 미만만 사용. 시작은 데이터의 첫 이벤트 시각.
DATA_END = pd.Timestamp("2015-09-15 21:00:00")
LAST_DATE = (DATA_END - pd.Timedelta(seconds=1)).normalize()  # 마지막 관측일(UTC)
FIRST_4WEEKS_END = pd.Timestamp("2015-05-31")  # 시작 4주(05-03~05-30)의 끝 경계(미만)
COLOR, COLOR_LIGHT = "#2a6fbb", "#b9cfe9"
UNKNOWN = -1  # 카테고리를 알 수 없는 첫 조회
TABLE_ROWS = 200
METRICS = [
    ("active_users", "활성 사용자", "명"),
    ("buyers", "구매자", "명"),
    ("purchase_count", "구매 횟수", "건"),
    ("addtocart_count", "장바구니에 담은 횟수", "회"),
    ("event_count", "이벤트 횟수", "건"),
]

st.set_page_config(page_title="퍼널 주요 지표 대시보드", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: clamp(1.3rem, 2.4vw, 2.3rem); font-weight: 700; overflow: visible; }
    [data-testid="stMetricValue"] > div { overflow: visible; text-overflow: clip; white-space: nowrap; }
    [data-testid="stMetricLabel"] p { font-size: 1.02rem; white-space: normal; overflow: visible; text-overflow: clip; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── 데이터 ─────────────────────────────────────────────────
def day_to_datetime(days):
    """parquet의 day(2015-05-03부터의 일수, UTC) → 그날 00:00 UTC의 datetime."""
    return pd.to_datetime(DAY0_MS + days.astype("int64") * 86_400_000, unit="ms")


@st.cache_data(show_spinner="이벤트 데이터를 불러오는 중입니다...")
def load_events():
    df = pd.read_parquet(EVENTS)
    df["transactionid"] = df["transactionid"].where(df["transactionid"] >= 0).astype("float64")  # -1은 결측(구매가 아닌 행)
    df["date"] = day_to_datetime(df["day"])  # UTC 기준, 시간대 변환 없음
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.drop(columns="day"), load_data_start()


@st.cache_data(show_spinner=False)
def load_data_start():
    """데이터의 첫 이벤트 시각(UTC). 이벤트 전체를 읽지 않고 meta.json만 읽는다."""
    return pd.to_datetime(json.loads(META.read_text(encoding="utf-8"))["data_start_ms"], unit="ms")


@st.cache_data(show_spinner="첫 조회 데이터를 불러오는 중입니다...")
def load_first_view():
    if not FIRST_VIEW.exists():
        return None
    fv = pd.read_parquet(FIRST_VIEW)  # categoryid·root_categoryid의 -1(=UNKNOWN)은 카테고리를 알 수 없는 첫 조회
    fv["categoryid"] = fv["categoryid"].astype("int32")
    fv["root_categoryid"] = fv["root_categoryid"].astype("int32")
    fv["date"] = day_to_datetime(fv["day"]).astype("datetime64[ns]")
    # 30일 재방문(ret30)과 30일째가 관측 기간 안인지(obs30)는 analysis/19_export_dashboard_data.py가 미리 계산해 두었다.
    return fv.drop(columns="day")


# ── 탭 1: 주요 지표와 추이 ──────────────────────────────────
def metric_table(d, key):
    """key(date 또는 month)별 주요 지표 5개."""
    tx = d[d["event"] == "transaction"]
    table = pd.DataFrame(
        {
            "active_users": d.groupby(key)["visitorid"].nunique(),
            "buyers": tx.groupby(key)["visitorid"].nunique(),
            "purchase_count": tx.groupby(key)["transactionid"].nunique(),
            "addtocart_count": d[d["event"] == "addtocart"].groupby(key).size(),
            "event_count": d.groupby(key).size(),
        }
    )
    return table.fillna(0).astype("int64")


def line_chart(x, y, title, unit, hollow, labels, hover_extra=None):
    """선그래프. 한 기간 전체가 관측되지 않은 점은 빈 점으로 표시한다."""
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers+text" if labels else "lines+markers",
            line=dict(color=COLOR, width=2.5),
            marker=dict(
                size=[9 if labels or h else 3 for h in hollow],
                color=["white" if h else COLOR for h in hollow],
                line=dict(color=COLOR, width=2),
            ),
            text=[f"{v:,}" for v in y] if labels else None,
            # 직전 값보다 크게 떨어진 점은 선과 겹치지 않게 아래에 표기
            textposition=["bottom center" if i > 0 and y[i] < y[i - 1] * 0.7 else "top center" for i in range(len(y))],
            customdata=hover_extra,
            hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:,}" + unit + "</b>"
            + ("<br>%{customdata}" if hover_extra is not None else "")
            + "<extra></extra>",
        )
    )
    top = max(list(y) + [1])
    fig.update_layout(
        title=dict(text=f"<b>{title}</b> ({unit})", font=dict(size=15), x=0),
        height=300,
        margin=dict(l=10, r=20, t=50, b=10),
        showlegend=False,
        yaxis=dict(range=[0, top * (1.2 if labels else 1.1)], tickformat=",", rangemode="tozero"),
        xaxis=dict(tickformat="%Y-%m" if labels else "%m-%d", showgrid=False),
    )
    if labels:
        fig.update_xaxes(dtick="M1", tickformat="%Y-%m")
    return fig


def render_category_scatter(fv, start_date, end_date):
    """카테고리별 장바구니 전환율(x)·구매 전환율(y) 산포도. 풍선 크기는 첫 조회 방문자 수, 색은 대분류."""
    st.subheader("카테고리별 전환율 산포도")
    if fv is None:
        st.info("산포도에는 첫 조회 데이터(data/processed/dashboard/first_view.parquet)가 필요합니다. 먼저 `python analysis/19_export_dashboard_data.py`를 실행하세요.")
        return
    c1, c2 = st.columns([2, 3])
    min_v = int(c1.number_input(
        "최소 첫 조회 방문자 수", min_value=1, max_value=100000, value=DEFAULT_MIN_VISITORS, step=50, key="scatter_min",
        help="첫 조회 방문자가 이 값 미만인 카테고리는 전환율이 크게 흔들려 산포도에서 제외합니다.",
    ))
    exclude_early = c2.checkbox(
        "시작 4주(2015-05-03~05-30)에 처음 조회한 방문자 제외", value=False, key="scatter_early",
        help="데이터 시작 전의 행동이 관측되지 않아, 이 기간의 '첫 조회'는 실제 첫 조회가 아닐 수 있습니다.",
    )
    d = fv[(fv["date"] >= start_date) & (fv["date"] <= end_date)]
    if exclude_early:
        d = d[~d["early"]]
    known = d[d["categoryid"] != UNKNOWN]
    g = known.groupby(["root_categoryid", "categoryid"]).agg(
        visitors=("visitorid", "size"), carted=("carted", "sum"), purchased=("purchased", "sum")
    ).reset_index()
    n_all = len(g)
    g = g[g["visitors"] >= min_v].copy()
    if g.empty:
        st.info("조건에 맞는 카테고리가 없습니다. 최소 첫 조회 방문자 수를 낮춰 보세요.")
        return
    g["cart_rate"] = g["carted"] / g["visitors"] * 100
    g["purchase_rate"] = g["purchased"] / g["visitors"] * 100
    overall_cart, overall_buy = d["carted"].mean() * 100, d["purchased"].mean() * 100

    # 같은 대분류는 같은 색. 색은 전체 데이터의 대분류 순서로 고정해 필터가 바뀌어도 색이 바뀌지 않게 한다.
    palette = SCATTER_COLORS
    root_ids = sorted(r for r in fv["root_categoryid"].unique() if r != UNKNOWN)
    color_of = {r: palette[i % len(palette)] for i, r in enumerate(root_ids)}
    sizeref = 2.0 * g["visitors"].max() / (56**2)
    fig = go.Figure()
    for r in g.groupby("root_categoryid")["visitors"].sum().sort_values(ascending=False).index:
        sub = g[g["root_categoryid"] == r]
        fig.add_trace(
            go.Scatter(
                x=sub["cart_rate"], y=sub["purchase_rate"], mode="markers", name=root_label(r),
                marker=dict(size=sub["visitors"], sizemode="area", sizeref=sizeref, sizemin=4, color=color_of[r], opacity=0.66,
                            line=dict(width=0.8, color="white")),
                customdata=sub[["categoryid", "visitors", "carted", "purchased"]].to_numpy(),
                hovertemplate="<b>%{fullData.name} › 카테고리 %{customdata[0]}</b><br>첫 조회 방문자 %{customdata[1]:,}명"
                "<br>장바구니 전환율 %{x:.2f}% (%{customdata[2]:,}명)<br>구매 전환율 %{y:.2f}% (%{customdata[3]:,}명)<extra></extra>",
            )
        )
    fig.add_vline(x=overall_cart, line_dash="dash", line_color="#6b7280", annotation_text=f"전체 평균 {overall_cart:.2f}%",
                  annotation_position="top", annotation_font_size=11)
    fig.add_hline(y=overall_buy, line_dash="dash", line_color="#6b7280", annotation_text=f"전체 평균 {overall_buy:.2f}%",
                  annotation_position="top right", annotation_font_size=11)
    fig.update_layout(
        height=620, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="장바구니 전환율 (%)", ticksuffix="%", rangemode="tozero", showgrid=True),
        yaxis=dict(title="구매 전환율 (%)", ticksuffix="%", rangemode="tozero", showgrid=True),
        legend=dict(title=dict(text="대분류 (범례를 눌러 켜고 끄기)"), itemsizing="constant"),
    )
    st.plotly_chart(fig, width="stretch", key="category-scatter")
    st.caption(
        f"점 하나가 카테고리 하나입니다({len(g):,}개 표시 · 첫 조회 방문자 {min_v:,}명 미만 {n_all - len(g):,}개 제외 · 대분류 {g['root_categoryid'].nunique()}개). "
        f"풍선 크기는 그 카테고리를 첫 조회한 방문자 수, 같은 대분류는 같은 색입니다. 점선은 선택 기간 전체 평균입니다. "
        f"카테고리를 알 수 없는 방문자 {len(d) - len(known):,}명은 제외했습니다."
    )


def render_trend_tab(events, data_start, start_date, end_date, fv=None):
    sel = events[(events["date"] >= start_date) & (events["date"] <= end_date)]
    sel_start = max(start_date, data_start)
    sel_end = min(end_date + pd.Timedelta(days=1), DATA_END)
    observed_days = (sel_end - sel_start) / pd.Timedelta(days=1)
    st.caption(f"선택 기간: {start_date:%Y-%m-%d} ~ {end_date:%Y-%m-%d} · 관측 {observed_days:.1f}일 · 이벤트 {len(sel):,}건")

    st.subheader("주요 지표")
    tx_sel = sel[sel["event"] == "transaction"]
    kpis = {
        "active_users": sel["visitorid"].nunique(),
        "buyers": tx_sel["visitorid"].nunique(),
        "purchase_count": tx_sel["transactionid"].nunique(),
        "addtocart_count": int((sel["event"] == "addtocart").sum()),
        "event_count": len(sel),
    }
    helps = {
        "active_users": "view, addtocart, transaction 중 하나라도 1회 이상 수행한 고유 방문자",
        "buyers": "transaction을 1회 이상 수행한 고유 방문자",
        "purchase_count": "고유 transactionid 수(주문 기준). 한 주문에 여러 상품이 묶여도 1회",
        "addtocart_count": "addtocart 이벤트 행 수",
        "event_count": "전체 이벤트 행 수(view + addtocart + transaction)",
    }
    for col, (key, label, unit) in zip(st.columns(5), METRICS):
        col.metric(f"{label} ({unit})", f"{kpis[key]:,}", help=helps[key], border=True)

    st.subheader("일자별 추이")
    daily = metric_table(sel, "date")
    daily_hollow = [
        (d == data_start.normalize() and data_start > data_start.normalize()) or d == (DATA_END - pd.Timedelta(seconds=1)).normalize()
        for d in daily.index
    ]
    cols_a, cols_b = st.columns(3), st.columns(2)
    for col, (key, label, unit) in zip(list(cols_a) + list(cols_b), METRICS):
        col.plotly_chart(line_chart(daily.index, daily[key].tolist(), label, unit, daily_hollow, labels=False), width="stretch")
    st.caption("빈 점은 하루 전체가 아니라 일부 시간만 관측된 날입니다(데이터 시작일과 종료일). 활성 사용자·구매자는 그날의 고유 방문자입니다.")

    st.subheader("월별 추이")
    monthly = metric_table(sel, "month")
    month_start = monthly.index
    month_end = month_start + pd.offsets.MonthBegin(1)
    lo = pd.Series(month_start).where(pd.Series(month_start) > sel_start, sel_start).values
    hi = pd.Series(month_end).where(pd.Series(month_end) < sel_end, sel_end).values
    obs = (pd.to_datetime(hi) - pd.to_datetime(lo)) / pd.Timedelta(days=1)
    cal = (month_end - month_start).days
    monthly_hollow = [o < c - 1e-9 for o, c in zip(obs, cal)]
    extra = [f"관측 {o:.1f}일 / {c}일" for o, c in zip(obs, cal)]
    cols_a, cols_b = st.columns(3), st.columns(2)
    for col, (key, label, unit) in zip(list(cols_a) + list(cols_b), METRICS):
        col.plotly_chart(
            line_chart(monthly.index, monthly[key].tolist(), label, unit, monthly_hollow, labels=True, hover_extra=extra),
            width="stretch",
        )
    st.caption("빈 점은 한 달 전체가 아니라 일부만 관측되었거나 선택 기간에 일부만 포함된 달입니다. 이 달은 값이 작게 나올 수 있어 다른 달과 직접 비교하지 마세요.")

    render_category_scatter(fv, start_date, end_date)

    with st.expander("지표 정의와 해석 시 유의사항"):
        st.markdown(
            """
- **활성 사용자:** 기간 안에 view, addtocart, transaction 중 하나라도 수행한 고유 방문자(`visitorid`)
- **구매자:** 기간 안에 transaction을 수행한 고유 방문자
- **구매 횟수:** 고유 `transactionid` 수. transaction 행 수와 다릅니다.
- **고유 방문자 지표는 합산할 수 없습니다.** 선택 기간 전체의 활성 사용자와 구매자는 일자별·월별 값을 더한 값이 아니라 기간 전체에서 다시 센 값입니다.
- **UTC 기준:** 날짜와 월 구분은 UTC입니다. 첫날(2015-05-03)은 03:00 이후, 마지막 날(2015-09-15)은 21:00 이전 이벤트만 있습니다.
- **방문자 식별:** `visitorid`가 실제 사람과 1:1인지는 확인되지 않았습니다.
- **시작 4주(2015-05-03~05-30):** 이전 행동이 관측되지 않아 신규·활성 해석에 영향을 줄 수 있습니다.
- **카테고리별 전환율 산포도:** 방문자를 첫 조회한 상품의 카테고리로 나누고, 그 방문자가 첫 조회 *이후* 장바구니에 담았거나 구매한 비율(시간 제한 없음)을 그렸습니다. 첫 조회한 상품·카테고리에서 담거나 구매했다는 뜻이 아닙니다. 첫 조회일이 선택한 기간에 속한 방문자만 집계하며, 카테고리 미확인 방문자와 표본이 작은 카테고리는 뺍니다.
            """
        )


# ── 공통: 히트맵 테이블 ─────────────────────────────────────
# 대분류 구분용 색(색상환을 황금각으로 나누고 명도·채도를 번갈아 바꿔 인접한 색이 비슷해지지 않게 한 24색)
SCATTER_COLORS = [
    "#bf1818", "#3e6cda", "#69aa0e", "#bf18aa", "#3edac0", "#aa620e", "#4118bf", "#45da3e", "#aa0e42", "#1880bf", "#cdda3e", "#890eaa",
    "#18bf6b", "#da5e3e", "#0e1baa", "#57bf18", "#da3ea7", "#0ea4aa", "#bf9418", "#853eda", "#0eaa27", "#bf182d", "#3e80da", "#7daa0e",
]
HEAT_BLUE = ((244, 248, 253), (31, 78, 140))      # 방문자 수·비율: 값이 클수록 진한 파랑
HEAT_ORANGE = ((253, 246, 236), (191, 96, 18))    # 소요 시간(분): 값이 클수록 진한 주황(크다고 좋은 값이 아니므로 색을 구분)
HEAT_TOP_QUANTILE = 0.95                          # 행이 많은 표는 상위 5% 값을 최대 진하기로 맞춰 극단값이 색을 독점하지 않게 함
HEAT_MANY_ROWS = 30


def heat_cell(norm, ramp):
    (r0, g0, b0), (r1, g1, b1) = ramp
    r, g, b = (round(a + (c - a) * norm) for a, c in ((r0, r1), (g0, g1), (b0, b1)))
    return f"background-color: rgb({r},{g},{b}); color: {'#ffffff' if norm > 0.55 else '#1f2933'}"


def heatmap_table(out, blue_cols, formats, orange_cols=(), sort_col=None, height=None):
    """열마다 값이 클수록 배경이 진해지는 히트맵 표. 정렬 기준 열은 머리글에 ▼를 붙인다."""
    name = lambda c: f"▼ {c}" if c == sort_col else c
    shown = out.rename(columns={c: name(c) for c in out.columns})
    styles = pd.DataFrame("", index=shown.index, columns=shown.columns)
    for cols, ramp in ((blue_cols, HEAT_BLUE), (orange_cols, HEAT_ORANGE)):
        for col in cols:
            values = pd.to_numeric(shown[name(col)], errors="coerce")
            if values.notna().sum() == 0:
                continue
            top = values.quantile(HEAT_TOP_QUANTILE) if values.notna().sum() > HEAT_MANY_ROWS else values.max()
            if not top or top <= 0:
                top = values.max() if values.max() > 0 else 1
            norm = values.clip(lower=0, upper=top) / top
            styles[name(col)] = ["" if pd.isna(n) else heat_cell(float(n), ramp) for n in norm]
    styler = shown.style.apply(lambda _: styles, axis=None).format({name(c): f for c, f in formats.items()}, na_rep="-")
    st.dataframe(styler, hide_index=True, width="stretch", height=height, column_config={"순위": st.column_config.NumberColumn(width="small")})


# ── 탭 2: 카테고리별 첫 조회 성과 ───────────────────────────
def root_label(v):
    return "카테고리 미확인" if v == UNKNOWN else f"대분류 {v}"


def cat_label(root, cat):
    return "카테고리 미확인" if cat == UNKNOWN else f"대분류 {root} › 카테고리 {cat}"


METRIC_CHOICES = {
    "첫 조회 방문자 수": "visitors",
    "장바구니 전환율": "cart_rate",
    "구매 전환율": "purchase_rate",
    "30일 재방문율": "r30",
}
METRIC_NAME = {v: k for k, v in METRIC_CHOICES.items()}
DEFAULT_MIN_VISITORS = 100
DEFAULT_MIN_R30 = 1000  # 30일 재방문율은 전체 평균이 1% 미만이라 전환율보다 큰 표본이 필요하다


def rank_table(d, keys, total, sort_by="visitors", min_visitors=0):
    """keys 단위 첫 조회 방문자 수와 이후 장바구니·구매 전환율.

    sort_by 지표의 내림차순으로 정렬한다. 비율로 정렬할 때는 그 비율의 분모(전환율은 방문자 수, 30일 재방문율은
    30일째를 관찰할 수 있는 방문자 수)가 min_visitors 미만인 항목을 순위에서 제외한다(표본이 작으면 비율이 크게 흔들림).
    제외한 항목 수는 attrs["excluded"]에 둔다.
    """
    g = d.groupby(keys).agg(
        visitors=("visitorid", "size"), carted=("carted", "sum"), purchased=("purchased", "sum"),
        obs30=("obs30", "sum"), ret30=("ret30", "sum"),
    ).reset_index()
    g["share"] = g["visitors"] / total * 100
    g["cart_rate"] = g["carted"] / g["visitors"] * 100
    g["purchase_rate"] = g["purchased"] / g["visitors"] * 100
    g["r30"] = g["ret30"] / g["obs30"].where(g["obs30"] > 0) * 100
    excluded = 0
    by_rate = sort_by != "visitors"
    denom = "obs30" if sort_by == "r30" else "visitors"
    threshold = max(min_visitors, 1) if sort_by == "r30" else min_visitors  # 30일째를 관찰할 수 없는 항목은 비율이 없다
    if by_rate and threshold >= 1:
        excluded = int((g[denom] < threshold).sum())
        g = g[g[denom] >= threshold]
    order = [sort_by] + (["visitors"] if by_rate else []) + keys
    g = g.sort_values(order, ascending=[False] * (2 if by_rate else 1) + [True] * len(keys), kind="stable").reset_index(drop=True)
    g.attrs["excluded"] = excluded
    return g


def item_table(d, total, sort_by="visitors", min_visitors=0):
    """상품 단위. 카테고리는 그 상품을 가장 많이 첫 조회한 시점의 카테고리로 표시한다."""
    by_cat = rank_table(d, ["itemid", "root_categoryid", "categoryid"], total)
    dominant = by_cat.drop_duplicates("itemid")[["itemid", "root_categoryid", "categoryid"]]
    items = rank_table(d, ["itemid"], total, sort_by, min_visitors)
    excluded = items.attrs["excluded"]
    items = items.merge(dominant, on="itemid", how="left")
    items.attrs["excluded"] = excluded
    return items


def denom_label(metric):
    return "30일 관찰 가능 방문자" if metric == "r30" else "첫 조회 방문자"


def excluded_note(n, min_visitors, metric="visitors"):
    return f" · {denom_label(metric)} {min_visitors:,}명 미만 {n:,}개 제외" if n else ""


def show_table(df, first_col, metric, limit=None, excluded=0, min_visitors=0):
    """순위·항목·방문자·전환율 히트맵 표. 정렬 기준 열은 머리글에 ▼를 붙인다."""
    if df.empty:
        st.info("조건에 맞는 항목이 없습니다. 최소 첫 조회 방문자 수를 낮춰 보세요.")
        return
    shown = df.head(limit) if limit else df
    out = pd.DataFrame({"순위": range(1, len(shown) + 1)})
    for col, label in first_col:
        out[label] = shown[col].tolist()
    out["첫 조회 방문자"] = shown["visitors"].tolist()
    out["전체 대비(%)"] = shown["share"].round(2).tolist()
    out["장바구니 전환율(%)"] = shown["cart_rate"].round(2).tolist()
    out["구매 전환율(%)"] = shown["purchase_rate"].round(2).tolist()
    out["30일 재방문율(%)"] = shown["r30"].round(2).tolist()
    out["30일 재방문자"] = shown["ret30"].tolist()
    out["30일 관찰 가능"] = shown["obs30"].tolist()
    column_of = {"visitors": "첫 조회 방문자", "cart_rate": "장바구니 전환율(%)", "purchase_rate": "구매 전환율(%)", "r30": "30일 재방문율(%)"}
    metric_cols = ["첫 조회 방문자", "전체 대비(%)", "장바구니 전환율(%)", "구매 전환율(%)", "30일 재방문율(%)", "30일 재방문자", "30일 관찰 가능"]
    heatmap_table(
        out, blue_cols=metric_cols, sort_col=column_of[metric], height=min(38 + 35 * len(out), 460),
        formats={"첫 조회 방문자": "{:,.0f}", "전체 대비(%)": "{:.2f}", "장바구니 전환율(%)": "{:.2f}", "구매 전환율(%)": "{:.2f}",
                 "30일 재방문율(%)": "{:.2f}", "30일 재방문자": "{:,.0f}", "30일 관찰 가능": "{:,.0f}"},
    )
    notes = []
    if limit and len(df) > limit:
        notes.append(f"{METRIC_NAME[metric]} 상위 {limit:,}개만 표시(전체 {len(df):,}개)")
    if excluded:
        notes.append(f"{denom_label(metric)} {min_visitors:,}명 미만 {excluded:,}개는 순위에서 제외")
    if notes:
        st.caption(" · ".join(notes))


def bar_chart(labels, values, visitors, metric, selected=None, overall=None, height=360):
    is_rate = metric != "visitors"
    colors = [COLOR if selected is None or lab == selected else COLOR_LIGHT for lab in labels]
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h", marker_color=colors, customdata=visitors,
            text=[f"{v:.2f}%" if is_rate else f"{v:,}명" for v in values], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br><b>" + ("%{x:.2f}%" if is_rate else "%{x:,}명") + "</b><br>첫 조회 방문자 %{customdata:,}명<extra></extra>",
        )
    )
    top = max(list(values) + ([overall] if overall else []) + [1e-9])
    fig.update_layout(
        height=height, margin=dict(l=10, r=60, t=28 if is_rate else 10, b=10), showlegend=False, bargap=0.3,
        xaxis=dict(tickformat=(",.2f" if top < 2 else ",.1f") if is_rate else ",", ticksuffix="%" if is_rate else "", range=[0, top * 1.3], showgrid=True),
        yaxis=dict(autorange="reversed", showgrid=False),
    )
    if is_rate and overall:
        fig.add_vline(x=overall, line_dash="dash", line_color="#6b7280",
                      annotation_text=f"전체 {overall:.2f}%", annotation_position="top", annotation_font_size=11)
    return fig


def render_first_view_tab(fv, start_date, end_date):
    if fv is None:
        st.error("첫 조회 데이터(data/processed/dashboard/first_view.parquet)가 없습니다. 먼저 `python analysis/19_export_dashboard_data.py`를 실행하세요.")
        return
    exclude_early = st.checkbox(
        "시작 4주(2015-05-03~05-30)에 처음 조회한 방문자 제외",
        value=False, key="fv_early",
        help="데이터 시작 전의 행동이 관측되지 않아, 이 기간의 '첫 조회'는 실제 첫 조회가 아닐 수 있습니다.",
    )
    ctl1, ctl2 = st.columns([3, 2])
    metric_name = ctl1.radio(
        "순위·그래프 기준 지표", list(METRIC_CHOICES), horizontal=True, key="fv_metric",
        help="아래 상위 10 순위, 그래프, 표의 정렬이 이 지표 기준으로 바뀝니다. 네 지표는 모든 표에 함께 표시됩니다.",
    )
    metric = METRIC_CHOICES[metric_name]
    is_r30 = metric == "r30"
    min_v = int(ctl2.number_input(
        "30일 재방문율 순위의 최소 관찰 가능 방문자 수" if is_r30 else "전환율 순위의 최소 첫 조회 방문자 수",
        min_value=1, max_value=100000, value=DEFAULT_MIN_R30 if is_r30 else DEFAULT_MIN_VISITORS, step=500 if is_r30 else 50,
        key="fv_min_r30" if is_r30 else "fv_min_conv", disabled=metric == "visitors",
        help="비율 기준 순위에서는 그 비율의 분모가 이 값 미만인 항목을 제외합니다. 방문자가 적으면 비율이 크게 흔들려 순위가 왜곡됩니다.",
    ))
    d = fv[(fv["date"] >= start_date) & (fv["date"] <= end_date)]
    if exclude_early:
        d = d[~d["early"]]
    total = len(d)
    if total == 0:
        st.info("선택한 조건에 해당하는 첫 조회 방문자가 없습니다.")
        return
    known = d[d["categoryid"] != UNKNOWN]
    obs30_total, ret30_total = int(d["obs30"].sum()), int(d["ret30"].sum())
    r30_total = ret30_total / obs30_total * 100 if obs30_total else None
    overall = {"visitors": None, "cart_rate": d["carted"].mean() * 100, "purchase_rate": d["purchased"].mean() * 100, "r30": r30_total}[metric]
    st.caption(
        f"첫 조회일이 {start_date:%Y-%m-%d} ~ {end_date:%Y-%m-%d}인 방문자 기준 · 방문자마다 관측 기간에서 가장 이른 view 1건을 '첫 조회'로 집계"
    )

    # 상단 지표
    kpi = [
        ("첫 조회 방문자 (명)", f"{total:,}", "view를 1회 이상 한 방문자. 각자의 가장 이른 view 기준"),
        ("카테고리 확인 방문자 (명)", f"{len(known):,}", f"전체의 {len(known) / total * 100:.1f}%. 나머지는 이력 파일에 없거나 이력 시작(2015-05-10) 이전이라 카테고리를 알 수 없음"),
        ("장바구니 전환율", f"{d['carted'].mean() * 100:.2f}%", "첫 조회 이후 addtocart를 1회 이상 한 방문자 비율(상품·카테고리 무관)"),
        ("구매 전환율", f"{d['purchased'].mean() * 100:.2f}%", "첫 조회 이후 transaction을 1회 이상 한 방문자 비율(상품·카테고리 무관)"),
        ("30일 재방문율", "-" if r30_total is None else f"{r30_total:.2f}%",
         f"첫 조회일로부터 정확히 30일째 날짜에 이벤트(종류·상품 무관)가 있는 방문자 비율. 30일째를 관찰할 수 있는 {obs30_total:,}명 중 {ret30_total:,}명"),
    ]
    for col, (label, value, help_text) in zip(st.columns(5), kpi):
        col.metric(label, value, help=help_text, border=True)

    # 주요 성과: 상위 10
    st.subheader(f"주요 성과 · 상위 10 (기준: {metric_name})")
    st.caption(
        "첫 조회 방문자 수, 장바구니 전환율, 구매 전환율, 30일 재방문율을 함께 보여 줍니다. 전환율은 그 항목을 첫 조회한 방문자가 이후 장바구니에 담거나 구매한 비율이고, "
        "30일 재방문율은 첫 조회일로부터 정확히 30일째에 돌아온 비율입니다. 전체 평균과 비교해 읽으세요. 표의 배경색은 열마다 값이 클수록 진하고, ▼가 붙은 열이 정렬 기준입니다."
    )
    root_rank = rank_table(known, ["root_categoryid"], total, metric, min_v)
    cat_rank = rank_table(known, ["root_categoryid", "categoryid"], total, metric, min_v)
    item_rank = item_table(d, total, metric, min_v)
    ex = lambda t: dict(excluded=t.attrs.get("excluded", 0), min_visitors=min_v)

    with st.container(border=True):
        st.markdown("**대분류 기준**")
        r = root_rank.head(10).copy()
        r["label"] = [root_label(v) for v in r["root_categoryid"]]
        show_table(r, [("label", "대분류")], metric, **ex(root_rank))
    with st.container(border=True):
        st.markdown("**카테고리 기준**")
        r = cat_rank.head(10).copy()
        r["root"] = [root_label(v) for v in r["root_categoryid"]]
        r["cat"] = [f"카테고리 {v}" for v in r["categoryid"]]
        show_table(r, [("root", "대분류"), ("cat", "카테고리")], metric, **ex(cat_rank))
    with st.container(border=True):
        st.markdown("**아이템(상품) 기준**")
        r = item_rank.head(10).copy()
        r["item"] = [f"상품 {v}" for v in r["itemid"]]
        r["root"] = [root_label(v) if v != UNKNOWN else "카테고리 미확인" for v in r["root_categoryid"]]
        r["cat"] = ["-" if v == UNKNOWN else f"카테고리 {v}" for v in r["categoryid"]]
        show_table(r, [("item", "상품"), ("root", "대분류"), ("cat", "카테고리")], metric, **ex(item_rank))
    st.caption(
        f"전체 평균: 장바구니 전환율 {d['carted'].mean() * 100:.2f}%, 구매 전환율 {d['purchased'].mean() * 100:.2f}%, "
        f"30일 재방문율 {'-' if r30_total is None else f'{r30_total:.2f}%'}. "
        "카테고리 미확인 방문자는 대분류·카테고리 순위에서 제외했고, 아이템 순위에는 포함했습니다. 상품의 카테고리는 가장 많이 첫 조회된 시점의 값입니다."
    )

    def option_text(label, visitors, row):
        extra = "" if metric == "visitors" else f" · {METRIC_NAME[metric]} {row[metric]:.2f}%"
        return f"{label} ({visitors:,}명{extra})"

    # 드릴다운
    st.subheader("세부 탐색 · 대분류 → 카테고리 → 상품")
    st.caption(
        "대분류를 고르면 아래 카테고리가, 카테고리를 고르면 아래 상품이 그 범위로 바뀝니다. '전체'를 고르면 해당 단계의 제한이 없습니다. "
        f"정렬·그래프 기준: {metric_name}"
    )

    # 1단계: 대분류
    st.markdown("##### 1단계 · 대분류")
    root_all = rank_table(d, ["root_categoryid"], total, metric, min_v)
    root_excluded = root_all.attrs["excluded"]
    root_all = pd.concat([root_all[root_all["root_categoryid"] != UNKNOWN], root_all[root_all["root_categoryid"] == UNKNOWN]], ignore_index=True)
    root_rows = {r["root_categoryid"]: r for _, r in root_all.iterrows()}
    root_sel = st.selectbox(
        "대분류 선택", [None] + root_all["root_categoryid"].tolist(), key="fv_root",
        format_func=lambda v: f"전체 ({total:,}명)" if v is None else option_text(root_label(v), int(root_rows[v]["visitors"]), root_rows[v]),
    )
    c1, c2 = st.columns([2, 3])
    top_roots = root_all[root_all["root_categoryid"] != UNKNOWN].head(12)
    with c1:
        st.plotly_chart(
            bar_chart([root_label(v) for v in top_roots["root_categoryid"]], top_roots[metric].tolist(), top_roots["visitors"].tolist(), metric,
                      selected=None if root_sel is None else root_label(root_sel), overall=overall),
            width="stretch",
        )
        st.caption(f"{metric_name} 상위 12개 대분류(카테고리 미확인 제외)")
    with c2:
        t = root_all.copy()
        t["label"] = [root_label(v) for v in t["root_categoryid"]]
        show_table(t, [("label", "대분류")], metric, excluded=root_excluded, min_visitors=min_v)

    scope = d if root_sel is None else d[d["root_categoryid"] == root_sel]

    # 2단계: 카테고리
    st.markdown("##### 2단계 · 카테고리" + ("" if root_sel in (None, UNKNOWN) else f" ({root_label(root_sel)} 안)"))
    cat_all = rank_table(scope[scope["categoryid"] != UNKNOWN], ["root_categoryid", "categoryid"], total, metric, min_v)
    cat_excluded = cat_all.attrs["excluded"]
    cat_sel = None
    if root_sel == UNKNOWN:
        st.info("카테고리를 알 수 없는 방문자라 세부 카테고리가 없습니다. 아래 3단계에서 상품은 볼 수 있습니다.")
    elif cat_all.empty:
        st.info("이 범위에는 조건에 맞는 카테고리가 없습니다." if cat_excluded else "이 범위에는 카테고리가 확인된 첫 조회가 없습니다.")
    else:
        cat_rows = {(r["root_categoryid"], r["categoryid"]): r for _, r in cat_all.iterrows()}
        cat_sel = st.selectbox(
            "카테고리 선택", [None] + list(cat_rows), key="fv_cat",
            format_func=lambda v: f"전체 ({int(cat_all['visitors'].sum()):,}명)" if v is None
            else option_text(cat_label(*v), int(cat_rows[v]["visitors"]), cat_rows[v]),
        )
        c1, c2 = st.columns([2, 3])
        top_cats = cat_all.head(12)
        with c1:
            st.plotly_chart(
                bar_chart([cat_label(r, c) for r, c in zip(top_cats["root_categoryid"], top_cats["categoryid"])],
                          top_cats[metric].tolist(), top_cats["visitors"].tolist(), metric,
                          selected=None if cat_sel is None else cat_label(*cat_sel), overall=overall),
                width="stretch",
            )
            st.caption(f"{metric_name} 상위 12개 카테고리")
        with c2:
            t = cat_all.copy()
            t["root"] = [root_label(v) for v in t["root_categoryid"]]
            t["cat"] = [f"카테고리 {v}" for v in t["categoryid"]]
            show_table(t, [("root", "대분류"), ("cat", "카테고리")], metric, limit=TABLE_ROWS, excluded=cat_excluded, min_visitors=min_v)
        if cat_sel is not None:
            scope = scope[scope["categoryid"] == cat_sel[1]]

    # 3단계: 상품
    where = "전체" if root_sel is None else root_label(root_sel)
    if cat_sel is not None:
        where += f" › 카테고리 {cat_sel[1]}"
    st.markdown(f"##### 3단계 · 상품 ({where})")
    items = item_table(scope, total, metric, min_v)
    item_excluded = items.attrs["excluded"]
    st.caption(f"이 범위의 첫 조회 방문자 {len(scope):,}명 · 상품 {len(items) + item_excluded:,}개{excluded_note(item_excluded, min_v, metric)}")
    c1, c2 = st.columns([2, 3])
    top_items = items.head(12)
    with c1:
        if not top_items.empty:
            st.plotly_chart(
                bar_chart([f"상품 {v}" for v in top_items["itemid"]], top_items[metric].tolist(), top_items["visitors"].tolist(), metric, overall=overall),
                width="stretch",
            )
            st.caption(f"{metric_name} 상위 12개 상품")
    with c2:
        t = items.copy()
        t["item"] = [f"상품 {v}" for v in t["itemid"]]
        t["where"] = [cat_label(r, c) for r, c in zip(t["root_categoryid"], t["categoryid"])]
        show_table(t, [("item", "상품"), ("where", "대분류 › 카테고리")], metric, limit=TABLE_ROWS)

    with st.expander("지표 정의와 해석 시 유의사항"):
        st.markdown(
            """
- **첫 조회:** 방문자마다 관측 기간에서 가장 이른 `view` 1건입니다. 같은 시각이면 파일 순서의 첫 행을 씁니다. 첫 조회일이 선택한 기간에 속한 방문자만 집계합니다.
- **카테고리:** 첫 조회 시각 이전(같은 시각 포함)에 적용된 가장 최근 값입니다. 이후 시점의 값으로 채우지 않습니다. 카테고리 이름은 제공되지 않아 ID로 표시합니다.
- **카테고리 미확인:** 이력 파일에 없는 상품이거나, 상품의 첫 이력(가장 이른 2015-05-10 03:00 UTC) 이전 이벤트입니다. 5월 첫 조회일수록 미확인 비중이 높습니다.
- **장바구니 전환율:** 그 항목을 첫 조회한 방문자 중, 첫 조회 이후 장바구니에 1회 이상 담은 방문자의 비율입니다. 분모는 그 항목의 첫 조회 방문자입니다.
- **구매 전환율:** 같은 방문자 중 첫 조회 이후 구매(transaction)를 1회 이상 한 방문자의 비율입니다. 장바구니를 거치지 않은 구매도 포함하며, '장바구니 담은 방문자 중 구매 비율'이 아닙니다.
- **전환율 해석:** 첫 조회한 상품·카테고리에서 담거나 구매했다는 뜻이 아니라, 그 방문자가 이후 어떤 상품이든 담거나 구매했다는 뜻입니다. 전체 평균과 비교해 읽으세요.
- **30일 재방문율:** 그 항목을 첫 조회한 방문자 중 첫 조회일로부터 정확히 30일째 UTC 날짜에 이벤트(view·장바구니·구매, 상품·카테고리 무관)가 있는 방문자의 비율입니다. 1~29일째나 31일째 이후의 활동은 세지 않습니다. 분모는 30일째가 데이터 기간(2015-09-15까지) 안에 들어오는 방문자이며, 표의 '30일 관찰 가능'입니다. 관찰 가능 방문자가 0명이면 표에 None으로 표시합니다.
- **30일 재방문율의 표본:** 전체 평균이 1%보다 훨씬 낮아 재방문자가 몇 명만 달라져도 값이 크게 움직입니다. '30일 재방문자' 수를 함께 보고, 재방문자가 적은 항목의 순위 차이는 크게 해석하지 마세요.
- **리텐션 탭과의 차이:** 이 탭은 다른 지표와 같은 방문자 집단을 비교하도록 첫 조회(view)를 Day 0으로 씁니다. 리텐션 탭의 재방문율은 첫 이벤트(종류 무관)를 Day 0으로 쓰므로 값이 조금 다를 수 있습니다.
- **최소 방문자 수:** 방문자가 적은 항목은 비율이 크게 흔들려(예: 3명 중 1명 = 33%) 비율 순위에서 제외합니다. 기본값은 전환율 100명, 30일 재방문율 1,000명(30일 관찰 가능 방문자 기준)이며, 화면에서 바꿀 수 있습니다. 방문자 수 기준 순위에는 적용하지 않습니다.
- **상품의 카테고리:** 여러 카테고리로 첫 조회된 상품은 가장 많이 첫 조회된 카테고리로 표시합니다. 같은 상품이 카테고리별 탐색에서는 각 카테고리 범위에서 따로 집계됩니다.
- **표의 색:** 열마다 값이 클수록 배경이 진해집니다. 행이 30개를 넘는 표는 상위 5% 값을 가장 진한 색으로 맞춥니다. 열끼리 색을 비교하지 말고, 같은 열 안에서 비교하세요.
- **시작 4주:** 데이터 시작 전의 조회가 보이지 않아 실제 첫 조회가 아닐 수 있습니다. 위 체크박스로 제외할 수 있습니다.
- **UTC 기준:** 날짜는 UTC입니다. `visitorid`가 실제 사람과 1:1인지는 확인되지 않았습니다.
            """
        )


# ── 탭 3: 퍼널 (24시간 전환 기준) ───────────────────────────
FUNNEL_METRICS = {
    "View → 구매 전환율": "v2b",
    "View → 장바구니 전환율": "v2c",
    "장바구니 → 구매 전환율": "c2b",
    "첫 조회 방문자 수": "visitors",
}
FUNNEL_NAME = {v: k for k, v in FUNNEL_METRICS.items()}
TIME_EDGES = [0, 1, 5, 30, 60, 180, 360, 720, 1440]  # 분
TIME_LABELS = ["1분 이내", "1~5분", "5~30분", "30분~1시간", "1~3시간", "3~6시간", "6~12시간", "12~24시간"]
COUNT_EDGES = [0, 1, 2, 3, 4, 5, 9, float("inf")]
COUNT_LABELS = ["1회", "2회", "3회", "4회", "5회", "6~9회", "10회 이상"]
STAGE_COLORS = ["#1f4e8c", "#2a6fbb", "#5b97d3"]


@st.cache_data(show_spinner="퍼널 데이터를 불러오는 중입니다...")
def load_funnel():
    if not FUNNEL.exists():
        return None
    f = pd.read_parquet(FUNNEL)  # 소요 시간·횟수는 24시간 이내 전환한 방문자만 값이 있고 나머지는 -1(결측)
    f["root_categoryid"] = f["root_categoryid"].astype("int32")
    f["categoryid"] = f["categoryid"].astype("int32")
    f["date"] = day_to_datetime(f["day"])
    f["v2c_min"] = (f["v2c_ms"].astype("float64") / 60000).where(f["v2c_ms"] >= 0)
    f["c2b_min"] = (f["c2b_ms"].astype("float64") / 60000).where(f["c2b_ms"] >= 0)
    f["views_n"] = f["views_n"].astype("float64").where(f["views_n"] >= 0)
    f["carts_n"] = f["carts_n"].astype("float64").where(f["carts_n"] >= 0)
    return f[["date", "early", "obs", "root_categoryid", "categoryid", "c24", "b24", "v2c_min", "c2b_min", "views_n", "carts_n"]]


def funnel_agg(s, keys):
    g = s.groupby(keys).agg(
        visitors=("c24", "size"), carted=("c24", "sum"), bought=("b24", "sum"),
        med_v2c=("v2c_min", "median"), med_c2b=("c2b_min", "median"),
    ).reset_index()
    g["v2c"] = g["carted"] / g["visitors"] * 100
    g["c2b"] = g["bought"] / g["carted"].where(g["carted"] > 0) * 100
    g["v2b"] = g["bought"] / g["visitors"] * 100
    return g


def rank_funnel(g, metric, min_n):
    """선택 지표 내림차순. 전환율은 그 지표의 분모(장바구니→구매는 장바구니 방문자)가 min_n 미만이면 제외."""
    excluded = 0
    if metric != "visitors":
        denom = "carted" if metric == "c2b" else "visitors"
        excluded = int((g[denom] < min_n).sum())
        g = g[g[denom] >= min_n]
    g = g.sort_values([metric, "visitors"], ascending=[False, False], na_position="last", kind="stable").reset_index(drop=True)
    g.attrs["excluded"] = excluded
    return g


def funnel_chart(v, c, b):
    rate = lambda a, p: f"{a / p * 100:.2f}%" if p else "-"
    fig = go.Figure(
        go.Funnel(
            y=["View", "장바구니 담기", "구매"], x=[v, c, b], textposition=["inside", "outside", "outside"],
            insidetextfont=dict(color="white", size=14), outsidetextfont=dict(size=13),
            text=[f"{v:,}명", f"{c:,}명 · 이전 단계 대비 {rate(c, v)}", f"{b:,}명 · 이전 단계 대비 {rate(b, c)}"],
            textinfo="text", marker=dict(color=STAGE_COLORS), connector=dict(fillcolor="rgba(42,111,187,0.15)", line=dict(width=0)),
            hovertemplate="%{y}<br><b>%{x:,}명</b><extra></extra>",
        )
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, funnelmode="stack")
    return fig


def dist_chart(values, edges, labels, title, summary):
    """구간별 방문자 수 막대그래프."""
    values = values.dropna()
    if values.empty:
        return None
    counts = pd.cut(values, bins=edges, labels=labels, include_lowest=True, right=True).value_counts().reindex(labels, fill_value=0)
    pct = counts / counts.sum() * 100
    fig = go.Figure(
        go.Bar(
            x=labels, y=counts.tolist(), marker_color=COLOR, customdata=pct.round(1).tolist(),
            text=[f"{p:.1f}%<br>{n:,}" for p, n in zip(pct, counts)], textposition="outside", cliponaxis=False, textfont=dict(size=11),
            hovertemplate="%{x}<br><b>%{y:,}명</b> (%{customdata}%)<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:12px'>{summary}</span>", font=dict(size=14), x=0),
        height=330, margin=dict(l=10, r=10, t=70, b=10), showlegend=False, bargap=0.25,
        yaxis=dict(tickformat=",", range=[0, counts.max() * 1.3], showgrid=True), xaxis=dict(showgrid=False),
    )
    return fig


def funnel_panel(s, key):
    """한 집단의 24시간 퍼널·전환 소요 시간 분포·이전 단계 수행 횟수 분포."""
    v, c, b = len(s), int(s["c24"].sum()), int(s["b24"].sum())
    left, right = st.columns([3, 2])
    left.plotly_chart(funnel_chart(v, c, b), width="stretch", key=f"{key}-funnel")
    with right:
        st.metric("View → 장바구니 전환율", f"{c / v * 100:.2f}%" if v else "-", help="첫 조회 후 24시간 이내 장바구니에 담은 방문자 / View 방문자", border=True)
        st.metric("장바구니 → 구매 전환율", f"{b / c * 100:.2f}%" if c else "-", help="장바구니에 담은 방문자 중 첫 조회 후 24시간 이내 구매한 방문자 비율", border=True)
        st.metric("View → 구매 전환율", f"{b / v * 100:.2f}%" if v else "-", help="View 방문자 중 첫 조회 후 24시간 이내 구매까지 이어진 비율", border=True)

    t1, t2 = st.columns(2)
    for col, name, vals, tag in [
        (t1, "View → 장바구니 소요 시간", s["v2c_min"], "t1"),
        (t2, "장바구니 → 구매 소요 시간", s["c2b_min"], "t2"),
    ]:
        vals = vals.dropna()
        fig = dist_chart(vals, TIME_EDGES, TIME_LABELS, name, f"전환 {len(vals):,}명 · 중앙값 {vals.median():.1f}분" if len(vals) else "") if len(vals) else None
        if fig is None:
            col.info(f"{name}: 전환한 방문자가 없습니다.")
        else:
            col.plotly_chart(fig, width="stretch", key=f"{key}-{tag}")
    n1, n2 = st.columns(2)
    for col, name, vals, tag in [
        (n1, "장바구니 전환 전 조회(view) 횟수", s["views_n"], "n1"),
        (n2, "구매 전환 전 장바구니 담기 횟수", s["carts_n"], "n2"),
    ]:
        vals = vals.dropna()
        fig = dist_chart(vals, COUNT_EDGES, COUNT_LABELS, name, f"전환 {len(vals):,}명 · 평균 {vals.mean():.2f}회 · 중앙값 {vals.median():.0f}회") if len(vals) else None
        if fig is None:
            col.info(f"{name}: 전환한 방문자가 없습니다.")
        else:
            col.plotly_chart(fig, width="stretch", key=f"{key}-{tag}")


def funnel_table(g, first_col, metric, limit=None, excluded=0, min_n=0):
    """퍼널 비교 히트맵 표. 방문자 수·전환율은 파랑, 전환 소요 시간은 주황으로 값이 클수록 진합니다."""
    if g.empty:
        st.info("조건에 맞는 항목이 없습니다. 최소 표본 수를 낮춰 보세요.")
        return
    shown = g.head(limit) if limit else g
    out = pd.DataFrame({"순위": range(1, len(shown) + 1)})
    for col, label in first_col:
        out[label] = shown[col].tolist()
    out["첫 조회 방문자"] = shown["visitors"].tolist()
    out["장바구니 방문자"] = shown["carted"].tolist()
    out["구매 방문자"] = shown["bought"].tolist()
    out["View→장바구니(%)"] = shown["v2c"].round(2).tolist()
    out["장바구니→구매(%)"] = shown["c2b"].round(2).tolist()
    out["View→구매(%)"] = shown["v2b"].round(2).tolist()
    out["담기까지 중앙값(분)"] = shown["med_v2c"].round(1).tolist()
    out["구매까지 중앙값(분)"] = shown["med_c2b"].round(1).tolist()
    column_of = {"visitors": "첫 조회 방문자", "v2c": "View→장바구니(%)", "c2b": "장바구니→구매(%)", "v2b": "View→구매(%)"}
    count_cols = ["첫 조회 방문자", "장바구니 방문자", "구매 방문자"]
    rate_cols = ["View→장바구니(%)", "장바구니→구매(%)", "View→구매(%)"]
    time_cols = ["담기까지 중앙값(분)", "구매까지 중앙값(분)"]
    formats = {**{c: "{:,.0f}" for c in count_cols}, **{c: "{:.2f}" for c in rate_cols}, **{c: "{:.1f}" for c in time_cols}}
    heatmap_table(out, blue_cols=count_cols + rate_cols, orange_cols=time_cols, formats=formats, sort_col=column_of[metric],
                  height=min(38 + 35 * len(out), 460))
    notes = []
    if limit and len(g) > limit:
        notes.append(f"{FUNNEL_NAME[metric]} 상위 {limit:,}개만 표시(전체 {len(g):,}개)")
    if excluded:
        notes.append(f"표본 {min_n:,}명 미만 {excluded:,}개는 순위에서 제외")
    if notes:
        st.caption(" · ".join(notes))


def render_funnel_tab(f, start_date, end_date):
    if f is None:
        st.error("퍼널 데이터(data/processed/dashboard/funnel.parquet)가 없습니다. 먼저 `python analysis/19_export_dashboard_data.py`를 실행하세요.")
        return
    exclude_early = st.checkbox(
        "시작 4주(2015-05-03~05-30)에 처음 조회한 방문자 제외", value=False, key="funnel_early",
        help="데이터 시작 전의 행동이 관측되지 않아, 이 기간의 '첫 조회'는 실제 첫 조회가 아닐 수 있습니다.",
    )
    ctl1, ctl2 = st.columns([3, 2])
    metric_name = ctl1.radio(
        "비교 지표 (순위·그래프 기준)", list(FUNNEL_METRICS), horizontal=True, key="funnel_metric",
        help="대분류·카테고리 비교의 순위, 막대 그래프, 표 정렬이 이 지표 기준으로 바뀝니다.",
    )
    metric = FUNNEL_METRICS[metric_name]
    min_n = int(ctl2.number_input(
        "전환율 순위의 최소 표본 수", min_value=1, max_value=100000, value=DEFAULT_MIN_VISITORS, step=50, key="funnel_min_n",
        disabled=metric == "visitors",
        help="전환율 순위에서 그 지표의 분모(View→장바구니·View→구매는 첫 조회 방문자, 장바구니→구매는 장바구니 방문자)가 이 값 미만인 항목을 제외합니다.",
    ))
    d0 = f[(f["date"] >= start_date) & (f["date"] <= end_date)]
    if exclude_early:
        d0 = d0[~d0["early"]]
    d = d0[d0["obs"]]
    if d.empty:
        st.info("선택한 조건에 해당하는 방문자가 없습니다.")
        return
    st.caption(
        f"기준: 첫 조회 후 **24시간 이내** 전환 · 첫 조회일 {start_date:%Y-%m-%d} ~ {end_date:%Y-%m-%d} · 분석 대상 방문자 {len(d):,}명 "
        f"(마지막 24시간에 처음 조회해 24시간을 다 관찰할 수 없는 {len(d0) - len(d):,}명 제외) · "
        "표의 배경색은 열마다 값이 클수록 진합니다(방문자 수·전환율은 파랑, 소요 시간은 주황)"
    )

    overall = {"v2c": d["c24"].mean() * 100, "c2b": d["b24"].sum() / max(d["c24"].sum(), 1) * 100, "v2b": d["b24"].mean() * 100, "visitors": None}
    root_name = {-1: "카테고리 미확인"}

    # 전체 퍼널
    st.subheader("전체 퍼널")
    funnel_panel(d, "all")

    # 대분류별 퍼널
    st.subheader("대분류별 퍼널")
    st.caption(f"방문자의 첫 조회 상품이 속한 대분류별 퍼널입니다(전환은 상품·카테고리 무관). 비교 기준: {metric_name}")
    g_root = funnel_agg(d, ["root_categoryid"])
    r_rank = rank_funnel(g_root, metric, min_n)
    r_ex = r_rank.attrs["excluded"]
    c1, c2 = st.columns([2, 3])
    top = r_rank[r_rank["root_categoryid"] != UNKNOWN].head(12)
    with c1:
        if not top.empty:
            st.plotly_chart(
                bar_chart([root_label(v) for v in top["root_categoryid"]], top[metric].tolist(), top["visitors"].tolist(), metric, overall=overall[metric]),
                width="stretch", key="root-compare",
            )
            st.caption(f"{metric_name} 상위 12개 대분류(카테고리 미확인 제외)")
    with c2:
        t = r_rank.copy()
        t["label"] = [root_label(v) for v in t["root_categoryid"]]
        funnel_table(t, [("label", "대분류")], metric, excluded=r_ex, min_n=min_n)
    root_rows = g_root.sort_values("visitors", ascending=False)
    root_choice = st.selectbox(
        "퍼널을 볼 대분류", root_rows["root_categoryid"].tolist(), key="funnel_root",
        format_func=lambda v: f"{root_label(v)} ({int(root_rows.set_index('root_categoryid').loc[v, 'visitors']):,}명)",
    )
    st.markdown(f"**{root_label(root_choice)} 퍼널**")
    funnel_panel(d[d["root_categoryid"] == root_choice], f"root-{root_choice}")

    # 카테고리별 퍼널
    st.subheader("카테고리별 퍼널")
    scope_root = st.selectbox(
        "카테고리를 볼 대분류 범위", [None] + [v for v in root_rows["root_categoryid"].tolist() if v != UNKNOWN], key="funnel_cat_root",
        format_func=lambda v: "전체 대분류" if v is None else root_label(v),
    )
    s = d[d["categoryid"] != UNKNOWN]
    if scope_root is not None:
        s = s[s["root_categoryid"] == scope_root]
    g_cat = funnel_agg(s, ["root_categoryid", "categoryid"])
    c_rank = rank_funnel(g_cat, metric, min_n)
    c_ex = c_rank.attrs["excluded"]
    c1, c2 = st.columns([2, 3])
    top = c_rank.head(12)
    with c1:
        if not top.empty:
            st.plotly_chart(
                bar_chart([cat_label(r, c) for r, c in zip(top["root_categoryid"], top["categoryid"])], top[metric].tolist(), top["visitors"].tolist(), metric,
                          overall=overall[metric]),
                width="stretch", key="cat-compare",
            )
            st.caption(f"{metric_name} 상위 12개 카테고리")
    with c2:
        t = c_rank.copy()
        t["root"] = [root_label(v) for v in t["root_categoryid"]]
        t["cat"] = [f"카테고리 {v}" for v in t["categoryid"]]
        funnel_table(t, [("root", "대분류"), ("cat", "카테고리")], metric, limit=TABLE_ROWS, excluded=c_ex, min_n=min_n)
    if g_cat.empty:
        st.info("이 범위에는 카테고리가 확인된 방문자가 없습니다.")
    else:
        cat_rows = g_cat.sort_values("visitors", ascending=False).set_index(["root_categoryid", "categoryid"])
        cat_choice = st.selectbox(
            "퍼널을 볼 카테고리", list(cat_rows.index), key="funnel_cat",
            format_func=lambda k: f"{cat_label(*k)} ({int(cat_rows.loc[k, 'visitors']):,}명)",
        )
        st.markdown(f"**{cat_label(*cat_choice)} 퍼널**")
        funnel_panel(s[(s["root_categoryid"] == cat_choice[0]) & (s["categoryid"] == cat_choice[1])], f"cat-{cat_choice[0]}-{cat_choice[1]}")

    with st.expander("지표 정의와 해석 시 유의사항"):
        st.markdown(
            """
- **퍼널:** 고유 방문자 기준 View → 장바구니 담기 → 구매입니다. 첫 View 이후에 장바구니, 그 이후에 구매가 있어야 하고(같은 시각 포함), 상품과 카테고리는 따지지 않습니다.
- **24시간 기준:** 첫 View로부터 장바구니와 구매가 모두 24시간 이내여야 전환으로 봅니다. 시간 제한이 없는 퍼널보다 전환이 적게 잡힙니다.
- **관찰 가능 방문자:** 데이터가 2015-09-15 21:00 UTC에서 끝나므로, 그 전 24시간 안에 처음 조회한 방문자는 24시간을 끝까지 관찰할 수 없어 제외했습니다.
- **단계별 전환율:** 장바구니→구매는 *앞 단계를 수행한 방문자 중* 다음 단계를 수행한 비율입니다.
- **카테고리 기준:** 방문자의 첫 조회 상품이 속한 카테고리(첫 조회 시점 값)로 방문자를 나눕니다. 그 방문자가 이후 어떤 상품을 담거나 구매했는지는 따지지 않습니다. 카테고리를 알 수 없는 방문자는 '카테고리 미확인'으로 따로 둡니다.
- **전환 소요 시간 분포:** 24시간 이내에 그 단계로 전환한 방문자만 대상입니다. View→장바구니는 첫 View부터, 장바구니→구매는 첫 장바구니 담기부터 잰 시간입니다.
- **이전 단계 수행 횟수:** 장바구니 전환 방문자는 첫 View부터 첫 장바구니 담기까지(양끝 포함)의 view 이벤트 수, 구매 전환 방문자는 첫 장바구니 담기부터 구매까지(양끝 포함)의 addtocart 이벤트 수입니다.
- **최소 표본 수:** 표본이 작은 항목은 전환율이 크게 흔들려 순위에서 제외합니다. 기본값은 100명이며 바꿀 수 있습니다.
- **표의 색:** 열마다 값이 클수록 배경이 진해집니다(방문자 수·전환율은 파랑, 전환 소요 시간은 주황). 행이 30개를 넘는 표는 상위 5% 값을 가장 진한 색으로 맞춥니다. 열끼리 색을 비교하지 말고, 같은 열 안에서 비교하세요.
- **시작 4주(2015-05-03~05-30):** 이전 행동이 관측되지 않아 첫 조회가 실제 첫 조회가 아닐 수 있습니다. 위 체크박스로 제외할 수 있습니다.
            """
        )


# ── 탭 4: 리텐션 ─────────────────────────────────────────
RET_METRICS = {
    "revisit": {"name": "재방문율", "target": "방문자", "day0": "선택 기간의 첫 이벤트일", "back": "전체 이벤트"},
    "repurchase": {"name": "재구매율", "target": "구매자", "day0": "선택 기간의 첫 구매일", "back": "구매"},
    "buyer_revisit": {"name": "구매자의 재방문율", "target": "구매자", "day0": "선택 기간의 첫 구매일", "back": "전체 이벤트"},
}
RET_NAME_TO_KEY = {v["name"]: k for k, v in RET_METRICS.items()}
RET_MIN_POINT = 30  # 관찰 가능 방문자가 이보다 적은 N은 값이 지나치게 흔들려 그래프와 표에 표시하지 않는다
Z95 = 1.959964
COHORT_TOP_QUANTILE = 0.99  # 코호트 히트맵: N >= 1 칸 재방문율의 이 분위수를 가장 진한 색으로 맞춘다


def period_index(dates, grain):
    """날짜(Series)를 기준별 기간 번호로 바꾼다. 주는 월요일 시작이다."""
    if grain == "day":
        return (dates - pd.Timestamp("1970-01-01")).dt.days
    if grain == "week":
        return ((dates - pd.Timestamp("1970-01-01")).dt.days + 3) // 7
    return dates.dt.year * 12 + dates.dt.month - 1


def period_of(date, grain):
    return int(period_index(pd.Series([date]), grain).iloc[0])


# 마지막 '온전한' 기간. 일별은 앞선 분석 기준대로 마지막 관측일을 포함하고, 주·월은 일부만 관측된 마지막 주(09-14~)·달(9월)을 뺀다.
RET_LAST_COMPLETE = {
    "day": period_of(LAST_DATE, "day"),
    "week": period_of(LAST_DATE + pd.Timedelta(days=1), "week") - 1,
    "month": period_of(LAST_DATE + pd.Timedelta(days=1), "month") - 1,
}
RET_GRAINS = {
    "day": {"label": "일별", "axis": "N-day", "prefix": "D", "word": "일", "show": (1, 7, 30), "key_n": 30, "table_n": (1, 2, 3, 7, 14, 30, 60, 90), "default": (1, 30)},
    "week": {"label": "주별", "axis": "N-week", "prefix": "W", "word": "주", "show": (1, 2, 4), "key_n": 4, "table_n": (1, 2, 3, 4, 8, 12, 16), "default": (1, 12)},
    "month": {"label": "월별", "axis": "N-month", "prefix": "M", "word": "개월", "show": (1, 2, 3), "key_n": 1, "table_n": (1, 2, 3), "default": (1, 3)},
}
for _grain, _spec in RET_GRAINS.items():
    _spec["max_n"] = RET_LAST_COMPLETE[_grain] - period_of(pd.Timestamp("2015-05-03"), _grain)
RET_LABEL_TO_GRAIN = {v["label"]: k for k, v in RET_GRAINS.items()}


@st.cache_data(show_spinner="리텐션 데이터를 불러오는 중입니다...")
def load_retention():
    if not RETENTION.exists():
        return None
    v = pd.read_parquet(RETENTION)
    for col in ["root_categoryid", "categoryid", "buy_root_categoryid", "buy_categoryid"]:
        v[col] = v[col].astype("int32")
    v["date"] = day_to_datetime(v["day"]).astype("datetime64[ns]")
    v = v.drop(columns="day")
    return v.sort_values(["visitorid", "date"], kind="stable").reset_index(drop=True)


@st.cache_data(show_spinner="선택한 방문일의 리텐션을 계산하는 중입니다...", max_entries=48)
def retention_frame(_v, metric, grain, start_date, end_date):
    """선택 기간에 방문(구매 지표는 구매)한 방문자의 Day 0·카테고리·N별 재방문자 집계.

    Day 0은 선택 기간 안의 첫 활동일이고 카테고리는 그날 첫 행동의 값이다. 재방문은 Day 0 이후라면 선택 기간 밖이어도 센다.
    """
    in_range = _v["date"].between(start_date, end_date)
    if metric == "revisit":
        anchor = _v.loc[in_range, ["visitorid", "date", "root_categoryid", "categoryid"]]
    else:
        anchor = _v.loc[in_range & _v["buy"], ["visitorid", "date", "buy_root_categoryid", "buy_categoryid"]].rename(
            columns={"buy_root_categoryid": "root_categoryid", "buy_categoryid": "categoryid"}
        )
    anchor = anchor.drop_duplicates("visitorid").rename(columns={"date": "day0"})  # 날짜순 정렬이라 첫 행이 Day 0
    activity = _v.loc[_v["buy"], ["visitorid", "date"]] if metric == "repurchase" else _v[["visitorid", "date"]]
    joined = activity.merge(anchor, on="visitorid")
    joined = joined[joined["date"] >= joined["day0"]]
    joined = joined.assign(cohort=period_index(joined["day0"], grain), period=period_index(joined["date"], grain))
    joined = joined.drop_duplicates(["visitorid", "period"])
    joined["n"] = joined["period"] - joined["cohort"]
    s = joined.groupby(["day0", "cohort", "root_categoryid", "categoryid", "n"], as_index=False).size().rename(columns={"size": "returned"})
    s["max_n"] = (RET_LAST_COMPLETE[grain] - s["cohort"]).clip(lower=0)  # 온전한 기간까지 관찰 가능한 최대 N(N=0은 항상 포함)
    return s


def retention_scope(r, metric, grain, start_date, end_date, exclude_early):
    s = retention_frame(r, metric, grain, start_date, end_date)
    return s[s["day0"] >= FIRST_4WEEKS_END] if exclude_early else s


@st.cache_data(show_spinner=False)
def retention_curves(_r, metric, grain, keys, start_date, end_date, exclude_early):
    """keys 단위 N별 관찰 가능 방문자·재방문자·재방문율·Wilson 95% 구간. keys가 빈 튜플이면 전체 한 곡선."""
    keys = list(keys)
    s = retention_scope(_r, metric, grain, start_date, end_date, exclude_early)
    s = s[s["n"] <= s["max_n"]]  # 일부만 관측된 마지막 주·달의 칸은 곡선에서 뺀다
    columns = keys + ["n", "eligible", "returned", "rate", "low", "high"]
    if s.empty:
        return pd.DataFrame(columns=columns)
    s = s.assign(_all=0)
    group = keys or ["_all"]
    ns = pd.RangeIndex(RET_GRAINS[grain]["max_n"] + 1)
    sizes = s[s["n"] == 0].pivot_table(index=group, columns="max_n", values="returned", aggfunc="sum", fill_value=0).reindex(columns=ns, fill_value=0)
    eligible = pd.DataFrame(sizes.to_numpy()[:, ::-1].cumsum(axis=1)[:, ::-1], index=sizes.index, columns=ns)  # max_n >= N인 코호트 합
    returned = s.pivot_table(index=group, columns="n", values="returned", aggfunc="sum", fill_value=0).reindex(index=sizes.index, columns=ns, fill_value=0)
    long = pd.DataFrame({"eligible": eligible.stack(), "returned": returned.stack()}).rename_axis(group + ["n"]).reset_index()
    long = long[long["eligible"] > 0].copy()
    p = long["returned"] / long["eligible"]
    denom = 1 + Z95**2 / long["eligible"]
    center = (p + Z95**2 / (2 * long["eligible"])) / denom
    half = Z95 * np.sqrt(p * (1 - p) / long["eligible"] + Z95**2 / (4 * long["eligible"] ** 2)) / denom
    long["rate"], long["low"], long["high"] = p * 100, (center - half) * 100, (center + half) * 100
    return long[columns]


def curve_top(curves, n_range):
    """그래프 Y축 상한. 표시 범위의 N에서 표본이 충분한 점만 본다."""
    tops = [c.loc[c["n"].between(*n_range) & (c["eligible"] >= RET_MIN_POINT), "high" if band else "rate"].max() for c, band in curves if not c.empty]
    tops = [t for t in tops if pd.notna(t)]
    return (max(tops) if tops else 1) * 1.1


def retention_chart(curve, title, n_range, y_top, grain, reference=None, band=False, height=280):
    """리텐션 곡선. 표본이 RET_MIN_POINT 미만인 N은 그리지 않는다. 주·월은 점이 적어 표식을 함께 그린다."""
    spec = RET_GRAINS[grain]
    mode = "lines" if grain == "day" else "lines+markers"
    fig = go.Figure()
    c = curve[curve["n"].between(*n_range)].sort_values("n")
    c = c[c["eligible"] >= RET_MIN_POINT]
    if band and not c.empty:
        fig.add_trace(go.Scatter(
            x=list(c["n"]) + list(c["n"])[::-1], y=list(c["high"]) + list(c["low"])[::-1], fill="toself", fillcolor="rgba(42,111,187,0.14)",
            line=dict(width=0), hoverinfo="skip", name="95% 신뢰구간",
        ))
    if reference is not None:
        ref = reference[reference["n"].between(*n_range) & (reference["eligible"] >= RET_MIN_POINT)].sort_values("n")
        fig.add_trace(go.Scatter(x=ref["n"], y=ref["rate"], mode=mode, line=dict(color="#9aa5b1", width=1.5, dash="dash"), marker=dict(size=5), name="전체",
                                 hovertemplate="전체 %{y:.2f}%<extra></extra>"))
    if not c.empty:
        fig.add_trace(go.Scatter(
            x=c["n"], y=c["rate"], mode=mode, line=dict(color=COLOR, width=2.2), marker=dict(size=7), name="값",
            customdata=np.column_stack([c["returned"], c["eligible"]]),
            hovertemplate="%{x}" + spec["word"] + "째 <b>%{y:.2f}%</b><br>재방문 %{customdata[0]:,}명 / 관찰 가능 %{customdata[1]:,}명<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0), height=height, margin=dict(l=10, r=15, t=48, b=10), showlegend=False,
        hovermode="x unified" if reference is not None else "closest",
        xaxis=dict(title=dict(text=spec["axis"], font=dict(size=11)), range=[n_range[0] - 0.5, n_range[1] + 0.5], showgrid=False, zeroline=False, dtick=None if grain == "day" else 1),
        yaxis=dict(title=dict(text="%", font=dict(size=11)), range=[0, y_top], tickformat=",.2f" if y_top < 2 else ",.1f", ticksuffix="%", rangemode="tozero"),
    )
    return fig


def retention_summary(r, grain, keys, start_date, end_date, exclude_early, metric, min_n):
    """keys 단위로 세 지표의 주요 N을 한 표로 만든다. 선택 지표의 기준 N 내림차순, 그 분모가 min_n 미만이면 제외."""
    spec = RET_GRAINS[grain]
    key_n = spec["key_n"]
    frames = []
    for key in RET_METRICS:
        c = retention_curves(r, key, grain, tuple(keys), start_date, end_date, exclude_early)
        if c.empty:
            continue
        indexed = c.set_index(keys)
        frames.append(indexed.loc[indexed["n"] == 0, "eligible"].rename(f"{key}_size"))
        for n in spec["show"]:
            at = indexed[indexed["n"] == n]
            frames.append(at["rate"].where(at["eligible"] >= RET_MIN_POINT).rename(f"{key}_r{n}"))
        at_key = indexed[indexed["n"] == key_n]
        frames.append(at_key["eligible"].rename(f"{key}_obs"))
        frames.append(at_key["returned"].rename(f"{key}_ret"))
    if not frames:
        return pd.DataFrame(), 0
    wide = pd.concat(frames, axis=1).reset_index()
    for col in (f"{metric}_obs", f"{metric}_ret", f"{metric}_r{key_n}"):
        if col not in wide:
            wide[col] = np.nan
    enough = wide[f"{metric}_obs"].fillna(0) >= min_n
    excluded = int((~enough).sum())
    wide = wide[enough].sort_values([f"{metric}_r{key_n}", f"{metric}_obs"], ascending=False, na_position="last", kind="stable").reset_index(drop=True)
    return wide, excluded


def retention_table(wide, first_col, metric, grain, limit=None):
    if wide.empty:
        st.info("조건에 맞는 항목이 없습니다. 최소 표본 수를 낮춰 보세요.")
        return
    spec = RET_GRAINS[grain]
    key_label = f"{spec['prefix']}{spec['key_n']}"
    shown = wide.head(limit) if limit else wide
    out = pd.DataFrame({"순위": range(1, len(shown) + 1)})
    for col, label in first_col:
        out[label] = shown[col].tolist()
    count_cols, rate_cols = [], []
    for key, info in RET_METRICS.items():
        if key != "repurchase":  # 재구매율과 구매자의 재방문율은 대상이 같아 인원 열을 한 번만 둔다
            label = info["target"] + "(명)"
            out[label] = shown[f"{key}_size"].tolist() if f"{key}_size" in shown else None
            count_cols.append(label)
        for n in spec["show"]:
            label = f"{info['name']} {spec['prefix']}{n}(%)"
            out[label] = shown[f"{key}_r{n}"].round(2).tolist() if f"{key}_r{n}" in shown else None
            rate_cols.append(label)
    name = RET_METRICS[metric]["name"]
    ret_label, obs_label = f"{name} {key_label} 해당자(명)", f"{name} {key_label} 관찰 가능(명)"
    out[ret_label], out[obs_label] = shown[f"{metric}_ret"].tolist(), shown[f"{metric}_obs"].tolist()
    count_cols += [ret_label, obs_label]
    heatmap_table(
        out, blue_cols=count_cols + rate_cols, sort_col=f"{name} {key_label}(%)", height=min(38 + 35 * len(out), 460),
        formats={**{c: "{:,.0f}" for c in count_cols}, **{c: "{:.2f}" for c in rate_cols}},
    )
    if limit and len(wide) > limit:
        st.caption(f"{name} {key_label} 상위 {limit:,}개만 표시(전체 {len(wide):,}개)")


def retention_grid(r, grain, keys, labels, metric, n_range, start_date, end_date, exclude_early, overall, tag):
    """labels에 있는 항목마다 선택 지표의 곡선을 그린다. Y축을 맞춰 그래프끼리 비교할 수 있게 한다. 회색 점선은 전체 값."""
    curves = retention_curves(r, metric, grain, tuple(keys), start_date, end_date, exclude_early)
    panels = []
    for key_value, label in labels:
        mask = np.logical_and.reduce([curves[k] == v for k, v in zip(keys, key_value)])
        panels.append((key_value, label, curves[mask]))
    top = curve_top([(c, False) for _, _, c in panels] + [(overall, False)], n_range)
    for start in range(0, len(panels), 3):
        cols = st.columns(3)
        for col, (key_value, label, c) in zip(cols, panels[start:start + 3]):
            size = int(c.loc[c["n"] == 0, "eligible"].sum())
            fig = retention_chart(c, f"<b>{label}</b><br><span style='font-size:11px'>{RET_METRICS[metric]['target']} {size:,}명</span>", n_range, top, grain, reference=overall, height=250)
            col.plotly_chart(fig, width="stretch", key=f"ret-{grain}-{tag}-{metric}-{'-'.join(map(str, key_value))}")


def cohort_label(index, grain):
    """코호트 기간 번호를 행 이름으로 바꾼다. †는 일부만 관측된 기간이다."""
    first, last = period_of(pd.Timestamp("2015-05-03"), grain), period_of(LAST_DATE, grain)
    mark = " †" if index in (first, last) else ""
    if grain == "day":
        return f"{pd.Timestamp('1970-01-01') + pd.Timedelta(days=index):%Y-%m-%d}{mark}"
    if grain == "week":
        return f"{pd.Timestamp('1970-01-01') + pd.Timedelta(days=index * 7 - 3):%Y-%m-%d} 주{mark}"
    return f"{index // 12}-{index % 12 + 1:02d}{mark}"


@st.cache_data(show_spinner=False)
def cohort_matrix(_r, metric, grain, root, start_date, end_date, exclude_early, last_n):
    """코호트(행) × N(열)의 재방문자 수와 재방문율. 관찰할 수 없는 칸은 비워 둔다."""
    s = retention_scope(_r, metric, grain, start_date, end_date, exclude_early)
    if root is not None:
        s = s[s["root_categoryid"] == root]
    if s.empty:
        return None
    last_observed = period_of(LAST_DATE, grain)
    columns = pd.RangeIndex(last_n + 1)
    index = sorted(s["cohort"].unique())
    counts = s[s["n"] <= last_n].pivot_table(index="cohort", columns="n", values="returned", aggfunc="sum", fill_value=0).reindex(index=index, columns=columns, fill_value=0)
    sizes = s[s["n"] == 0].groupby("cohort")["returned"].sum().reindex(index, fill_value=0)
    observable = pd.DataFrame({n: [c + n <= last_observed for c in index] for n in columns}, index=index)
    incomplete = pd.DataFrame({n: [c + n > RET_LAST_COMPLETE[grain] for c in index] for n in columns}, index=index)
    return counts, sizes, observable, incomplete


def render_cohort_heatmap(r, grain, metric, n_range, start_date, end_date, exclude_early):
    """코호트 히트맵. 칸에 재방문자 수와 재방문율을 쓰고, 재방문율이 클수록 진하게 칠한다."""
    spec = RET_GRAINS[grain]
    info = RET_METRICS[metric]
    st.subheader(f"코호트 히트맵 · {info['name']} ({spec['label']})")
    roots = retention_curves(r, metric, grain, ("root_categoryid",), start_date, end_date, exclude_early)
    roots = roots[roots["n"] == 0].sort_values("eligible", ascending=False)["root_categoryid"].tolist()
    root = st.selectbox("히트맵 범위", [None] + roots, key=f"ret_cohort_root_{metric}", format_func=lambda v: "전체" if v is None else root_label(v))
    last_n = max(n_range[1], 1)
    matrix = cohort_matrix(r, metric, grain, root, start_date, end_date, exclude_early, last_n)
    if matrix is None:
        st.info("선택한 조건에 해당하는 방문자가 없습니다.")
        return
    counts, sizes, observable, incomplete = matrix
    rates = counts.div(sizes.where(sizes > 0), axis=0) * 100
    pool = rates.loc[:, 1:].where(observable.loc[:, 1:] & ~incomplete.loc[:, 1:]).to_numpy().ravel()
    pool = pool[~np.isnan(pool)]
    top = float(np.quantile(pool, COHORT_TOP_QUANTILE)) if pool.size else 0.0
    columns = [f"{spec['prefix']}{n}" for n in counts.columns]
    rows = [cohort_label(i, grain) for i in counts.index]
    text, styles = [], []
    for i, cohort in enumerate(counts.index):
        text_row, style_row = [], []
        for n in counts.columns:
            if not observable.loc[cohort, n]:
                text_row.append("")
                style_row.append("")
                continue
            rate = rates.loc[cohort, n]
            mark = "*" if n > 0 and incomplete.loc[cohort, n] else ""
            text_row.append(f"{int(counts.loc[cohort, n]):,} ({'100%' if n == 0 else f'{rate:.2f}%'}){mark}")
            norm = 1.0 if n == 0 else (min(rate / top, 1.0) ** 0.5 if top > 0 else 0.0)  # 작은 값의 차이가 보이도록 제곱근 눈금
            style_row.append(heat_cell(norm, HEAT_BLUE))
        text.append(text_row)
        styles.append(style_row)
    shown = pd.DataFrame(text, index=rows, columns=columns)
    shown.index.name = "코호트"
    style_frame = pd.DataFrame(styles, index=rows, columns=columns)
    st.dataframe(shown.style.apply(lambda _: style_frame, axis=None), width="stretch", height=min(38 + 35 * len(rows), 620))
    first_word = {"day": "날", "week": "주", "month": "달"}[grain]
    st.caption(
        f"행은 {info['day0']}이 속한 {first_word}, 열은 그로부터 N번째 {spec['word'] if grain != 'month' else '달'}입니다. 칸은 `재방문자 수 (재방문율)`이고 재방문율은 코호트 인원 대비입니다. "
        f"{spec['prefix']}0은 코호트 인원입니다. 색은 재방문율이며 작은 값의 차이가 보이도록 제곱근 눈금을 쓰고, {top:.2f}%(상위 1%) 이상은 같은 진하기입니다. "
        f"빈칸은 데이터 기간을 넘어 관찰할 수 없는 칸으로 0명과 다릅니다. † 일부만 관측된 코호트 기간, * 일부만 관측된 마지막 {first_word}의 칸이라 값이 작게 나옵니다. "
        f"열 범위는 위의 그래프 N 범위 상한을 따릅니다. {start_date:%Y-%m-%d} ~ {end_date:%Y-%m-%d}에 활동한 방문자만 집계하므로, 선택 기간에 일부만 걸친 주·달의 코호트 인원은 그 기간의 방문자만 포함합니다."
    )


def render_retention_tab(r):
    if r is None:
        st.error("리텐션 데이터(data/processed/dashboard/retention.parquet)가 없습니다. 먼저 `python analysis/19_export_dashboard_data.py`를 실행하세요.")
        return
    first_date, last_date = pd.Timestamp("2015-05-03").date(), LAST_DATE.date()
    picked = st.date_input(
        "방문일 선택", value=(first_date, last_date), min_value=first_date, max_value=last_date, format="YYYY-MM-DD", key="ret_dates",
        help="이 탭의 표·그래프·코호트 히트맵은 이 기간에 방문한 모든 방문자(재구매율·구매자의 재방문율은 이 기간에 구매한 구매자)를 대상으로 합니다. "
        "이전에 방문한 적이 있어도 포함합니다. Day 0은 이 기간의 첫 활동일이고, 그 이후의 재방문은 기간 밖이어도 셉니다. 하루만 보려면 시작일과 종료일을 같은 날짜로 선택하세요.",
    )
    if not isinstance(picked, (tuple, list)) or len(picked) != 2:
        st.info("종료일까지 선택해 주세요. 하루만 보려면 같은 날짜를 시작일과 종료일로 선택하면 됩니다.")
        return
    start_date, end_date = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
    grain_label = st.radio(
        "리텐션 기준", [v["label"] for v in RET_GRAINS.values()], horizontal=True, key="ret_grain",
        help="이 탭의 모든 표와 그래프가 이 기준으로 바뀝니다. 일별은 정확히 N일째, 주별은 달력 주(월요일 시작), 월별은 달력 월로 N을 셉니다.",
    )
    grain = RET_LABEL_TO_GRAIN[grain_label]
    spec = RET_GRAINS[grain]
    prefix, word, key_n, max_n = spec["prefix"], spec["word"], spec["key_n"], spec["max_n"]
    key_label = f"{prefix}{key_n}"
    exclude_early = st.checkbox(
        "Day 0이 시작 4주(2015-05-03~05-30)인 방문자 제외", value=False, key="ret_early",
        help="데이터 시작 전의 행동이 관측되지 않아, 이 기간의 첫 이벤트·첫 구매는 실제 첫 방문·첫 구매가 아닐 수 있습니다.",
    )
    ctl1, ctl2, ctl3 = st.columns([3, 2, 2])
    metric_name = ctl1.radio(
        "코호트·대분류·카테고리 비교 지표", [v["name"] for v in RET_METRICS.values()], horizontal=True, key="ret_metric",
        help=f"코호트 히트맵, 대분류·카테고리별 그래프와 표의 정렬이 이 지표 기준으로 바뀝니다. 세 지표의 {', '.join(f'{prefix}{n}' for n in spec['show'])}은 표에 함께 표시됩니다.",
    )
    metric = RET_NAME_TO_KEY[metric_name]
    min_n = int(ctl2.number_input(
        f"표의 최소 표본 수 ({key_label} 관찰 가능 방문자)", min_value=1, max_value=100000, value=DEFAULT_MIN_VISITORS, step=50, key=f"ret_min_n_{grain}",
        help=f"선택 지표의 {key_label} 관찰 가능 방문자가 이 값 미만인 항목을 표와 그래프 목록에서 제외합니다.",
    ))
    n_range = ctl3.slider("그래프 N 범위", 0, max_n, spec["default"], key=f"ret_n_range_{grain}", help=f"{prefix}0은 정의상 100%라 기본 범위에서 뺐습니다.")
    basis = {
        "day": "N일째는 Day 0으로부터 **정확히 N일째** UTC 날짜",
        "week": "N주째는 Day 0이 속한 **달력 주(월요일 시작)로부터 N번째 주**, 그 주에 한 번이라도 활동하면 1명",
        "month": "N개월째는 Day 0이 속한 **달력 월로부터 N번째 달**, 그 달에 한 번이라도 활동하면 1명",
    }[grain]
    denominator = "분모는 N일째를 관찰할 수 있는 방문자" if grain == "day" else f"분모는 그 {word if grain == 'week' else '달'}을 온전히 관찰할 수 있는 방문자"
    st.caption(
        f"{start_date:%Y-%m-%d} ~ {end_date:%Y-%m-%d}에 방문한 모든 방문자 기준(구매 지표는 이 기간의 구매자) · Day 0은 이 기간의 첫 활동일 · {basis} · {denominator} · 재방문은 상품·카테고리와 무관하게 셉니다"
    )

    overall = {k: retention_curves(r, k, grain, (), start_date, end_date, exclude_early) for k in RET_METRICS}
    if overall["revisit"].empty:
        st.info("선택한 조건에 해당하는 방문자가 없습니다.")
        return

    def at(curve, n, col):
        row = curve[curve["n"] == n]
        return row[col].iloc[0] if len(row) else None

    # 주요 지표
    st.subheader(f"주요 지표 · {key_n}{word}째")
    helps = {
        "revisit": f"선택 기간의 첫 이벤트(종류 무관)일을 Day 0으로, {key_n}{word}째에 이벤트(종류 무관)가 있는 방문자 비율",
        "repurchase": f"선택 기간의 첫 구매일을 Day 0으로, {key_n}{word}째에 구매가 있는 구매자 비율",
        "buyer_revisit": f"선택 기간의 첫 구매일을 Day 0으로, {key_n}{word}째에 이벤트(종류 무관)가 있는 구매자 비율",
    }
    for col, (key, info) in zip(st.columns(3), RET_METRICS.items()):
        rate, returned, eligible = at(overall[key], key_n, "rate"), at(overall[key], key_n, "returned"), at(overall[key], key_n, "eligible")
        value = "-" if rate is None else f"{rate:.2f}%"
        note = "" if rate is None else f" 관찰 가능 {int(eligible):,}명 중 {int(returned):,}명"
        col.metric(info["name"], value, help=helps[key] + "." + note, border=True)

    rows = []
    rate_cols = [f"{prefix}{n}(%)" for n in spec["table_n"]]
    for key, info in RET_METRICS.items():
        c = overall[key]
        row = {"지표": info["name"], "Day 0": info["day0"], "재방문 기준": info["back"], "대상(명)": at(c, 0, "eligible")}
        for n, label in zip(spec["table_n"], rate_cols):
            row[label] = at(c, n, "rate")
        row[f"{key_label} 해당자(명)"] = at(c, key_n, "returned")
        rows.append(row)
    heatmap_table(pd.DataFrame(rows), blue_cols=[], height=38 + 35 * len(rows), formats={"대상(명)": "{:,.0f}", f"{key_label} 해당자(명)": "{:,.0f}", **{c: "{:.2f}" for c in rate_cols}})

    st.subheader(f"전체 리텐션 곡선 ({spec['label']})")
    for col, (key, info) in zip(st.columns(3), RET_METRICS.items()):
        c = overall[key]
        col.plotly_chart(
            retention_chart(c, f"<b>{info['name']}</b><br><span style='font-size:11px'>Day 0: {info['day0']} · 재방문: {info['back']}</span>", n_range, curve_top([(c, True)], n_range), grain, band=True),
            width="stretch", key=f"ret-overall-{grain}-{key}",
        )
    st.caption("음영은 95% 신뢰구간입니다. 지표마다 값의 크기가 달라 Y축 범위가 서로 다릅니다.")

    # 코호트 히트맵
    render_cohort_heatmap(r, grain, metric, n_range, start_date, end_date, exclude_early)

    selected_key = at(overall[metric], key_n, "returned")
    if selected_key is not None and selected_key < 100:
        st.warning(
            f"{metric_name}의 {key_label} 해당자는 전체에서 {int(selected_key):,}명뿐입니다. 대분류·카테고리로 나누면 대부분 0~몇 명이라 {key_label} 순위는 우연의 영향이 큽니다. "
            "앞쪽 구간이나 곡선 전체의 모양을 함께 보세요."
        )

    # 대분류별
    st.subheader(f"대분류별 리텐션 · {metric_name}")
    st.caption(
        f"방문자를 Day 0 행동({RET_METRICS[metric]['day0']})의 상품이 속한 대분류에 배정합니다. 표는 {metric_name} {key_label} 내림차순이고, "
        f"그래프는 {RET_METRICS[metric]['target']} 수가 많은 순입니다. 회색 점선은 전체 값입니다. 관찰 가능 방문자가 {RET_MIN_POINT}명 미만인 값은 그래프에 그리지 않고 표에 None으로 표시합니다."
    )
    root_wide, root_excluded = retention_summary(r, grain, ["root_categoryid"], start_date, end_date, exclude_early, metric, min_n)
    t = root_wide.copy()
    if not t.empty:
        t["label"] = [root_label(v) for v in t["root_categoryid"]]
    retention_table(t, [("label", "대분류")], metric, grain)
    if root_excluded:
        st.caption(f"{key_label} 관찰 가능 방문자 {min_n:,}명 미만 {root_excluded:,}개는 표와 그래프에서 제외")
    if not t.empty:
        order = t.sort_values(f"{metric}_size", ascending=False)["root_categoryid"].tolist()
        order = [v for v in order if v != UNKNOWN] + [v for v in order if v == UNKNOWN]  # 카테고리 미확인은 맨 뒤
        retention_grid(r, grain, ["root_categoryid"], [((v,), root_label(v)) for v in order], metric, n_range, start_date, end_date, exclude_early, overall[metric], "root")

    # 카테고리별
    st.subheader(f"카테고리별 리텐션 · {metric_name}")
    roots = retention_curves(r, "revisit", grain, ("root_categoryid",), start_date, end_date, exclude_early)
    roots = roots[(roots["n"] == 0) & (roots["root_categoryid"] != UNKNOWN)].sort_values("eligible", ascending=False)["root_categoryid"].tolist()
    scope_root = st.selectbox("카테고리를 볼 대분류 범위", [None] + roots, index=1 if roots else 0, key="ret_root", format_func=lambda v: "전체 대분류" if v is None else root_label(v))
    keys = ["root_categoryid", "categoryid"]
    cat_wide, cat_excluded = retention_summary(r, grain, keys, start_date, end_date, exclude_early, metric, min_n)
    cat_wide = cat_wide[cat_wide["categoryid"] != UNKNOWN] if not cat_wide.empty else cat_wide
    if scope_root is not None and not cat_wide.empty:
        cat_wide = cat_wide[cat_wide["root_categoryid"] == scope_root]
    t = cat_wide.copy()
    if not t.empty:
        t["root"] = [root_label(v) for v in t["root_categoryid"]]
        t["cat"] = [f"카테고리 {v}" for v in t["categoryid"]]
    retention_table(t, [("root", "대분류"), ("cat", "카테고리")], metric, grain, limit=TABLE_ROWS)
    if not t.empty:
        chart_count = int(st.number_input(
            f"그래프로 볼 카테고리 수 ({RET_METRICS[metric]['target']} 많은 순, 표 조건을 충족한 {len(t):,}개 중)",
            min_value=1, max_value=min(len(t), 60), value=min(len(t), 12), step=3, key=f"ret_chart_count_{grain}_{metric}_{scope_root}",
        ))
        top = t.sort_values(f"{metric}_size", ascending=False).head(chart_count)
        retention_grid(r, grain, keys, [((a, b), f"{root_label(a)} › 카테고리 {b}") for a, b in zip(top["root_categoryid"], top["categoryid"])],
                       metric, n_range, start_date, end_date, exclude_early, overall[metric], "cat")

    # 선택 카테고리 상세
    st.markdown("##### 카테고리 하나의 세 지표 보기")
    sizes = retention_curves(r, "revisit", grain, tuple(keys), start_date, end_date, exclude_early)
    sizes = sizes[(sizes["n"] == 0) & (sizes["categoryid"] != UNKNOWN)]
    if scope_root is not None:
        sizes = sizes[sizes["root_categoryid"] == scope_root]
    sizes = sizes.sort_values("eligible", ascending=False)
    if sizes.empty:
        st.info("이 범위에는 카테고리가 확인된 방문자가 없습니다.")
    else:
        size_of = {(a, b): int(e) for a, b, e in zip(sizes["root_categoryid"], sizes["categoryid"], sizes["eligible"])}
        choice = st.selectbox(
            "카테고리 선택", list(size_of), key=f"ret_cat_{scope_root}",
            format_func=lambda k: f"{cat_label(*k)} (방문자 {size_of[k]:,}명)",
        )
        for col, (key, info) in zip(st.columns(3), RET_METRICS.items()):
            curves = retention_curves(r, key, grain, tuple(keys), start_date, end_date, exclude_early)
            c = curves[(curves["root_categoryid"] == choice[0]) & (curves["categoryid"] == choice[1])]
            size = int(c.loc[c["n"] == 0, "eligible"].sum())
            title = f"<b>{info['name']}</b><br><span style='font-size:11px'>{info['target']} {size:,}명</span>"
            if size < RET_MIN_POINT:
                col.info(f"{info['name']}: {info['target']} {size:,}명으로, 표본 {RET_MIN_POINT}명 미만이라 곡선을 표시하지 않습니다.")
            else:
                col.plotly_chart(
                    retention_chart(c, title, n_range, curve_top([(c, True), (overall[key], False)], n_range), grain, reference=overall[key], band=True),
                    width="stretch", key=f"ret-detail-{grain}-{key}-{choice[0]}-{choice[1]}",
                )
        st.caption("음영은 이 카테고리의 95% 신뢰구간, 회색 점선은 전체 값입니다.")

    returned_note = ", ".join(
        f"{RET_METRICS[k]['name']} {int(at(overall[k], key_n, 'returned')):,}명" for k in ("repurchase", "buyer_revisit") if at(overall[k], key_n, "returned") is not None
    )
    with st.expander("지표 정의와 해석 시 유의사항"):
        st.markdown(
            f"""
- **대상과 Day 0:** 선택 기간에 이벤트(view·장바구니·구매 중 무엇이든)가 있는 모든 방문자이며, Day 0은 선택 기간의 첫 이벤트일입니다. 재구매율과 구매자의 재방문율은 선택 기간에 구매한 구매자가 대상이고 Day 0은 선택 기간의 첫 구매일입니다. 선택 기간 이전의 활동은 보지 않으며, 재방문은 Day 0 이후라면 선택 기간 밖이어도 셉니다.
- **재방문율:** Day 0으로부터 N번째 기간에 이벤트가 있는 방문자의 비율입니다.
- **재구매율:** Day 0으로부터 N번째 기간에 구매가 있는 구매자의 비율입니다. Day 0 당일의 추가 구매는 세지 않으며, 주별·월별에서는 N=0 기간에 포함됩니다.
- **구매자의 재방문율:** Day 0으로부터 N번째 기간에 이벤트(종류 무관)가 있는 구매자의 비율입니다.
- **신규와 기존 방문자:** 데이터 전체 기간을 선택하면 Day 0이 최초 관측일이 되어 첫 방문 코호트와 같습니다. 일부 기간을 선택하면 신규와 기존 방문자가 섞이므로 전체 기간보다 값이 높게 나오는 경향이 있으며, 신규 방문자의 리텐션으로 읽으면 안 됩니다.
- **일별:** Day 0으로부터 정확히 N일째의 활동만 셉니다. 앞뒤 날짜의 활동은 세지 않으므로 'N일 이내 재방문'이 아닙니다. 마지막 날(2015-09-15)은 21시 이전까지만 관측되지만 분석 기준에 따라 포함합니다.
- **주별·월별:** Day 0이 속한 달력 주(월요일 시작)·달력 월을 N=0으로 두고, N번째 주·달에 한 번이라도 활동하면 1명으로 셉니다. UTC 기준입니다. Day 0 이전의 활동은 세지 않습니다.
- **달력 기간의 차이:** 같은 1주째라도 일요일 방문자는 다음 날부터, 월요일 방문자는 7일 뒤부터 1주째가 됩니다. 그래서 주별 W1을 일별 D7과 직접 비교할 수 없습니다.
- **일부만 관측된 기간:** 마지막 주(09-14~09-15)와 마지막 달(9월)은 며칠만 관측되어 값이 작게 나옵니다. 곡선·표·주요 지표의 N≥1 값에서는 이 기간을 분자와 분모에서 모두 빼고, 코호트 히트맵에는 * 표시로 보여 줍니다. 이 기간의 코호트는 N=0 인원에만 들어갑니다. 첫 주 코호트(04-27 주)는 05-03 하루뿐입니다.
- **방문일 선택:** 이 탭 위쪽 캘린더는 대시보드 맨 위의 공통 기간 선택과 별개입니다.
- **코호트 히트맵의 분모:** 각 칸의 재방문율은 그 코호트 인원 대비입니다. 위 체크박스·방문일 선택을 적용한 뒤의 인원입니다.
- **카테고리 배정:** 방문자를 Day 0 행동의 상품 카테고리 1개에 배정합니다. 값은 그 시각에 적용된 카테고리이고, 같은 시각이면 파일 순서의 첫 행을 씁니다. 재방문율과 구매 기반 지표는 Day 0 행동이 달라 같은 방문자도 다른 카테고리에 들어갈 수 있습니다.
- **재방문과 카테고리:** 재방문·재구매의 대상 상품은 따지지 않습니다. 같은 카테고리를 다시 샀다는 뜻이 아닙니다.
- **첫 구매에 여러 카테고리가 있는 경우:** 구매자의 12.8%는 첫 구매일에 2개 이상 카테고리를 샀지만, 첫 구매 행의 카테고리에만 집계합니다.
- **카테고리 미확인:** 이력 파일에 없는 상품이거나 상품의 첫 이력(가장 이른 2015-05-10) 이전 이벤트입니다. 전체 값에는 포함하고 '카테고리 미확인'으로 따로 표시합니다.
- **표본:** 현재 조건에서 {key_label} 해당자는 {returned_note or '없음'}입니다. 해당자가 적으면 카테고리별 차이를 단정하지 말고 신뢰구간과 해당자 수를 함께 보세요. 관찰 가능 방문자가 {RET_MIN_POINT}명 미만인 값은 표시하지 않습니다.
- **뒤쪽 N:** N이 커질수록 관찰 가능한 코호트가 줄고, 끝으로 갈수록 시작 4주 코호트만 남습니다. 뒤쪽 곡선의 상승을 회복으로 해석하지 마세요. 월별은 온전히 관측된 달이 5~8월뿐이라 M3까지만, 그것도 5월 코호트만 볼 수 있습니다.
- **시작 4주:** 전체 기간을 선택해도 이전 행동이 관측되지 않아 첫 방문·첫 구매로 단정할 수 없고, 재방문율이 이후 코호트보다 높게 나타납니다. 위 체크박스로 제외해 비교할 수 있습니다.
- **인과 해석:** 카테고리 간 차이는 관찰된 차이이며, 그 카테고리가 재방문을 만들었다는 근거는 아닙니다. `visitorid`가 실제 사람과 1:1인지는 확인되지 않았습니다.
            """
        )


# ── 공통 상단 ─────────────────────────────────────────────
if not EVENTS.exists():
    st.error("대시보드 데이터(data/processed/dashboard/)가 없습니다. 먼저 `python analysis/19_export_dashboard_data.py`를 실행하세요.")
    st.stop()
data_start = load_data_start()
min_date, max_date = data_start.normalize().date(), (DATA_END - pd.Timedelta(seconds=1)).normalize().date()

st.title("📊 퍼널 주요 지표 대시보드")
st.caption(
    f"출처: Retailrocket 추천 시스템 데이터셋(CC BY-NC-SA 4.0, 비영리·교육 목적) · 모든 날짜·시각은 UTC 기준 · 분석 가능 기간 {min_date} ~ {max_date} "
    f"(완전 중복 제거, {DATA_END:%Y-%m-%d %H:%M} UTC 미만)"
)

picked = st.date_input(
    "기간 선택",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    format="YYYY-MM-DD",
    help="시작일과 종료일을 차례로 선택하세요. 종료일은 포함됩니다. 주요 지표·추이, 카테고리, 퍼널 탭에 적용되며, 두 번째·세 번째 탭에서는 방문자의 첫 조회일에 적용됩니다. 리텐션 탭은 탭 안의 방문일 선택을 씁니다.",
)
if not isinstance(picked, (tuple, list)) or len(picked) != 2:
    st.info("종료일까지 선택해 주세요. 하루만 보려면 같은 날짜를 시작일과 종료일로 선택하면 됩니다.")
    st.stop()
start_date, end_date = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])

# st.tabs는 기본적으로 네 탭을 매번 모두 계산한다. 선택된 탭만 계산해 메모리와 실행 시간을 줄인다(on_change="rerun").
# 계산하지 않은 탭의 위젯은 Streamlit이 상태를 지우므로, 닫힌 탭의 위젯 값은 session_state에 다시 써서 돌아왔을 때 유지한다.
TAB_NAMES = ["주요 지표 · 추이", "카테고리별 첫 조회 성과", "퍼널", "리텐션"]
TAB_RENDERERS = {
    TAB_NAMES[0]: lambda: render_trend_tab(load_events()[0], data_start, start_date, end_date, load_first_view()),
    TAB_NAMES[1]: lambda: render_first_view_tab(load_first_view(), start_date, end_date),
    TAB_NAMES[2]: lambda: render_funnel_tab(load_funnel(), start_date, end_date),
    TAB_NAMES[3]: lambda: render_retention_tab(load_retention()),
}
tab_widget_keys = st.session_state.setdefault("_tab_widget_keys", {})  # 탭 이름 → 그 탭의 위젯 키
tabs = st.tabs(TAB_NAMES, key="main_tab", on_change="rerun")
open_index = next((i for i, tab in enumerate(tabs) if tab.open), 0)
open_name = TAB_NAMES[open_index]
for name in TAB_NAMES:
    if name != open_name:
        for key in tab_widget_keys.get(name, ()):
            if key in st.session_state:
                st.session_state[key] = st.session_state[key]
keys_before = set(st.session_state.keys())
with tabs[open_index]:
    TAB_RENDERERS[open_name]()
tab_widget_keys.setdefault(open_name, set()).update(set(st.session_state.keys()) - keys_before)
