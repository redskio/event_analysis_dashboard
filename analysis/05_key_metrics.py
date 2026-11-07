"""이벤트 테이블과 방문자 테이블에서 주요 지표를 추출한다.

실행: 프로젝트 루트에서 python analysis/05_key_metrics.py
산출: outputs/tables/key_metrics.csv
"""
from pathlib import Path

import pandas as pd

EVENTS = Path("data/processed/events_clean.csv")
VISITORS = Path("data/processed/visitor_summary.csv")
OUT = Path("outputs/tables/key_metrics.csv")

ev = pd.read_csv(EVENTS, usecols=["visitorid", "event", "transactionid"], dtype={"transactionid": "Int64"})
vs = pd.read_csv(VISITORS)

# 이벤트 테이블 기준
from_events = {
    "active_users": ev["visitorid"].nunique(),
    "buyers": ev.loc[ev["event"] == "transaction", "visitorid"].nunique(),
    "purchase_count": ev["transactionid"].nunique(),
    "addtocart_count": int((ev["event"] == "addtocart").sum()),
    "event_count": len(ev),
}
# 방문자 테이블 기준
from_visitors = {
    "active_users": int(((vs["view_count"] > 0) | (vs["addtocart_count"] > 0) | (vs["purchase_count"] > 0)).sum()),
    "buyers": int((vs["purchase_count"] > 0).sum()),
    "purchase_count": int(vs["purchase_count"].sum()),
    "addtocart_count": int(vs["addtocart_count"].sum()),
    "event_count": int(vs[["view_count", "addtocart_count"]].sum().sum() + (ev["event"] == "transaction").sum()),
}
labels = {
    "active_users": "활성 사용자 수 (view·addtocart·transaction 중 1회 이상, 고유 방문자)",
    "buyers": "구매자 수 (transaction 1회 이상, 고유 방문자)",
    "purchase_count": "구매 횟수 (고유 transactionid)",
    "addtocart_count": "장바구니에 담은 횟수 (addtocart 행)",
    "event_count": "이벤트 횟수 (전체 행)",
}
table = pd.DataFrame(
    {"metric": [labels[k] for k in labels], "event_table": list(from_events.values()), "visitor_table": list(from_visitors.values())}
)
table["match"] = table["event_table"] == table["visitor_table"]
OUT.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT, index=False, encoding="utf-8-sig")
print(table.to_string())

# 참고: 보조 수치
tx_rows = int((ev["event"] == "transaction").sum())
view_rows = int((ev["event"] == "view").sum())
all_users = ev["visitorid"].nunique()
print("\n참고")
print("view 기록이 없는 활성 사용자:", all_users - ev.loc[ev.event == "view", "visitorid"].nunique(), "(view 1회 이상 기준 활성 사용자:", ev.loc[ev.event == "view", "visitorid"].nunique(), ")")
print("transaction 행:", tx_rows, "| view 행:", view_rows, "| 구매 방문자 중 view 없는 방문자:",
      len(set(ev.loc[ev.event == 'transaction', 'visitorid']) - set(ev.loc[ev.event == 'view', 'visitorid'])))
print("이벤트 구성 합:", view_rows + from_events["addtocart_count"] + tx_rows)
print("구매자/활성 사용자(%):", round(from_events["buyers"] / from_events["active_users"] * 100, 2))
print("구매자 중 view 있는 사용자:", len(set(ev.loc[ev.event == 'transaction', 'visitorid']) & set(ev.loc[ev.event == 'view', 'visitorid'])))
