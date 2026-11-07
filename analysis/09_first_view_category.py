"""방문자의 첫 상품 조회(view)가 속한 카테고리를 집계하고 시각화한다.

실행: 프로젝트 루트에서 python analysis/09_first_view_category.py
입력: data/processed/events_with_category.csv
산출: outputs/tables/first_view_by_category.csv, outputs/figures/first_view_by_category.png

집계 기준: 방문자(visitorid)마다 관측 기간에서 가장 이른 view 이벤트 1건을 '첫 상품 조회'로 보고,
그 시점에 적용된 categoryid / root_categoryid 기준으로 방문자 수를 센다.
카테고리를 알 수 없는 행(category_match_status != matched)은 '카테고리 미확인'으로 따로 센다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SRC = Path("data/processed/events_with_category.csv")
TABLE = Path("outputs/tables/first_view_by_category.csv")
FIGURE = Path("outputs/figures/first_view_by_category.png")
TOP_LEAF = 15

df = pd.read_csv(
    SRC,
    usecols=["timestamp", "visitorid", "event", "itemid", "categoryid", "root_categoryid", "category_match_status"],
    dtype={"categoryid": "Int64", "root_categoryid": "Int64"},
)
assert df["timestamp"].is_monotonic_increasing  # 같은 시각이면 파일 순서의 첫 행을 사용

views = df[df["event"] == "view"]
first = views.drop_duplicates("visitorid", keep="first").copy()
n_first = len(first)

# 같은 방문자의 첫 view가 같은 밀리초에 여러 건이라 선택이 임의가 된 경우를 점검
min_ts = views.groupby("visitorid")["timestamp"].transform("min")
tied = views[views["timestamp"] == min_ts].groupby("visitorid").size()
tied_diff = views[views["timestamp"] == min_ts].groupby("visitorid")["categoryid"].nunique(dropna=False)
print(f"첫 view 방문자: {n_first:,} | 같은 시각에 view가 여러 건: {(tied > 1).sum():,}명 (그중 카테고리가 서로 다름 {(tied_diff[tied[tied > 1].index] > 1).sum()}명)")
assert n_first == views["visitorid"].nunique()

known = first["category_match_status"].eq("matched")
print("첫 view 카테고리 매칭:", first["category_match_status"].value_counts().to_dict(), f"| 매칭률 {known.mean() * 100:.2f}%")


def summarize(col, level):
    g = first.loc[known].groupby(col).size().sort_values(ascending=False).rename("visitors").reset_index()
    g = g.rename(columns={col: "categoryid"})
    g["level"] = level
    g["share_of_all_pct"] = (g["visitors"] / n_first * 100).round(2)
    g["share_of_known_pct"] = (g["visitors"] / known.sum() * 100).round(2)
    return g


root = summarize("root_categoryid", "root")
leaf = summarize("categoryid", "category")
# 카테고리는 대분류가 하나로 정해진다(전 이벤트에서 카테고리당 대분류 1개 확인). 라벨 표기용으로 붙인다.
cat_to_root = first.loc[known].drop_duplicates("categoryid").set_index("categoryid")["root_categoryid"]
assert first.loc[known].groupby("categoryid")["root_categoryid"].nunique().max() == 1
leaf["root_categoryid"] = leaf["categoryid"].map(cat_to_root).astype("Int64")
unknown = first.loc[~known, "category_match_status"].value_counts()
unk = pd.DataFrame(
    {
        "categoryid": [pd.NA] * len(unknown),
        "visitors": unknown.values,
        "level": ["unknown:" + s for s in unknown.index],
        "share_of_all_pct": (unknown.values / n_first * 100).round(2),
        "share_of_known_pct": [pd.NA] * len(unknown),
    }
)
root["root_categoryid"] = root["categoryid"]
table = pd.concat([root, leaf, unk], ignore_index=True)
TABLE.parent.mkdir(parents=True, exist_ok=True)
table[["level", "root_categoryid", "categoryid", "visitors", "share_of_all_pct", "share_of_known_pct"]].to_csv(TABLE, index=False, encoding="utf-8-sig")
assert root["visitors"].sum() == known.sum() == leaf["visitors"].sum()
print(f"대분류 {len(root)}개, 카테고리 {len(leaf)}개 | 상위 5 대분류:\n{root.head(5).to_string(index=False)}")
print(f"상위 {TOP_LEAF} 카테고리의 합 비중(전체 대비): {leaf.head(TOP_LEAF)['share_of_all_pct'].sum():.2f}%")
mon = pd.to_datetime(first["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m")
print("첫 view 월별 카테고리 확인 비율(%):\n", known.groupby(mon).mean().mul(100).round(1).to_string())

# 그래프 ------------------------------------------------------------
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID, BAR, GRAY = "#1f2933", "#616e7c", "#e8ecf1", "#2a6fbb", "#9aa5b1"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8.2), dpi=160, gridspec_kw={"width_ratios": [1, 1]})


def hbar(ax, labels, values, colors, title):
    y = list(range(len(values)))[::-1]
    ax.barh(y, values, color=colors, height=0.68)
    ax.set_yticks(y, labels, fontsize=10.5, color=INK)
    ax.tick_params(axis="y", length=0)
    top = max(values)
    ax.set_xlim(0, top * 1.28)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.tick_params(axis="x", colors=MUTED, length=0, labelsize=9.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    for yi, v in zip(y, values):
        ax.text(v + top * 0.012, yi, f"{v:,}명 ({v / n_first * 100:.1f}%)", va="center", fontsize=9.5, color=INK)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)


# 왼쪽: 대분류 전체 + 카테고리 미확인(회색)
l_labels = [f"대분류 {int(c)}" for c in root["categoryid"]] + ["카테고리 미확인"]
l_values = root["visitors"].tolist() + [int(unknown.sum())]
l_colors = [BAR] * len(root) + [GRAY]
hbar(ax1, l_labels, l_values, l_colors, f"대분류별 첫 조회 방문자 수 (대분류 {len(root)}개)")
# 오른쪽: 카테고리 상위
top = leaf.head(TOP_LEAF)
hbar(ax2, [f"대분류 {int(r)} › 카테고리 {int(c)}" for r, c in zip(top["root_categoryid"], top["categoryid"])], top["visitors"].tolist(), [BAR] * len(top), f"카테고리별 첫 조회 방문자 수 (상위 {TOP_LEAF}개)")

fig.suptitle(f"방문자의 첫 상품 조회 카테고리 (첫 조회 방문자 {n_first:,}명)", x=0.02, ha="left", fontsize=17, fontweight="bold", color=INK)
fig.text(
    0.02, 0.012,
    f"방문자별 가장 이른 view 1건의 카테고리 기준, 비율은 첫 조회 방문자 전체 대비. 카테고리를 알 수 없는 {(~known).sum():,}명({(~known).mean() * 100:.1f}%)은 회색으로 따로 표시. "
    "카테고리 이름은 제공되지 않아 ID로 표시. 카테고리별 집계에는 소속 대분류를 함께 표기.",
    fontsize=9.5, color=MUTED,
)
fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.07, wspace=0.62)
FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE, facecolor="white")
print("저장:", FIGURE, "|", TABLE)
