import pandas as pd

# ============================================================
# 1. 文件路径：按你的实际位置修改
# ============================================================

TRIP_FILE = r"物理信息实验数据_50站点.csv"
STATION_FILE = r"station_index_id_50站点.csv"

OUTPUT_FILE = r"物理信息实验数据_1号线2号线_demo.csv"


# ============================================================
# 2. 读取站点编号表，提取 1号线 + 2号线对应的站点编号
#    你现有数据中：
#      L010 = 1号线
#      L020 = 2号线
# ============================================================

station_df = pd.read_csv(STATION_FILE, encoding="gbk")

selected_lines = ["L010", "L020"]

selected_station_ids = (
    station_df.loc[
        station_df["BlongedLineNumber"].isin(selected_lines),
        "index"
    ]
    .dropna()
    .astype(int)
    .unique()
)

selected_station_ids = set(selected_station_ids)

print("1号线、2号线站点编号数量：", len(selected_station_ids))
print("站点编号：", sorted(selected_station_ids))


# ============================================================
# 3. 读取已经完成特征工程的 50站点实验数据
# ============================================================

data = pd.read_csv(TRIP_FILE)

# 这 6 个字段必须全部属于 1号线或2号线站点集合
filter_columns = [
    "O_Station",
    "D_Station",
    "user_best_O",
    "user_best_D",
    "user_second_D",
    "O_best_D",
]

# 转成数值，避免 CSV 中出现 1.0 / "1" 等格式导致匹配失败
for col in filter_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")


# ============================================================
# 4. 筛选
#
# 条件：
# O_Station
# D_Station
# user_best_O
# user_best_D
# user_second_D
# O_best_D
#
# 六个编号全部必须位于 1号线或2号线
# ============================================================

mask = pd.Series(True, index=data.index)

for col in filter_columns:
    mask &= data[col].isin(selected_station_ids)

demo_data = data.loc[mask].copy().reset_index(drop=True)


# ============================================================
# 5. 输出
#    保留原文件中的全部字段，只筛选记录，不删除其他列
# ============================================================

demo_data.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# 6. 输出统计信息
# ============================================================

print("\n" + "=" * 60)
print("筛选完成")
print("=" * 60)

print(f"原始记录数：{len(data):,}")
print(f"筛选后记录数：{len(demo_data):,}")
print(f"保留比例：{len(demo_data) / len(data) * 100:.2f}%")

print("\n筛选后各字段涉及的站点数量：")
for col in filter_columns:
    print(
        f"{col:20s}: "
        f"{demo_data[col].nunique():3d} 个"
    )

print(f"\n输出文件：{OUTPUT_FILE}")
