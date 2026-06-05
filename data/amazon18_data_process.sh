python3 amazon18_data_process.py \
    --dataset Industrial_and_Scientific \    #Industrial_and_Scientific \
    --user_k 5 \
    --item_k 5 \
    --st_year 1996 \
    --st_month 10 \
    --ed_year 2018 \
    --ed_month 10 \
    --output_path ./Amazon18


    #  --dataset  /mnt/c/Users/omni/Desktop/下载/Industrial_and_Scientific.csv  \ # 例如 Industrial
    #  --user_k 5 \
    #  --item_k 5 \
    #  --st_year 2017 \
    #  --st_month 10 \
    #  --ed_year 2018 \
    #  --ed_month 11 \
    #  --output_path ./data/Amazon18

# 将 Amazon 原始评论数据 处理成 MiniOneRec 可用的格式。
# 脚本会尝试从 ../ 目录，即项目根目录读取以下文件：
# meta_{category}.json  {category}_5.json 或 {category}.json
# 输出：--output_path ./Amazon18 \
# --dataset Industrial_and_Scientific

# 原始数据 (../)
#     │
#     ├─ meta_Industrial_and_Scientific.json    ← 商品元数据
#     │
#     ├─ Industrial_and_Scientific_5.json       ← 评论数据（压缩版）
#     │  或
#     └─ Industrial_and_Scientific.json          ← 评论数据（完整版）
#             │
#             ↓
#     ┌─────────────────────────────────────┐
#     │  处理步骤：                           │
#     │  1. 清洗 HTML 标签                     │
#     │  2. 时间戳过滤 (1996-10 ~ 2018-10)    │
#     │  3. K-core 过滤 (user_k=5, item_k=5)  │
#     │  4. 构建交互序列                       │
#     │  5. 8:1:1 划分训练/验证/测试           │
#     └─────────────────────────────────────┘
#             │
#             ↓
#     输出到 ./Amazon18/Industrial_and_Scientific/