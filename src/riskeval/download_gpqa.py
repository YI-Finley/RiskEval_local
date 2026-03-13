from datasets import load_dataset

ds = load_dataset("nichenshun/gpqa_diamond")

# 给每条数据加上 id
data_with_id = []
for i, row in enumerate(ds["train"]):
    row = dict(row)
    row["id"] = str(i)  # 用行号作为唯一 id
    data_with_id.append(row)

import json
with open("/home/datasets/feiyifang/RiskEval/data/gpqa_diamond_test.jsonl", "w", encoding="utf-8") as f:
    for row in data_with_id:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
