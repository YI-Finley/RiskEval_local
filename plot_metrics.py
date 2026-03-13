import json
import matplotlib.pyplot as plt

# 修改成你的 summary.json 路径
summary_path = "/home/datasets/feiyifang/RiskEval/output/qwen3-4b_judge_test/summary.json"

with open(summary_path) as f:
    summary = json.load(f)

metrics = summary["metrics_by_penalty"]

penalties = [float(p) for p in metrics.keys()]
abstention = [metrics[p]["abstention_rate"] for p in metrics]
regret = [metrics[p]["avg_normalized_regret"] for p in metrics]
consistency = [metrics[p]["policy_consistency"] for p in metrics]

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

axs[0].plot(penalties, abstention, marker="o")
axs[0].set_title("Abstention Rate")
axs[0].set_xlabel("Penalty")
axs[0].set_ylabel("Rate")

axs[1].plot(penalties, regret, marker="o")
axs[1].set_title("Normalized Regret")
axs[1].set_xlabel("Penalty")
axs[1].set_ylabel("Regret")

axs[2].plot(penalties, consistency, marker="o")
axs[2].set_title("Policy Consistency")
axs[2].set_xlabel("Penalty")
axs[2].set_ylabel("Consistency")

plt.tight_layout()
plt.savefig("metrics_plot.png")  # 保存成图片文件

