import json

# Input and output paths
src_path = "/home/comp/23481501/datasets/RiskEval/data/gsm8k_main_test.jsonl"
dst_path = "/home/comp/23481501/datasets/RiskEval/data/gsm8k_main_test_riskeval.jsonl"

def convert_line(idx, obj):
    """
    Convert GSM8K format to simplified RiskEval format.
    """
    return {
        "id": str(idx),
        "question": obj["question"].strip(),
        "answer": obj["answer"].strip()
    }

def main():
    with open(src_path, "r", encoding="utf-8") as fin, \
         open(dst_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            obj = json.loads(line)
            new_obj = convert_line(idx, obj)
            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

    print(f"Conversion complete. Output file: {dst_path}")

if __name__ == "__main__":
    main()

