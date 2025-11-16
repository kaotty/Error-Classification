import json
import re

# 输入和输出文件路径
input_file = "/Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/test/qwen-4b_aime2025.json"
output_file = "/Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/test/qwen-4b_aime2025.jsonl"

# 读取JSON文件
print(f"Reading {input_file}...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total records: {len(data)}")

# 处理每条记录
converted_count = 0
with open(output_file, 'w', encoding='utf-8') as f:
    for record in data:
        trace_id = record.get("trace_id", "")

        # 解析trace_id: problem_{problem_id}-{model_id}-{answer_id}
        # 例如: problem_1-Qwen3_4B-1
        match = re.match(r'problem_(\d+)-(.+)-(\d+)', trace_id)

        if match:
            problem_id = int(match.group(1))
            model_id = match.group(2)
            answer_id = int(match.group(3))

            # 创建新的记录，添加分解后的字段
            new_record = {
                "model_id": model_id,
                "problem_id": problem_id,
                "answer_id": answer_id,
                "trace_id": trace_id,  # 保留原始trace_id
                "problem_text": record.get("problem_text", ""),
                "ground_truth_answer": record.get("ground_truth_answer", ""),
                "generated_by_model": record.get("generated_by_model", ""),
                "reasoning_trace": record.get("reasoning_trace", ""),
                "predicted_answer": record.get("predicted_answer", "")
            }

            # 写入JSONL文件（每行一个JSON对象）
            f.write(json.dumps(new_record, ensure_ascii=False) + '\n')
            converted_count += 1
        else:
            print(f"Warning: Could not parse trace_id: {trace_id}")

print(f"\n✓ Conversion complete!")
print(f"  - Input: {input_file}")
print(f"  - Output: {output_file}")
print(f"  - Converted: {converted_count} records")
