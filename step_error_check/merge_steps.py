import json

problem_id = 13
input_file = f"/Users/cusgadmin/Desktop/Project/LLM/reasoning/data/labeled_steps_problem_{problem_id}.json"
output_file = f"/Users/cusgadmin/Desktop/Project/LLM/reasoning/data/merged_steps_problem_{problem_id}.json"

# 读取标注数据
with open(input_file, "r", encoding='utf-8') as f:
    data = json.load(f)

# 合并步骤并保存详细信息
merged_steps_detail = []
merged_step_indices = []  # 记录每个合并步骤包含的原始步骤索引
merged_labels = []  # 记录每个合并步骤的标签序列

current_content = ""
current_indices = []
current_labels = []
last_label = ''

for i, item in enumerate(data["labeled_steps"]):
    label = item["label"]
    reasoning_step = item["reasoning_step"]
    step_index = item["step_index"]

    # 判断是否需要合并到上一个步骤
    should_merge = False
    merge_reason = ""

    if label == "Execute Solving Strategy and Calculate" and len(merged_steps_detail) > 0:
        should_merge = True
        merge_reason = "Execute step merged with previous step"
    elif label == "Setting Problem Solving Strategy" and last_label == "Exploring Alternative Approach":
        should_merge = True
        merge_reason = "Setting strategy merged after alternative approach"
    elif label == "Review Previous Steps" and last_label == "Review Previous Steps":
        should_merge = True
        merge_reason = "Consecutive review steps merged"
    elif label == "Obtaining Intermediate Results" and last_label == "Obtaining Intermediate Results":
        should_merge = True
        merge_reason = "Consecutive intermediate results steps merged"

    if should_merge:
        # 合并到上一个步骤
        merged_steps_detail[-1]["content"] += '\n\n' + reasoning_step
        merged_steps_detail[-1]["original_indices"].append(step_index)
        merged_steps_detail[-1]["labels"].append(label)
        merged_steps_detail[-1]["merge_count"] += 1
        merged_steps_detail[-1]["merge_reasons"].append(merge_reason)
    else:
        # 创建新的合并步骤
        merged_steps_detail.append({
            "merged_step_index": len(merged_steps_detail),
            "content": reasoning_step,
            "original_indices": [step_index],
            "labels": [label],
            "primary_label": label,
            "merge_count": 1,
            "merge_reasons": []
        })

    last_label = label

# 构建输出数据结构
output_data = {
    "metadata": {
        "problem_id": data.get("problem_id", "unknown"),
        "problem": data.get("problem", ""),
        "original_steps_count": len(data["labeled_steps"]),
        "merged_steps_count": len(merged_steps_detail),
        "reduction_rate": f"{(1 - len(merged_steps_detail) / len(data['labeled_steps'])) * 100:.2f}%"
    },
    "merge_rules": [
        "Execute Solving Strategy and Calculate steps are merged with previous steps",
        "Setting Problem Solving Strategy steps after Exploring Alternative Approach are merged",
        "Consecutive Review Previous Steps are merged together"
    ],
    "merged_steps": merged_steps_detail,
    "simple_steps": [step["content"] for step in merged_steps_detail]  # 简化版本，仅包含内容
}

# 打印预览
print("="*100)
print(f"原始步骤数: {output_data['metadata']['original_steps_count']}")
print(f"合并后步骤数: {output_data['metadata']['merged_steps_count']}")
print(f"压缩率: {output_data['metadata']['reduction_rate']}")
print("="*100)
print()

for step in merged_steps_detail:
    print(f"步骤 {step['merged_step_index'] + 1} [标签: {step['primary_label']}]")
    if step['merge_count'] > 1:
        print(f"  (合并了 {step['merge_count']} 个原始步骤: {step['original_indices']})")
    print(f"  标签序列: {' -> '.join(step['labels'])}")
    print()
    print(step["content"])
    print('-'*100)
    print()

# 保存到文件
with open(output_file, "w", encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n✓ 合并结果已保存到: {output_file}")
print(f"  - 原始步骤: {output_data['metadata']['original_steps_count']}")
print(f"  - 合并后步骤: {output_data['metadata']['merged_steps_count']}")
print(f"  - 压缩率: {output_data['metadata']['reduction_rate']}")