from openai import OpenAI
from simple_json_reader import get_json_item
import json
import os


openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
    timeout=1200.0
)

problem_id=13

item = get_json_item("/Users/cusgadmin/Desktop/Project/LLM/reasoning/open_router/aime2025_dpsk_r1.jsonl", problem_id)

reasoning_trace = item["reasoning"]


prompt_path = '/Users/cusgadmin/Desktop/Project/LLM/reasoning/data/aime2025.jsonl'
prompts = []
with open(prompt_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        prompts.append(data["question"])

problem = prompts[problem_id]

keywords = ['Wait,', 'Alternatively,', 'Let me verify', "let's verify", "another idea", "I should check",
            "Let me double-check", "Let's think again", "Let me step back"]

keywords = [
    'Alternatively'
    "Wait",
    "Wait,",
    "Hold on",
    "Hold on a second",
    "Instead",
    "Let’s explore alternative approaches",
    "But let’s check",
    "I should check",
    "Another thought",
    "Let me re-examine",
    "Looking at the answer choices",
    "Looking at the other choices",
    "Another angle",
    "Let’s also think about",
    "So back to",
    "Looking at the candidate answers",
    "Let me reconsider",
    "re-check",
    "First,",
    "Let me step back",
    "Let me double check",
    "Am I missing something",
    "Looking at other approaches",
    "But wait",
    "Let me verify",
    "I should double-check",
    "Let me confirm",
    "Let’s look at the options",
    "Looking at the answer options",
    "Another check",
    "Let me think about",
    "Another possibility",
    "second thought",
    "Let’s go back",
    "reconsider",
    "go though each option",
    "Hang on",
    "Hold on a minute",
    "Alternatively",
    "I’ll approach this from another angle",
    "Let me check",
    "Let’s check",
    "Let’s verify",
    "Let me double-check",
    "Looking at the options",
    "Let’s look at each choice",
    "Let me just confirm",
    "Let’s think again",
    "Another point",
    "Let’s proceed step by step",
    "Let’s break it down",
    "re-analyze",
    "re-examine",
    "another approach"
]

dual_keywords = []
for keyword in keywords:
    if '’' in keyword:
        dual_keywords.append(keyword.replace('’', "'"))
    else:
        dual_keywords.append(keyword)

keywords = keywords + dual_keywords

step_trace = reasoning_trace.split('\n\n')

parse_result = ['']
for step in step_trace:
    flag = 0
    for keyword in keywords:
        if keyword in step:
            parse_result.append(step)
            flag = 1
            break
    if flag == 0:
        parse_result[-1] = parse_result[-1] + '\n\n' + step


prefix = r"""Your are a experienced mathematician good at identifying the high-level function of a reasoning step.

### **Task Description**
Given a reasoning step and its previous reasoning steps, the task is to identify the high-level function of current reasoning step.

### **High-Level function of a reasoning step**
Reasoning step should belong to one of the following **7 high-level function**:
1. **Understanding the Problem**: Identifying given data, definitions, and the goal.
2. **Setting Problem Solving Strategy**: Choose a problem solving strategy to solve the problem.
3. **Execute Solving Strategy and Calculate**: Execute the problem solving strategy or do numerical calculation or algebraic manipulation.
4. **Obtaining Intermediate Results**: Obtaining intermediate results or new insights of the problem.
5. **Review Previous Steps**: Checking for errors or inconsistencies within previous reasoning steps.
6. **Exploring Alternative Approach**: Considering another method to solve the problem. This meaning should explicitly begin with expressions like 'alternatively, let's try' or 'let's try another approach'.
7. **Finalize and Present the Answer**: Writing the final result and ensuring clarity.

Here is the previous reasoning steps
<previous_reasoning_steps>
{previous_reasoning_steps}
</previous_reasoning_steps>

Here is the current reasoning step
<current_reasoning_step>
{current_reasoning_step}
</current_reasoning_step>

### **Output Format**
Output in json format enclosed by <output> and </output> tags.
```
<output>
{{
    "High_Level_Function_Name": str(High Level Function Name of current reasoning step),
}}
</output>
```
"""

labeled_result_sentences = ['']
labeled_result = []
k = 4

# 定义输出文件路径
output_file = f"/Users/cusgadmin/Desktop/Project/LLM/reasoning/data/labeled_steps_problem_{problem_id}.json"

for i in range(len(parse_result)):
    start_ind = max(0, len(labeled_result_sentences) - k)
    previous_reasoning_steps = '\n\n'.join(labeled_result_sentences[start_ind:])
    current_reasoning_step = parse_result[i]

    # 重新格式化 prompt (避免重复使用同一变量)
    current_prompt = r"""Your are a experienced mathematician good at identifying the high-level function of a reasoning step.

### **Task Description**
Given a reasoning step and its previous reasoning steps, the task is to identify the high-level function of current reasoning step.

### **High-Level function of a reasoning step**
Reasoning step should belong to one of the following **7 high-level function**:
1. **Understanding the Problem**: Identifying given data, definitions, and the goal.
2. **Setting Problem Solving Strategy**: Choose a problem solving strategy to solve the problem.
3. **Execute Solving Strategy and Calculate**: Execute the problem solving strategy or do numerical calculation or algebraic manipulation.
4. **Obtaining Intermediate Results**: Obtaining intermediate results or new insights of the problem.
5. **Review Previous Steps**: Checking for errors or inconsistencies within previous reasoning steps.
6. **Exploring Alternative Approach**: Considering another method to solve the problem. This meaning should explicitly begin with expressions like 'alternatively, let's try' or 'let's try another approach'.
7. **Finalize and Present the Answer**: Writing the final result and ensuring clarity.

Here is the previous reasoning steps
<previous_reasoning_steps>
{previous_reasoning_steps}
</previous_reasoning_steps>

Here is the current reasoning step
<current_reasoning_step>
{current_reasoning_step}
</current_reasoning_step>

### **Output Format**
Output in json format enclosed by <output> and </output> tags.
```
<output>
{{
    "High_Level_Function_Name": str(High Level Function Name of current reasoning step),
}}
</output>
```
"""
    current_prompt = current_prompt.format(
        previous_reasoning_steps=previous_reasoning_steps,
        current_reasoning_step=current_reasoning_step
    )

    messages=[
        {"role": "user", "content": current_prompt}
    ]

    try:
        print(f"\n正在为第 {i+1}/{len(parse_result)} 个步骤请求 API...\n")
        stream = openrouter_client.chat.completions.create(
            model="openai/gpt-5-mini",
            messages=messages,
            stream=True,
            max_tokens=32768,
            temperature=0.8,
        )

        # 流式输出内容和思考过程
        full_reasoning = ""
        full_content = ""
        reasoning_started = False
        finish_reason = None

        for chunk in stream:
            delta = chunk.choices[0].delta

            # 检查停止原因
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            # 输出思考过程 (reasoning)
            if hasattr(delta, 'reasoning') and delta.reasoning:
                if not reasoning_started:
                    print("\n=== 思考过程 ===\n", flush=True)
                    reasoning_started = True
                print(delta.reasoning, end="", flush=True)
                full_reasoning += delta.reasoning

            # 输出最终答案 (content)
            if delta.content:
                if reasoning_started and not full_content:
                    print("\n\n=== 最终答案 ===\n", flush=True)
                print(delta.content, end="", flush=True)
                full_content += delta.content

        print("\n")

        # 提取标签
        label = "Unknown"
        try:
            # 从 <output> 标签中提取 JSON
            if "<output>" in full_content and "</output>" in full_content:
                json_str = full_content.split("<output>")[1].split("</output>")[0].strip()
                # 移除可能的 ``` 标记
                if json_str.startswith("```"):
                    json_str = json_str.split("\n", 1)[1]
                if json_str.endswith("```"):
                    json_str = json_str.rsplit("\n", 1)[0]

                label_data = json.loads(json_str)
                label = label_data.get("High_Level_Function_Name", "Unknown")
            else:
                print("警告: 未找到 <output> 标签，尝试直接解析 JSON")
                label_data = json.loads(full_content)
                label = label_data.get("High_Level_Function_Name", "Unknown")
        except Exception as e:
            print(f"解析标签时出错: {e}")
            print(f"原始内容: {full_content}")

        # 保存标记结果
        labeled_result.append({
            "step_index": i,
            "reasoning_step": current_reasoning_step,
            "label": label,
            "raw_response": full_content
        })

        labeled_result_sentences.append(current_reasoning_step)

        print(f"步骤 {i+1} 标签: {label}\n")

        # 每次API调用后立即保存结果
        output_data = {
            "problem_id": problem_id,
            "problem": problem,
            "total_steps": len(parse_result),
            "processed_steps": len(labeled_result),
            "labeled_steps": labeled_result
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✓ 已保存到: {output_file}")
        print("="*50 + "\n")

    except Exception as e:
        print(f"错误: {e}")
        labeled_result.append({
            "step_index": i,
            "reasoning_step": current_reasoning_step,
            "label": "Error",
            "error": str(e)
        })

        # 即使出错也保存当前进度
        output_data = {
            "problem_id": problem_id,
            "problem": problem,
            "total_steps": len(parse_result),
            "processed_steps": len(labeled_result),
            "labeled_steps": labeled_result
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✓ 错误发生后已保存当前进度到: {output_file}")
        print("="*50 + "\n")

print(f"\n✓ 所有标注完成！最终结果已保存到: {output_file}")
print(f"总共处理了 {len(labeled_result)}/{len(parse_result)} 个步骤")