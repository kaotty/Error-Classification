from openai import OpenAI
import json

# 配置 OpenRouter API
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
    timeout=1200.0
)

# 文件路径配置
problem_id = 6
input_file = f"/Users/cusgadmin/Desktop/Project/LLM/reasoning/data/merged_steps_problem_{problem_id}.json"
output_file = f"/Users/cusgadmin/Desktop/Project/LLM/reasoning/data/step_check_problem_{problem_id}_v1.json"


# prompt_path = '/Users/cusgadmin/Desktop/Project/LLM/reasoning/data/aime2025.jsonl'
# prompts = []
# with open(prompt_path, 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         prompts.append(data["question"])

# problem = prompts[problem_id]

# 读取合并后的步骤数据
print(f"正在读取文件: {input_file}")
with open(input_file, "r", encoding='utf-8') as f:
    data = json.load(f)

# 获取问题描述和步骤列表
problem = data.get("metadata", {}).get("problem", "")
merged_steps = data.get("merged_steps", [])

# 如果使用的是旧格式（没有metadata），尝试从simple_steps获取
if not merged_steps and "simple_steps" in data:
    merged_steps = [{"content": step, "merged_step_index": i} for i, step in enumerate(data["simple_steps"])]

print(f"问题ID: {problem_id}")
print(f"总步骤数: {len(merged_steps)}")
print("="*100 + "\n")
print(problem)
# exit()
# 定义检查 prompt 模板

refer_answer = r"""Note that order does not matter here. This is because any permutation of the 6 pairs will automatically get ordered in alphabetical order. The same is true for within each of the pairs. In other words, **AB CH DI EJ FK GL** should be counted equally as **HC AB DI EJ FK GL**.

We construct two cases: **G** is the first letter of the last word and **G** is the second letter of the last word.

---

### Case 1: G is the first letter of the last word

Our first case is when **G** is the first letter of the last word. Then the second letter of the last word must be one of **H, I, J, K, L**. Call that set of 5 letters **Ω**.  
There are **5** ways to choose the second letter from **Ω**. The other 4 letters of **Ω** must be used in the other 5 words.

For the other 5 words, each of their first letters must be before **G** in the alphabet. Otherwise, the word with **G** will not be the last. There are 6 letters before **G**: **A, B, C, D, E, F**. Call that set of 6 letters **Σ**.  
Exactly one of the words must have two letters from **Σ**. The other 4 will have their first letter from **Σ** and the second letter from **Ω**. There are **4!** ways to determine the possible pairings of letters from **Σ** and **Ω**, respectively.

Therefore, this case has:

\[
5 \cdot \binom{6}{2} \cdot 4! = 5 \cdot 15 \cdot 24 = 1800
\]

orderings.

---

### Case 2: G is the second letter of the last word

The second case is when **G** is the second letter of the last word. You can see that the first letter of that word must be **F**. Otherwise, that word cannot be the last word.  
The other 5 words must start with **A, B, C, D, E**. The second letter of each of those words will come from **Ω**.  
There will be **5!** ways to distribute the elements of **Ω** to one of **A, B, C, D, E**. There are therefore:

\[
5! = 120
\]

orderings in the case.

---

### Total Orderings

In total, there are:

\[
1800 + 120 = 1920
\]

orderings. However, we want the probability. The number of ways to put the 12 letters into pairs is:

\[
11 \cdot 9 \cdot 7 \cdot 5 \cdot 3 \cdot 1
\]

This is true because we can say this: Start with **A**. It has 11 options for who it will partner with. There are now 10 letters left. Pick one of those letters. It has 9 options for who it will partner with. There are now 8 letters left, and so on, until there are only 2 letters left, and there is only 1 option for that last word.  
Therefore, there will be **11 · 9 · 7 · 5 · 3 · 1** options.

The probability is therefore:

\[
\frac{1920}{11 \cdot 9 \cdot 7 \cdot 5 \cdot 3 \cdot 1} = \frac{128}{693}.
\]

The requested answer is:

\[
128 + 693 = \boxed{821}.
\]
"""

"""
| Code | Label | Definition | Examples |
|------|-------|------------|----------|
| **FE** | Formula Error | Using wrong formula for context, applying theorem without meeting prerequisites, misapplying valid technique to invalid domain | Using L'Hôpital without indeterminate form |
| **CE** | Conceptual Error | Ignoring, misreading, or misunderstanding key constraints/conditions/objectives before solving | Misunderstanding inequality condition as equality condition|
| **CA** | Calculation Error | Failing at pure mathematical execution (arithmetic, algebraic simplification, counting) despite correct logic/formula | 2+3=6, simplifying x²/x as x² |
| **CS** | Contradictory Step | Inconsistent reasoning between preceding and subsequent steps | Stating x>0 then using x=-2 |
| **MS** | Missing Step | Jumping to incorrect conclusion by skipping critical intermediate reasoning | "Since x²=4, then x=2" (missing x=-2) |
| **MC** | Missing Case | Failing to consider all necessary cases/branches, leading to incomplete solution | Only considering positive roots when both exist |
| **HA** | Hallucination Error | Introducing non-existent facts, numbers, conditions, or theorems | Claiming "by Fermat's Last Theorem" for unrelated problem |
| **VE** | Verification Error | Failing to check solution against original physical/contextual constraints | Accepting negative length as answer |
"""

prompt_template = r"""You are an expert mathematical reasoning validator tasked with analyzing the correctness and dependencies of reasoning steps in a mathematical solution.

## **Context**
You are analyzing a step-by-step mathematical solution where each step has been parsed and needs to be validated. Your analysis will be used to construct an information flow graph (similar to a Graphviz diagram) showing how reasoning flows through the solution.

## **Your Task**
For the current reasoning step, you must:

1. **Assess Method Consistency**: Determine if the approach/method used in this step aligns with the reference solution's methodology.

2. **Validate Correctness**: Check if the mathematical logic, calculations, and conclusions in this step are correct.

3. **Classify Errors** (if any): If the step is incorrect, identify and categorize the type(s) of error using the provided error classification system.

4. **Identify Dependencies**: Determine which previous steps this current step depends on (information flow).

5. **Classify Relationships**: Categorize how this step relates to its dependencies.

6. **Write Summary**: Create a concise, descriptive summary of the current step's purpose and key results.

## **Analysis Framework**
### 1. Method Consistency Assessment
- Compare the approach/methodology with the reference solution
- Focus on whether the mathematical technique, strategy, or reasoning path is similar
- Note: Different notation or intermediate steps are acceptable if the core method is the same
- Examples of method consistency:
  • Both algebraic approach to solve the problem
  • Both apply the same theorem or formula
  • Both follow similar logical reasoning chains
- Examples of method inconsistency:
  • One uses algebraic approach while the other uses geometric
  • Different problem-solving strategies (e.g., direct proof vs. contradiction)
  • Alternative formulas or theorems to reach the same result

### 2. Correctness Validation
- Verify mathematical logic, calculations, and conclusions
- Check for computational errors, logical fallacies, or invalid assumptions
- Assess whether conclusions follow from premises
- If an error is found, classify it according to the error categories below
Tip: If current step has conclusions conflict with the reference solution, then the current step is incorrect.

### 3. Error Type Classification (if applicable)
When an error is detected, classify it using one or more of these categories:

| Code | Label | Definition | Examples |
|------|-------|------------|----------|
| **FE** | Formula Error | Using wrong formula for context, applying theorem without meeting prerequisites, misapplying valid technique to invalid domain | Using L'Hôpital without indeterminate form |
| **CE** | Conceptual Error | Ignoring, misreading, or misunderstanding key constraints/conditions/objectives before solving | Misunderstanding inequality condition as equality condition|
| **CA** | Calculation Error | Failing at pure mathematical execution (arithmetic, algebraic simplification, counting) despite correct logic/formula | 2+3=6, simplifying x²/x as x² |
| **CS** | Contradictory Step | Inconsistent reasoning between preceding and subsequent steps | Stating x>0 then using x=-2 |
| **MS** | Missing Step | Jumping to incorrect conclusion by skipping critical intermediate reasoning | "Since x²=4, then x=2" (missing x=-2) |
| **MC** | Missing Case | Failing to consider all necessary cases/branches, leading to incomplete solution | Only considering positive roots when both exist |
| **HA** | Hallucination Error | Introducing non-existent facts, numbers, conditions, or theorems | Claiming "by Fermat's Last Theorem" for unrelated problem |
| **VE** | Verification Error | Failing to check solution against original physical/contextual constraints | Accepting negative length as answer |

### 4. Dependency Identification
Determine parent steps where the current step:
- Uses computed results, equations, or values
- References established facts, properties, or constraints
- Builds upon previous reasoning or logic chains
- Verifies, validates, or reviews prior conclusions

### 5. Relationship Classification

| Type | Description | Examples |
|------|-------------|----------|
| **Progressive** | Advances solution by building on parent(s) | • Applying derived formulas<br>• Using computed values<br>• Extending logic chains |
| **Review** | Validates parent(s) without advancing | • Verifying calculations<br>• Confirming conditions<br>• Checking assumptions |
| **Corrective** | Identifies and fixes errors in parent(s) | • Recalculating values<br>• Revising approaches<br>• Correcting logic |

### 6. Summary Writing Guidelines for Graphviz Display

The summary will appear as a node label in a Graphviz diagram, so it must be:

**Structure**: `[Action] + [Key Operation/Method] + [Result/Value]`

**Length**: Maximum 60 characters for optimal display

**Content Priorities** (in order):
1. **Mathematical operation** performed (e.g., "Calculate", "Derive", "Verify", "Apply")
2. **Key value or formula** involved (use mathematical notation when possible)
3. **Specific result** obtained (include numerical values if critical)

## **Input Information**

### Mathematical Problem
Here is the math problem
<problem>
{problem}
</problem>

### Reference Solution
Here is the reference solution
<reference_solution>
{reference_solution}
</reference_solution>

### Previously Validated Steps
Here is the previous reasoning steps
<previous_reasoning_steps>
{previous_reasoning_steps}
</previous_reasoning_steps>

### Current Step to Analyze
Here is the current reasoning step
<current_reasoning_step>
{current_reasoning_step}
</current_reasoning_step>

### **Output Format**
Output in json format enclosed by <output> and </output> tags.
```
<output>
{{
    "Correctness": bool (True if the current reasoning step is correct, False if the current reasoning step is incorrect),
    "Error_Type": str (The type of error the current reasoning step contains, one of the following: FE, CE, CA, CS, MS, MC, HA, VE),
    "Incorrect_Reason": str (Detailed explanation of why the step is incorrect and what specific errors were made. Empty string if correct),
    "Method_Consistency": bool (True if the current step's approach/method aligns with the reference solution, False if using a different method),
    "Step_Relation": [
                        {{
                            "Parent_Step": str (Name of the parent step, e.g., "Step 1", "Step 2"),
                            "Relation": str (Relation between the current reasoning step and the parent step: "Progressive" or "Review"),
                        }}
                        ],
    "Summary": str (One sentence summary of the current reasoning step, including the main idea and the key results),
}}
</output>
```


## **Analysis Instructions**

1. First, read through ALL previous steps to understand the solution context
2. Compare the current step's methodology with the reference solution to assess consistency
3. Validate the mathematical correctness of the current step
4. If errors are found, classify them using the error type codes (FE, CE, CA, CS, MS, MC, HA, VE)
5. Identify what specific information the current step uses from previous steps
6. Be precise about dependencies - only mark actual information flow, not mere sequential ordering
7. Consider that a step may have multiple parent steps with different relationship types

Output your complete analysis:
"""

# 存储检查结果
check_results = []

# 遍历每个步骤进行检查
for i, step_data in enumerate(merged_steps):
    current_step_index = i
    current_step_content = step_data.get("content", step_data) if isinstance(step_data, dict) else step_data

    # 构建之前的步骤历史（带编号、正确性和错误原因）
    previous_steps_list = []
    for j in range(i):
        prev_content = merged_steps[j].get("content", merged_steps[j]) if isinstance(merged_steps[j], dict) else merged_steps[j]

        # 构建步骤头部信息
        step_header = f"Step {j+1}"

        # 如果已经检查过这个步骤，添加正确性信息
        if j < len(check_results):
            prev_result = check_results[j]
            correctness = prev_result.get("correctness")

            if correctness is True:
                step_header += " [✓ Correct]"
            elif correctness is False:
                step_header += " [✗ Incorrect]"
                incorrect_reason = prev_result.get("incorrect_reason", "")
                if incorrect_reason:
                    step_header += f"\nIncorrect Reason: {incorrect_reason}"
            else:
                step_header += " [? Unable to verify]"

        previous_steps_list.append(f"{step_header}:\n{prev_content}")

    previous_reasoning_steps = "\n\n".join(previous_steps_list) if previous_steps_list else "No previous steps."

    # 格式化当前 prompt
    current_prompt = prompt_template.format(
        problem=problem,
        reference_solution=refer_answer,
        previous_reasoning_steps=previous_reasoning_steps,
        current_reasoning_step=current_step_content
    )

    messages = [
        {"role": "user", "content": current_prompt}
    ]

    try:
        print(f"\n正在检查第 {i+1}/{len(merged_steps)} 个步骤...\n")
        stream = openrouter_client.chat.completions.create(
            model="openai/gpt-5-mini",
            messages=messages,
            stream=True,
            max_tokens=32768,
            temperature=0.3,  # 降低温度以获得更一致的检查结果
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
                    print("\n\n=== 检查结果 ===\n", flush=True)
                print(delta.content, end="", flush=True)
                full_content += delta.content

        print("\n")

        # 提取检查结果
        summary = ""
        correctness = True
        error_type = ""
        incorrect_reason = ""
        method_consistency = None
        step_relation = []

        try:
            # 从 <output> 标签中提取 JSON
            if "<output>" in full_content and "</output>" in full_content:
                json_str = full_content.split("<output>")[1].split("</output>")[0].strip()
                # 移除可能的 ``` 标记
                if json_str.startswith("```"):
                    json_str = json_str.split("\n", 1)[1]
                if json_str.endswith("```"):
                    json_str = json_str.rsplit("\n", 1)[0]

                check_data = json.loads(json_str)
                summary = check_data.get("Summary", "")
                correctness = check_data.get("Correctness", True)
                error_type = check_data.get("Error_Type", "")
                incorrect_reason = check_data.get("Incorrect_Reason", "")
                method_consistency = check_data.get("Method_Consistency", None)
                step_relation = check_data.get("Step_Relation", [])
            else:
                print("警告: 未找到 <output> 标签，尝试直接解析 JSON")
                check_data = json.loads(full_content)
                summary = check_data.get("Summary", "")
                correctness = check_data.get("Correctness", True)
                error_type = check_data.get("Error_Type", "")
                incorrect_reason = check_data.get("Incorrect_Reason", "")
                method_consistency = check_data.get("Method_Consistency", None)
                step_relation = check_data.get("Step_Relation", [])
        except Exception as e:
            print(f"解析检查结果时出错: {e}")
            print(f"原始内容: {full_content}")

        # 保存检查结果
        result_entry = {
            "step_index": current_step_index,
            "step_number": current_step_index + 1,
            "reasoning_step": current_step_content,
            "summary": summary,
            "correctness": correctness,
            "error_type": error_type,
            "incorrect_reason": incorrect_reason,
            "method_consistency": method_consistency,
            "step_relation": step_relation,
            "raw_response": full_content
        }

        check_results.append(result_entry)

        # 显示摘要
        print(f"步骤 {i+1} 检查结果:")
        if summary:
            print(f"  - 摘要: {summary}")
        print(f"  - 正确性: {'✓ 正确' if correctness else '✗ 错误'}")
        if not correctness and error_type:
            print(f"  - 错误类型: {error_type}")
        if incorrect_reason:
            print(f"  - 错误原因: {incorrect_reason}")
        if method_consistency is not None:
            consistency_str = '✓ 与参考解决方案一致' if method_consistency else '✗ 方法不一致'
            print(f"  - 方法一致性: {consistency_str}")
        if step_relation:
            print(f"  - 依赖关系: {step_relation}")
        print()

        # 每次检查后立即保存结果
        output_data = {
            "metadata": {
                "problem_id": problem_id,
                "problem": problem,
                "total_steps": len(merged_steps),
                "checked_steps": len(check_results),
                "correct_steps": sum(1 for r in check_results if r["correctness"]),
                "incorrect_steps": sum(1 for r in check_results if not r["correctness"])
            },
            "check_results": check_results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✓ 已保存到: {output_file}")
        print("="*100 + "\n")

    except Exception as e:
        print(f"错误: {e}")
        result_entry = {
            "step_index": current_step_index,
            "step_number": current_step_index + 1,
            "reasoning_step": current_step_content,
            "summary": "",
            "correctness": None,
            "error_type": "",
            "incorrect_reason": f"API调用错误: {str(e)}",
            "method_consistency": None,
            "step_relation": [],
            "error": str(e)
        }

        check_results.append(result_entry)

        # 即使出错也保存当前进度
        output_data = {
            "metadata": {
                "problem_id": problem_id,
                "problem": problem,
                "total_steps": len(merged_steps),
                "checked_steps": len(check_results),
                "correct_steps": sum(1 for r in check_results if r.get("correctness") == True),
                "incorrect_steps": sum(1 for r in check_results if r.get("correctness") == False)
            },
            "check_results": check_results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✓ 错误发生后已保存当前进度到: {output_file}")
        print("="*100 + "\n")

# 最终统计
print("\n" + "="*100)
print("检查完成！")
print(f"✓ 结果已保存到: {output_file}")
print(f"\n统计信息:")
print(f"  - 总步骤数: {len(merged_steps)}")
print(f"  - 已检查: {len(check_results)}")
print(f"  - 正确步骤: {sum(1 for r in check_results if r.get('correctness') == True)}")
print(f"  - 错误步骤: {sum(1 for r in check_results if r.get('correctness') == False)}")
print(f"  - 无法判断: {sum(1 for r in check_results if r.get('correctness') is None)}")
print("="*100)