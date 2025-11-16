"""
批量分析脚本 - 对所有answer应用三步分析（segment, merge, check）
"""
import json
import os
from openai import OpenAI
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 未安装，跳过

# 配置
JSONL_FILE = "/Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/test/deepseek_r1/aime2025_dpsk_30_10_wrong_answers_refined.jsonl"
SOLUTIONS_FILE = "/Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/test/solutions.jsonl"
GROUND_TRUTH_FILE = "/Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/test/aime2025.jsonl"
BASE_OUTPUT_DIR = "/Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/step_error_check/deepseek_r1_aime2025_v4"

# 并行处理配置
MAX_WORKERS = 20  # 同时处理的答案数量（可根据API限流调整）

# 创建输出目录
SEGMENT_DIR = os.path.join(BASE_OUTPUT_DIR, "segment")
MERGE_DIR = os.path.join(BASE_OUTPUT_DIR, "merge")
CHECK_DIR = os.path.join(BASE_OUTPUT_DIR, "check")

os.makedirs(SEGMENT_DIR, exist_ok=True)
os.makedirs(MERGE_DIR, exist_ok=True)
os.makedirs(CHECK_DIR, exist_ok=True)

# 创建线程锁（用于并行处理时的输出控制）
print_lock = threading.Lock()

# 配置 OpenRouter API
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("="*80)
    print("ERROR: OPENROUTER_API_KEY environment variable is not set!")
    print("="*80)
    print("\nPlease set your API key using one of these methods:")
    print("\n1. Export environment variable:")
    print('   export OPENROUTER_API_KEY="your_api_key_here"')
    print("\n2. Create .env file:")
    print('   echo "OPENROUTER_API_KEY=your_api_key_here" > .env')
    print("   pip install python-dotenv")
    print("   # Then add 'from dotenv import load_dotenv; load_dotenv()' at the top")
    print("\n" + "="*80)
    import sys
    sys.exit(1)

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    timeout=1200.0
)

# 关键词列表（用于segment步骤）
KEYWORDS = [
    'Alternatively',
    "Wait",
    "Wait,",
    "Hold on",
    "Hold on a second",
    "Instead",
    "Let's explore alternative approaches",
    "But let's check",
    "I should check",
    "Another thought",
    "Let me re-examine",
    "Looking at the answer choices",
    "Looking at the other choices",
    "Another angle",
    "Let's also think about",
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
    "Let's look at the options",
    "Looking at the answer options",
    "Another check",
    "Let me think about",
    "Another possibility",
    "second thought",
    "Let's go back",
    "reconsider",
    "go though each option",
    "Hang on",
    "Hold on a minute",
    "I'll approach this from another angle",
    "Let me check",
    "Let's check",
    "Let's verify",
    "Let me double-check",
    "Looking at the options",
    "Let's look at each choice",
    "Let me just confirm",
    "Let's think again",
    "Another point",
    "Let's proceed step by step",
    "Let's break it down",
    "re-analyze",
    "re-examine",
    "another approach"
]

# 处理单引号变体
dual_keywords = []
for keyword in KEYWORDS:
    if '’' in keyword:
        dual_keywords.append(keyword.replace('’', "'"))
    else:
        dual_keywords.append(keyword)
KEYWORDS = KEYWORDS + dual_keywords


def load_reference_solutions():
    """
    加载参考解答
    返回: dict {problem_id: solution_text}
    """
    solutions = {}

    if not os.path.exists(SOLUTIONS_FILE):
        print(f"Warning: Solutions file not found: {SOLUTIONS_FILE}")
        print("Step 3 (check) will run without reference solutions")
        return solutions

    try:
        with open(SOLUTIONS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                problem_id = data.get('problem_id')
                solution_text = data.get('solution_text', '')
                if problem_id and solution_text:
                    solutions[problem_id] = solution_text

        print(f"✓ Loaded {len(solutions)} reference solutions")
        return solutions
    except Exception as e:
        print(f"Warning: Error loading solutions file: {e}")
        print("Step 3 (check) will run without reference solutions")
        return {}


def load_ground_truth_answers():
    """
    加载标准答案
    返回: dict {problem_id: answer}
    """
    answers = {}

    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Warning: Ground truth file not found: {GROUND_TRUTH_FILE}")
        print("All records will be processed (no filtering)")
        return answers

    try:
        with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 注意：aime2025.jsonl 中的字段是 Problem_id（大写P）
                problem_id = data.get('Problem_id') or data.get('problem_id')
                answer = data.get('answer', '')
                if problem_id and answer:
                    answers[problem_id] = str(answer).strip()

        print(f"✓ Loaded {len(answers)} ground truth answers")
        return answers
    except Exception as e:
        print(f"Warning: Error loading ground truth file: {e}")
        print("All records will be processed (no filtering)")
        return {}


def should_process_record(record, ground_truth_answers):
    """
    判断是否应该处理该记录
    条件：extracted_answer 不正确 且 不为 0
    """
    problem_id = record.get('problem_id')
    extracted_answer = str(record.get('extracted_answer', '')).strip()

    # 如果没有标准答案，默认处理
    if not ground_truth_answers:
        return True

    # 如果 extracted_answer 为 0，跳过
    if extracted_answer == '0' or extracted_answer == '':
        return False

    # 获取标准答案
    ground_truth = ground_truth_answers.get(problem_id, '')

    # 如果没有找到标准答案，默认处理
    if not ground_truth:
        return True

    # 如果答案不正确，需要处理
    is_incorrect = extracted_answer != ground_truth

    return is_incorrect


def step1_segment(record, output_file):
    """
    Step 1: 分割和标注推理步骤
    """
    print(f"\n{'='*80}")
    print(f"Step 1: Segmenting and Labeling")
    print(f"{'='*80}")

    problem_id = record['problem_id']
    answer_id = record['answer_id']
    reasoning = record['reasoning']
    problem = record['problem_text']

    # 分割推理轨迹
    step_trace = reasoning.split('\n\n')
    parse_result = ['']

    for step in step_trace:
        flag = 0
        for keyword in KEYWORDS:
            if keyword in step:
                parse_result.append(step)
                flag = 1
                break
        if flag == 0:
            parse_result[-1] = parse_result[-1] + '\n\n' + step

    # 标注步骤
    prompt_template = r"""Your are a experienced mathematician good at identifying the high-level function of a reasoning step.

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

    labeled_result = []
    labeled_result_sentences = ['']
    k = 4

    for i in range(len(parse_result)):
        start_ind = max(0, len(labeled_result_sentences) - k)
        previous_reasoning_steps = '\n\n'.join(labeled_result_sentences[start_ind:])
        current_reasoning_step = parse_result[i]

        current_prompt = prompt_template.format(
            previous_reasoning_steps=previous_reasoning_steps,
            current_reasoning_step=current_reasoning_step
        )

        messages = [{"role": "user", "content": current_prompt}]

        try:
            print(f"Processing step {i+1}/{len(parse_result)}...")

            stream = openrouter_client.chat.completions.create(
                model="openai/gpt-5-mini",
                messages=messages,
                stream=True,
                max_tokens=32768,
                temperature=0.8,
            )

            full_content = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content

            # 提取标签
            label = "Unknown"
            try:
                if "<output>" in full_content and "</output>" in full_content:
                    json_str = full_content.split("<output>")[1].split("</output>")[0].strip()
                    if json_str.startswith("```"):
                        json_str = json_str.split("\n", 1)[1]
                    if json_str.endswith("```"):
                        json_str = json_str.rsplit("\n", 1)[0]
                    label_data = json.loads(json_str)
                    label = label_data.get("High_Level_Function_Name", "Unknown")
            except Exception as e:
                print(f"Error parsing label: {e}")

            labeled_result.append({
                "step_index": i,
                "reasoning_step": current_reasoning_step,
                "label": label,
                "raw_response": full_content
            })

            labeled_result_sentences.append(current_reasoning_step)
            print(f"Step {i+1} label: {label}")

        except Exception as e:
            print(f"Error: {e}")
            labeled_result.append({
                "step_index": i,
                "reasoning_step": current_reasoning_step,
                "label": "Error",
                "error": str(e)
            })

    # 保存结果
    output_data = {
        "problem_id": problem_id,
        "answer_id": answer_id,
        "model_id": record.get("model_id", ""),
        "problem": problem,
        "total_steps": len(parse_result),
        "processed_steps": len(labeled_result),
        "labeled_steps": labeled_result
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved to: {output_file}")
    return output_data


def step2_merge(segment_data, output_file):
    """
    Step 2: 合并相关步骤
    """
    print(f"\n{'='*80}")
    print(f"Step 2: Merging Steps")
    print(f"{'='*80}")

    labeled_steps = segment_data.get("labeled_steps", [])

    merged_steps_detail = []
    last_label = ''

    for i, item in enumerate(labeled_steps):
        label = item.get("label", "")
        reasoning_step = item.get("reasoning_step", "")
        step_index = item.get("step_index", i)

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
            merged_steps_detail[-1]["content"] += '\n\n' + reasoning_step
            merged_steps_detail[-1]["original_indices"].append(step_index)
            merged_steps_detail[-1]["labels"].append(label)
            merged_steps_detail[-1]["merge_count"] += 1
            merged_steps_detail[-1]["merge_reasons"].append(merge_reason)
        else:
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

    output_data = {
        "metadata": {
            "problem_id": segment_data.get("problem_id"),
            "answer_id": segment_data.get("answer_id"),
            "model_id": segment_data.get("model_id", ""),
            "problem": segment_data.get("problem", ""),
            "original_steps_count": len(labeled_steps),
            "merged_steps_count": len(merged_steps_detail),
            "reduction_rate": f"{(1 - len(merged_steps_detail) / len(labeled_steps)) * 100:.2f}%" if len(labeled_steps) > 0 else "0%"
        },
        "merge_rules": [
            "Execute Solving Strategy and Calculate steps are merged with previous steps",
            "Setting Problem Solving Strategy steps after Exploring Alternative Approach are merged",
            "Consecutive Review Previous Steps are merged together",
            "Consecutive Obtaining Intermediate Results steps are merged together"
        ],
        "merged_steps": merged_steps_detail,
        "simple_steps": [step["content"] for step in merged_steps_detail]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Original steps: {output_data['metadata']['original_steps_count']}")
    print(f"Merged steps: {output_data['metadata']['merged_steps_count']}")
    print(f"Reduction rate: {output_data['metadata']['reduction_rate']}")
    print(f"✓ Saved to: {output_file}")

    return output_data


def step3_check(merge_data, output_file, reference_solution=""):
    """
    Step 3: 检查步骤正确性
    """
    print(f"\n{'='*80}")
    print(f"Step 3: Checking Steps")
    print(f"{'='*80}")

    metadata = merge_data.get("metadata", {})
    merged_steps = merge_data.get("merged_steps", [])
    problem = metadata.get("problem", "")

    # 如果没有参考解答，创建占位符
    if not reference_solution:
        print("Note: No reference solution - creating placeholder results")
        check_results = []
        for i, step in enumerate(merged_steps):
            check_results.append({
                "step_index": i,
                "step_number": i + 1,
                "reasoning_step": step.get("content", ""),
                "summary": f"Step {i+1}: {step.get('primary_label', 'Unknown')}",
                "correctness": None,
                "error_type": "",
                "incorrect_reason": "Not checked - reference solution required",
                "method_consistency": None,
                "step_relation": [],
                "note": "Automated checking requires reference solution"
            })

        output_data = {
            "metadata": {
                "problem_id": metadata.get("problem_id"),
                "answer_id": metadata.get("answer_id"),
                "model_id": metadata.get("model_id", ""),
                "problem": problem,
                "total_steps": len(merged_steps),
                "checked_steps": 0,
                "correct_steps": 0,
                "incorrect_steps": 0
            },
            "check_results": check_results,
            "note": "This is a placeholder. Full checking requires reference solutions."
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved to: {output_file}")
        return output_data

    # 有参考解答，进行实际检查
    print("Checking with reference solution...")

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
| **MC** | Misunderstanding Conditions/Constraints | Ignoring, misreading, or misunderstanding key constraints/conditions/objectives  | Misunderstanding 'sum of a, b, c' as 'product of a, b, c'|
| **CA** | Calculation Error | Failing at pure mathematical execution (arithmetic, algebraic simplification, counting) despite correct logic/formula | 2+3=6, simplifying x²/x as x² |
| **CS** | Contradictory Step | Inconsistent reasoning between preceding and subsequent steps | Stating x>0 then using x=-2 |
| **UC** | Unsupported Conclusion | Jumping to incorrect conclusion by skipping critical intermediate reasoning | "f(x) = x³ - 3x has a maximum at x = 1\" (Wrong! it's actually a minimum) |
| **MB** | Missing Branch | Failing to consider all necessary branches/cases, leading to incomplete solution | Only considering positive roots when both exist |
| **HA** | Hallucination Error | Introducing non-existent facts, numbers, conditions, or theorems | Claiming "by Fermat's Last Theorem" for unrelated problem |
| **GA** | Guess Answer Error | Guessing the answer without valid mathematical reasoning | Guessing the answer as 128+693=821 |

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
    "Error_Type": str (The type of error the current reasoning step contains, one of the following: FE, MC, CA, CS, UC, MB, HA, GA),
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
4. If errors are found, classify them using the error type codes (FE, MC, CA, CS, UC, MB, HA, GA)
5. Identify what specific information the current step uses from previous steps
6. Be precise about dependencies - only mark actual information flow, not mere sequential ordering
7. Consider that a step may have multiple parent steps with different relationship types

Output your complete analysis:
"""

    check_results = []
    checked_steps = 0
    correct_steps = 0
    incorrect_steps = 0

    for i, step_data in enumerate(merged_steps):
        current_step_index = i
        current_step_content = step_data.get("content", "")

        # 构建之前的步骤历史
        previous_steps_list = []
        for j in range(i):
            prev_content = merged_steps[j].get("content", "")
            step_header = f"Step {j+1}"

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

            previous_steps_list.append(f"{step_header}:\n{prev_content}")

        previous_reasoning_steps = "\n\n".join(previous_steps_list) if previous_steps_list else "No previous steps."

        # 格式化当前 prompt
        current_prompt = prompt_template.format(
            problem=problem,
            reference_solution=reference_solution,
            previous_reasoning_steps=previous_reasoning_steps,
            current_reasoning_step=current_step_content
        )

        messages = [{"role": "user", "content": current_prompt}]

        try:
            print(f"Checking step {i+1}/{len(merged_steps)}...")

            stream = openrouter_client.chat.completions.create(
                model="openai/gpt-5-mini",
                messages=messages,
                stream=True,
                max_tokens=32768,
                temperature=0.3,
            )

            full_content = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content

            # 提取检查结果
            summary = ""
            correctness = True
            error_type = ""
            incorrect_reason = ""
            method_consistency = None
            step_relation = []

            try:
                if "<output>" in full_content and "</output>" in full_content:
                    json_str = full_content.split("<output>")[1].split("</output>")[0].strip()
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
            except Exception as e:
                print(f"Error parsing check result: {e}")

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
            checked_steps += 1

            if correctness:
                correct_steps += 1
                print(f"  ✓ Step {i+1}: Correct")
            else:
                incorrect_steps += 1
                print(f"  ✗ Step {i+1}: Incorrect ({error_type})")

        except Exception as e:
            print(f"Error checking step {i+1}: {e}")
            result_entry = {
                "step_index": current_step_index,
                "step_number": current_step_index + 1,
                "reasoning_step": current_step_content,
                "summary": "",
                "correctness": None,
                "error_type": "",
                "incorrect_reason": f"API error: {str(e)}",
                "method_consistency": None,
                "step_relation": [],
                "error": str(e)
            }
            check_results.append(result_entry)

        # 每次检查后保存结果
        output_data = {
            "metadata": {
                "problem_id": metadata.get("problem_id"),
                "answer_id": metadata.get("answer_id"),
                "model_id": metadata.get("model_id", ""),
                "problem": problem,
                "total_steps": len(merged_steps),
                "checked_steps": checked_steps,
                "correct_steps": correct_steps,
                "incorrect_steps": incorrect_steps
            },
            "check_results": check_results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Checked {checked_steps} steps: {correct_steps} correct, {incorrect_steps} incorrect")
    print(f"✓ Saved to: {output_file}")
    return output_data


def process_record(record, reference_solutions=None, record_number=None, total_records=None):
    """
    处理单条记录，执行三步分析
    """
    if reference_solutions is None:
        reference_solutions = {}

    problem_id = record['problem_id']
    answer_id = record['answer_id']

    # 使用线程锁保护输出
    with print_lock:
        if record_number and total_records:
            print(f"\n[{record_number}/{total_records}] {'#'*60}")
        else:
            print(f"\n{'#'*80}")
        print(f"Processing: Problem {problem_id}, Answer {answer_id}")
        print(f"Model: {record.get('model_id', 'Unknown')}")
        print(f"Predicted: {record.get('extracted_answer', 'N/A')}")

        # 检查是否有参考解答
        reference_solution = reference_solutions.get(problem_id, "")
        if reference_solution:
            print(f"Reference solution: Available ({len(reference_solution)} chars)")
        else:
            print(f"Reference solution: Not available")

        if record_number and total_records:
            print(f"{'#'*60}")
        else:
            print(f"{'#'*80}")

    # Step 1: Segment
    segment_file = os.path.join(SEGMENT_DIR, f"labeled_steps_problem_{problem_id}_answer_{answer_id}.json")
    if os.path.exists(segment_file):
        print(f"Segment file exists, loading: {segment_file}")
        with open(segment_file, 'r', encoding='utf-8') as f:
            segment_data = json.load(f)
    else:
        segment_data = step1_segment(record, segment_file)

    # Step 2: Merge
    merge_file = os.path.join(MERGE_DIR, f"merged_steps_problem_{problem_id}_answer_{answer_id}.json")
    if os.path.exists(merge_file):
        print(f"Merge file exists, loading: {merge_file}")
        with open(merge_file, 'r', encoding='utf-8') as f:
            merge_data = json.load(f)
    else:
        merge_data = step2_merge(segment_data, merge_file)

    # Step 3: Check
    check_file = os.path.join(CHECK_DIR, f"step_check_problem_{problem_id}_answer_{answer_id}.json")
    if os.path.exists(check_file):
        print(f"Check file exists, skipping: {check_file}")
    else:
        step3_check(merge_data, check_file, reference_solution=reference_solution)

    return True


def main():
    """
    主函数：批量处理所有记录（带答案过滤和并行处理）
    """
    print("="*80)
    print("Batch Analysis Pipeline (Parallel Mode)")
    print("="*80)
    print(f"Input file: {JSONL_FILE}")
    print(f"Ground truth file: {GROUND_TRUTH_FILE}")
    print(f"Solutions file: {SOLUTIONS_FILE}")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print(f"  - Segment: {SEGMENT_DIR}")
    print(f"  - Merge: {MERGE_DIR}")
    print(f"  - Check: {CHECK_DIR}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print("="*80)

    # 加载标准答案
    print("\nLoading ground truth answers...")
    ground_truth_answers = load_ground_truth_answers()

    # 加载参考解答
    print("Loading reference solutions...")
    reference_solutions = load_reference_solutions()

    # 读取所有记录
    print("\nReading records...")
    all_records = []
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            all_records.append(json.loads(line))

    print(f"Total records in file: {len(all_records)}")

    # 过滤记录：只处理答案不正确且不为0的记录
    print("\nFiltering records (incorrect answers only, excluding 0)...")
    filtered_records = []
    skipped_correct = 0
    skipped_zero = 0

    for record in all_records:
        problem_id = record.get('problem_id')
        predicted = str(record.get('extracted_answer', '')).strip()
        ground_truth = ground_truth_answers.get(problem_id, '')
        # print(ground_truth)
        # exit()

        if predicted == '0' or predicted == '':
            skipped_zero += 1
        elif ground_truth and predicted == ground_truth:
            skipped_correct += 1
        else:
            filtered_records.append(record)

    print(f"Records to process: {len(filtered_records)}")
    print(f"  - Skipped (correct answer): {skipped_correct}")
    print(f"  - Skipped (answer is 0): {skipped_zero}")
    print(f"  - Total skipped: {skipped_correct + skipped_zero}")

    if not filtered_records:
        print("\nNo records to process!")
        return

    # 并行处理记录
    print(f"\nProcessing {len(filtered_records)} records with {MAX_WORKERS} workers...")
    print("="*80)

    success_count = 0
    error_count = 0
    completed_count = 0

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_record = {}
        for i, record in enumerate(filtered_records, 1):
            future = executor.submit(
                process_record,
                record,
                reference_solutions,
                i,
                len(filtered_records)
            )
            future_to_record[future] = (i, record)

        # 收集结果
        for future in as_completed(future_to_record):
            completed_count += 1
            record_num, record = future_to_record[future]

            try:
                result = future.result()
                success_count += 1

                with print_lock:
                    print(f"\n✓ [{completed_count}/{len(filtered_records)}] Completed: Problem {record['problem_id']}, Answer {record['answer_id']}")

            except Exception as e:
                error_count += 1
                with print_lock:
                    print(f"\n❌ [{completed_count}/{len(filtered_records)}] Error: Problem {record['problem_id']}, Answer {record['answer_id']}")
                    print(f"   Error message: {e}")

    # 总结
    print("\n" + "="*80)
    print("Batch Processing Complete")
    print("="*80)
    print(f"Total records in file: {len(all_records)}")
    print(f"Filtered records: {len(filtered_records)}")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped (correct): {skipped_correct}")
    print(f"Skipped (zero): {skipped_zero}")
    print("="*80)


if __name__ == "__main__":
    main()
