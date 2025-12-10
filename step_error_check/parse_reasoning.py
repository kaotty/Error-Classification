"""
批量分析脚本 - 对所有answer应用三步分析（segment, merge, check）
"""

"""
需要配置如下路径参数：
JSONL_FILE: 输入文件路径 （格式与Error-Classification/test/claude/aime2025_claude_3_7_30_10_wrong_answers_refined.jsonl相同）
SOLUTIONS_FILE: 参考解答文件路径 
GROUND_TRUTH_FILE: 标准答案文件路径
BASE_OUTPUT_DIR: 输出目录路径

api_key = os.getenv("OPENROUTER_API_KEY") 需要在这里填openrouter的api key
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

api_key = os.getenv("OPENROUTER_API_KEY")

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    timeout=1200.0
)


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


def step1_segment(record, output_file):
    """
    Step 1: 分割和标注推理步骤
    """
    print(f"\n{'='*80}")
    print(f"Step 1: Segmenting and Labeling")
    print(f"{'='*80}")

    # 先检查 segment 文件是否已存在
    if os.path.exists(output_file):
        print(f"Segment file exists, loading from: {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                segment_data = json.load(f)
            print(f"✓ Loaded existing segment data")
            print(f"Total steps: {segment_data.get('total_steps', 0)}")
            print(f"Processed steps: {segment_data.get('processed_steps', 0)}")
            return segment_data
        except Exception as e:
            print(f"Warning: Error loading segment file: {e}")
            print("Will regenerate segment data...")

    problem_id = record['problem_id']
    answer_id = record['answer_id']
    reasoning = record['reasoning'] + '\n\n' + record['answer']
    problem = record['problem_text']

    # 分割推理轨迹
    step_trace = reasoning.split('\n\n')
    # parse_result = ['']
    parse_result = step_trace

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
        "model_id": record.get("model", ""),
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

    # 先检查 merge 文件是否已存在
    if os.path.exists(output_file):
        print(f"Merge file exists, loading from: {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                merge_data = json.load(f)
            print(f"✓ Loaded existing merge data")
            print(f"Original steps: {merge_data['metadata']['original_steps_count']}")
            print(f"Merged steps: {merge_data['metadata']['merged_steps_count']}")
            print(f"Reduction rate: {merge_data['metadata']['reduction_rate']}")
            return merge_data
        except Exception as e:
            print(f"Warning: Error loading merge file: {e}")
            print("Will regenerate merge data...")

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


def step3_generate_parsed_content(merge_file):
    """
    Step 3: 读取 merge 文件并生成 merged_content
    """
    print(f"\n{'='*80}")
    print(f"Step 3: Generating Parsed Content")
    print(f"{'='*80}")

    # 检查 merge 文件是否存在
    if not os.path.exists(merge_file):
        print(f"Error: Merge file not found: {merge_file}")
        return ""

    try:
        # 读取 merge 文件
        with open(merge_file, 'r', encoding='utf-8') as f:
            merge_data = json.load(f)

        # 从 merged_steps 生成 merged_content
        merged_steps_detail = merge_data.get("merged_steps", [])
        merged_content = '<parse>'.join([step["content"] for step in merged_steps_detail if step["content"].strip()])

        print(f"✓ Generated merged_content ({len(merged_content)} chars)")
        print(f"  - Merged steps: {len(merged_steps_detail)}")

        return merged_content

    except Exception as e:
        print(f"Error generating merged_content: {e}")
        return ""


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
    segment_data = step1_segment(record, segment_file)

    # Step 2: Merge
    merge_file = os.path.join(MERGE_DIR, f"merged_steps_problem_{problem_id}_answer_{answer_id}.json")
    merge_data = step2_merge(segment_data, merge_file)

    # Step 3: Generate parsed content
    merged_content = step3_generate_parsed_content(merge_file)

    # 返回 merged_content 用于添加到 JSONL 文件
    return merged_content


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
        is_coorect = record.get('is_correct', False)
        ground_truth = ground_truth_answers.get(problem_id, '')
        # print(ground_truth)
        # exit()

        if predicted == '0' or predicted == '' or predicted == "no_answer":
            skipped_zero += 1
        elif (ground_truth and predicted == ground_truth) or is_coorect:
            skipped_correct += 1
        else:
            print(ground_truth, predicted)
            filtered_records.append(record)

    print(f"Records to process: {len(filtered_records)}")
    print(f"  - Skipped (correct answer): {skipped_correct}")
    print(f"  - Skipped (answer is 0): {skipped_zero}")
    print(f"  - Total skipped: {skipped_correct + skipped_zero}")

    # exit()

    if not filtered_records:
        print("\nNo records to process!")
        return

    # 并行处理记录
    print(f"\nProcessing {len(filtered_records)} records with {MAX_WORKERS} workers...")
    print("="*80)

    success_count = 0
    error_count = 0
    completed_count = 0

    # 用于存储处理结果的字典 {record_index: merged_content}
    processed_results = {}

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
            record_num, record = future_to_record[future]
            try:
                merged_content = future.result()
                processed_results[record_num - 1] = merged_content  # 使用索引存储
                success_count += 1
            except Exception as e:
                with print_lock:
                    print(f"Error processing record {record_num}: {e}")
                processed_results[record_num - 1] = ""  # 失败时使用空字符串
                error_count += 1
            completed_count += 1

    # 将 merged_content 添加到原始记录并保存为新的 JSONL 文件
    output_jsonl = os.path.join(BASE_OUTPUT_DIR, "parsed_steps.jsonl")
    print(f"\nSaving results to: {output_jsonl}")

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for i, record in enumerate(filtered_records):
            # 添加 merged_content 字段
            record['parsed_steps'] = processed_results.get(i, "")
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(filtered_records)} records to: {output_jsonl}")

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
