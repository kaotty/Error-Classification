import os
import re
import json
import time
from typing import List, Dict, Any
from google import genai
from google.genai import types
from openai import OpenAI
from tqdm import tqdm

# ================= Configuration =================
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE")
deepseek_client = OpenAI(
    api_key=API_KEY, 
    base_url="https://api.deepseek.com" # 必须指定这个 Base URL
)
MODEL_NAME = "deepseek-reasoner"  # 或 gemini-3.0-pro-exp

INPUT_FILE = "testset.jsonl"
OUTPUT_FILE = "verification_results_dpsk_v1.jsonl" # 改为 .jsonl 以便追加写入

# ================= Taxonomy Definitions (不变) =================
TAXONOMY_HIERARCHY = """
[HIERARCHY]
1. ALGEBRA (Continuous & Symbolic)
   - A-1: Logical Inconsistency & Hallucination
   - A-2: Procedural & Mapping Execution Error

2. GEOMETRY (Spatial & Visual)
   - G-1: Spatial Structure & Decomposition Error
   - G-2: Unjustified Geometric Property & Relationship
   - G-3: Quantitative & Methodological Execution Error

3. COMBINATORICS & NUMBER THEORY (Discrete & Countable)
   - C-1: Incorrect Technical Derivation
   - C-2: Case Omission
   - C-3: Misidentification and Hallucination
   - C-4: Constraint Violation
"""

TAXONOMY_DEFINITIONS = """
[DEFINITIONS]
- A-1: The model introduces objectively false premises (hallucinations) that are not derived from the problem statement, or exhibits self-contradictory reasoning where a statement directly conflicts with a previous assertion within the same response. This represents a failure in the "Truthfulness" and "Consistency" of the reasoning chain.
- A-2: Errors occurring during the mechanical execution of algebraic steps or the translation between mathematical forms. This includes applying formulas incorrectly, manipulation slips (e.g., sign errors in inequalities, solving equations), or mapping failures between representations (e.g., Complex plane to Cartesian coordinates), assuming the underlying logic was otherwise correct.
- G-1: This category encompasses all errors regarding the structural construction and decomposition of the geometric figure. Its hallmark is a failure in "part-whole" logic, where the model establishes incorrect additive or subtractive relationships for angles, areas, or lengths (e.g., asserting Total Angle = Part A + Wrong Part). It also includes topological errors, such as misidentifying internal points as boundary points, confusing the order of points on a line, or creating impossible intersections. Essentially, this error implies that the model's mental "puzzle" of the figure is structurally broken, has missing pieces, or is assembled incorrectly.
- G-2: This category targets unjustified assertions of properties or relationships between geometric elements, assuming the spatial structure is otherwise perceived correctly. The model identifies the correct components but hallucinates strict geometric rules without proof, such as claiming congruence, similarity, parallelism, or perpendicularity where none exist. It also includes "specialization errors," where general figures are treated as specific ones (e.g., assuming an arbitrary triangle is isosceles). Essentially, the model imposes non-existent constraints, theorems, or attributes onto a valid geometric map.
- G-3: The model fails when translating geometric properties into mathematical frameworks (algebraic, coordinate, or trigonometric) or commits errors during the quantitative calculation of metrics (angles, lengths, areas). This includes incorrectly establishing coordinate systems, misapplying trigonometric formulas (e.g., Law of Cosines), or calculation slips within the analytic process.
- C-1: Errors in arithmetic, formula application, or algebraic manipulation within a specific step.
- C-2: The model adopts a correct strategy but fails to list all possible cases or boundary values.
- C-3: The model misinterprets the fundamental combinatorial structure of the problem or introduces phantom constraints that incorrectly narrow the solution space. Crucially, this category also encompasses baseless assertions, where the model fabricates premises, intermediate values, or conclusions without any derivation or grounding in the preceding context. This includes instances where specific numbers or rules are "hallucinated" into existence, breaking the logical chain with unfounded claims.
- C-4: The model ignores the constraint or considers cases that violates the constraint, leading to double-counting or treating dependent events as independent.
"""

# ================= Logic Functions =================

def parse_steps(response_text: str) -> List[str]:
    raw_steps = response_text.split('\n')
    steps = [s.strip() for s in raw_steps if s.strip()]
    return steps

def build_hierarchical_prompt(problem_text, previous_steps, current_step):
    history_text = ""
    if not previous_steps:
        history_text = "(This is the first step)"
    else:
        for idx, step in enumerate(previous_steps):
            history_text += f"Step {idx+1}: {step}\n"

    prompt = f"""
You are an expert Math Error Verifier. verify the [CURRENT STEP] based on the [PROBLEM] and [PREVIOUS STEPS].

{TAXONOMY_HIERARCHY}

{TAXONOMY_DEFINITIONS}

--------------------------------------------------
[PROBLEM]
{problem_text}

[PREVIOUS STEPS (Verified Correct)]
{history_text}

[CURRENT STEP TO VERIFY]
{current_step}

--------------------------------------------------
INSTRUCTIONS:
1. **Analyze Domain**: First, determine which domain the problem belongs to. The three domains are Algebra, Geometry, and Combinatorics/Number Theory.
2. **Verify Logic**: Check if the derivation in [CURRENT STEP] is logically valid and mathematically correct.
3. **Determine Status**:
   - If CORRECT, output status "CORRECT".
   - If INCORRECT, based on the domain selected in Stage 1, you must assign the specific error type. You are STRICTLY PROHIBITED from selecting an error type that does not belong to the chosen Major Category.
4. **Format**: Output strictly in JSON format.

JSON Output Example:
{{
  "thought": "The step assumes rows are independent, violating Sudoku rules.",
  "domain": "COMBINATORICS",
  "status": "INCORRECT",
  "error_type": "C-4"
}}
"""
    return prompt

def robust_parse_response(text: str) -> Dict[str, Any]:
    """
    【新增函数】使用正则暴力提取字段，完全绕过 json.loads。
    解决 LaTeX (如 \frac, \parallel) 导致的 Invalid \escape 报错。
    """
    if not text:
        return {"status": "API_ERROR", "thought": "Empty Input"}

    result = {
        "status": "CORRECT", 
        "error_type": None,
        "thought": None,
        "domain": None
    }
    
    # --- 核心修改：优先提取 Markdown JSON 代码块 ---
    # 寻找 ```json { ... } ``` 结构
    json_block_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    
    target_text = text # 默认在全文找
    if json_block_match:
        target_text = json_block_match.group(1) # 如果找到了代码块，只在代码块里找
        
        # 尝试直接解析标准 JSON
        try:
            data = json.loads(target_text)
            # 映射字段，防止模型用的键名不一样
            result["status"] = data.get("status", "CORRECT").upper()
            result["error_type"] = data.get("error_type")
            result["thought"] = data.get("thought")
            result["domain"] = data.get("domain")
            return result
        except:
            pass # 如果标准解析失败，回退到下面的正则提取

    # --- 回退方案：在 target_text 中用正则暴力提取 ---
    # 1. 提取 Status
    status_match = re.search(r'"status"\s*:\s*"(\w+)"', target_text, re.IGNORECASE)
    if status_match:
        result["status"] = status_match.group(1).upper()
    
    # 2. 提取 Error Type
    type_match = re.search(r'"error_type"\s*:\s*(?:")?([A-Z]-\d+|null)(?:")?', target_text, re.IGNORECASE)
    if type_match and type_match.group(1).lower() != 'null':
        result["error_type"] = type_match.group(1)

    # 3. 提取 Thought
    thought_match = re.search(r'"thought(?:_process)?"\s*:\s*"(.*?)(?:"\s*,\s*"|\s*})', target_text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1)
    
    # 4. 如果还是空的，把 Analysis Block 的前 100 个字拿来当 thought，方便调试
    if not result["thought"] and not json_block_match:
        # 既然没有 JSON 块，那整个 text 可能就是分析过程
        result["thought"] = "Parsed from raw text: " + text[:200].replace("\n", " ")

    return result

def verify_single_step(client, problem, previous_steps, current_step):
    prompt = build_hierarchical_prompt(problem, previous_steps, current_step)
    
    # ================= BRANCH 1: DEEPSEEK =================
    if "deepseek" in MODEL_NAME.lower():
        try:
            response = deepseek_client.chat.completions.create(
                model=MODEL_NAME, # 例如 "deepseek-reasoner"
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # 保持结果确定性
                stream=False
            )
            
            # DeepSeek 的返回结构解析
            raw_content = response.choices[0].message.content
            
            # 检查空响应
            if not raw_content:
                return {"status": "API_ERROR", "thought": "DeepSeek returned empty content"}
            
            # 复用你写好的正则解析函数
            return robust_parse_response(raw_content)

        except Exception as e:
            print(f"[DEEPSEEK API ERROR] {str(e)}")
            return {"status": "API_ERROR", "error_message": str(e)}

    # ================= BRANCH 2: GOOGLE GEMINI (原有逻辑) =================
    else:
        # --- Google 的 Safety Settings ---
        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        ]

        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=8192, # Google Flash 支持长输出
            safety_settings=safety_settings
        )
        
        try:
            # 注意：这里的 client 是从 main 传进来的 Google client
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config
            )
            
            if not response.text:
                reason = "UNKNOWN"
                if response.candidates:
                    reason = str(response.candidates[0].finish_reason)
                return {"status": "API_ERROR", "thought": f"Blocked/Empty: {reason}"}

            return robust_parse_response(response.text)

        except Exception as e:
            print(f"[GOOGLE API ERROR] {str(e)}") 
            return {"status": "API_ERROR", "error_message": str(e)}

# ================= Main Execution =================

def main():
    if "deepseek" not in MODEL_NAME.lower():
        if "YOUR_API_KEY" in os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY"): # 假设你用环境变量或直接填
             # 这里只是简单检查，你原本的逻辑可能直接写在代码里了，保持你原有的 Key 检查即可
             pass
        # 初始化 Google Client
        client = genai.Client(api_key="你的Google Key") # 填入你的 Google Key
    else:
        # 如果是 DeepSeek，传个 None 给 client 占位即可，反正用不到
        client = None
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 加载数据
    dataset = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    
    # # ================= 【新增代码 START】: 过滤指定 ID =================
    # # 1. 定义你想跑的散点 ID
    # target_ids = {11, 12, 19, 26, 32, 37, 41, 47}
    
    # # 2. 把 49-80 这个范围加进去 (注意：range(49, 81) 代表 [49, 80])
    # target_ids.update(range(49, 81))
    
    # # 3. 执行过滤
    # # 假设你的 json 里 "id" 是整数类型。如果是字符串，请用 str(item.get("id"))
    # original_count = len(dataset)
    # dataset = [item for item in dataset if item.get("id") in target_ids]
    
    # print(f"Filter applied: Reduced from {original_count} to {len(dataset)} problems.")
    # # ================= 【新增代码 END】 =================

    print(f"Loaded {len(dataset)} problems.")
    print(f"Results will be streamed to: {OUTPUT_FILE}")
    
    # 清空或初始化输出文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass # Just create/clear the file

    stats = {
        "total": 0,
        "step_acc": 0, # Step Localization Correct
        "type_acc": 0, # Type Classification Correct
        "fp": 0,       # False Positives
        "fn": 0        # False Negatives
    }

    # 使用 tqdm 显示进度条
    pbar = tqdm(dataset)
    
    for item in pbar:
        problem_id = item.get("id")
        problem_text = item.get("problem_text")
        response_text = item.get("response_text", "")
        gt_step_idx = item.get("step") 
        gt_type = item.get("type")     

        steps = parse_steps(response_text)
        
        # 结果容器
        problem_result = {
            "id": problem_id,
            "ground_truth": {"step": gt_step_idx, "type": gt_type},
            "prediction": {"step": None, "type": None},
            "is_step_correct": False,
            "is_type_correct": False,
            "trace": []
        }

        limit = len(steps)
        if gt_step_idx is not None:
             limit = min(gt_step_idx, len(steps))

        predicted_error_step = None
        predicted_error_type = None

        for i in range(limit):
            current_step_content = steps[i]
            previous_steps = steps[:i]
            
            api_res = verify_single_step(client, problem_text, previous_steps, current_step_content)
            
            status = api_res.get("status", "CORRECT").upper()
            error_type = api_res.get("error_type")

            problem_result["trace"].append({
                "step_index": i + 1,
                "content": current_step_content[:60] + "...",
                "model_status": status,
                "model_type": error_type,
                "thought": api_res.get("thought")
            })

            if status == "INCORRECT":
                predicted_error_step = i + 1
                predicted_error_type = error_type
                break 
            time.sleep(4)

        # === 结果判定 ===
        problem_result["prediction"]["step"] = predicted_error_step
        problem_result["prediction"]["type"] = predicted_error_type

        # 判定 Step
        if gt_step_idx is None and predicted_error_step is None:
            problem_result["is_step_correct"] = True
            problem_result["is_type_correct"] = True
            stats["step_acc"] += 1
            stats["type_acc"] += 1
        elif gt_step_idx == predicted_error_step:
            problem_result["is_step_correct"] = True
            stats["step_acc"] += 1
            if gt_type == predicted_error_type:
                problem_result["is_type_correct"] = True
                stats["type_acc"] += 1
        else:
            if predicted_error_step is not None and (gt_step_idx is None or predicted_error_step < gt_step_idx):
                stats["fp"] += 1
            else:
                stats["fn"] += 1

        stats["total"] += 1

        # === 实时写入文件 ===
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(problem_result, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        # === 更新进度条描述 (实时统计) ===
        curr_step_acc = stats["step_acc"] / stats["total"]
        curr_type_acc = stats["type_acc"] / stats["total"]
        pbar.set_description(f"Step Acc: {curr_step_acc:.1%} | Type Acc: {curr_type_acc:.1%}")

    # ================= 生成最终报告 =================
    
    # 防止除零错误
    final_total = stats['total'] if stats['total'] > 0 else 1
    step_acc_pct = stats['step_acc'] / final_total
    type_acc_pct = stats['type_acc'] / final_total
    
    # 构造报告字符串
    report_str = (
        "\n"
        "==================================================\n"
        "FINAL REPORT\n"
        "==================================================\n"
        f"Total Problems: {stats['total']}\n"
        f"Step Loc Accuracy: {step_acc_pct:.2%}\n"
        f"Type Cls Accuracy: {type_acc_pct:.2%}\n"
        f"False Positives: {stats['fp']}\n"
        f"False Negatives: {stats['fn']}\n"
        "==================================================\n"
    )

    # 1. 打印到终端
    print(report_str)

    # 2. 追加到日志文件末尾
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(report_str)

if __name__ == "__main__":
    main()