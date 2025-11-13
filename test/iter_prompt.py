import os
import json
import argparse
import functools
import multiprocessing
from utils.api_inference import create_OpenAIclient, openai_completion, prompt_to_chatml
from tqdm import tqdm
import re

# --- 1. Define our error taxonomy (must be in sync with prompt.txt) ---
# These are our "Ground Truth" labels
ERROR_CATEGORIES = [
    "calculation_error",
    "counting_error",
    "context_value_error",
    "hallucination",
    "unit_conversion_error",
    "operator_error",
    "formula_confusion_error",
    "missing_step",
    "contradictory_step",
    "no_error"
]

# --- 2. Argparse ---
parser = argparse.ArgumentParser(description="Run LLM-as-Examiner Evaluation")
# parser.add_argument("--input_file", type=str, default="/home/exouser/EIC/data/generated_cases_GSM8K")
parser.add_argument("--input_file", type=str, default="/home/exouser/EIC/data/generated_cases_GSM8K/calculation_error/calculation_error_100/generated_cases_clean.jsonl")
parser.add_argument("--output_file", type=str, default="/home/exouser/EIC/gpt-4o-mini-cot.json")
parser.add_argument("--prompt_file", type=str, default="/home/exouser/EIC/prompt.txt")
parser.add_argument("--num_procs", type=int, default=2)
parser.add_argument("--model", type=str, default="gpt-4o-mini")
args = parser.parse_args()

def load_data(filepath):
    """Loads data from a .jsonl file."""
    print(f"Loading dataset from {filepath}...")
    data = []
    with open(filepath, 'r') as f:
        # Loop through each line in the file
        for line in f:
            # Skip empty lines
            if line.strip():
                try:
                    # Parse each line as its own JSON object
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line: {line}")
    
    if not data:
        raise ValueError(f"Error: No valid JSON objects found in {filepath}")
        
    print(f"Loaded {len(data)} items.")
    return data

def load_prompt(filepath):
    """Loads the prompt template."""
    print(f"Loading prompt from {filepath}...")
    with open(filepath, 'r') as f:
        return f.read()

def parse_llm_response(response_text):
    """
    Parses JSON from the LLM's response.
    This new version *requires* the JSON to be in a ```json ... ``` block,
    and it will ignore any reasoning text that comes before it.
    """
    # Search for the JSON block *anywhere* in the response
    match = re.search(r'```json\s*([\s\S]+?)\s*```', response_text)
    
    if match:
        # If the block is found, extract its content
        json_str = match.group(1)
        try:
            # Try to parse the extracted string
            return json.loads(json_str.strip())
        except json.JSONDecodeError:
            # Handle cases where the content inside the block is not valid JSON
            print(f"Warning: Found JSON block, but failed to parse: {json_str}")
            return {
                "found_error": True,
                "step_number": -1,
                "error_type": "Parsing Error"
            }
    else:
        # If no ```json ... ``` block is found at all, it's a format error
        print(f"Warning: No ```json ... ``` block found in response: {response_text}")
        return {
            "found_error": True,
            "step_number": -1,
            "error_type": "Format Error (No JSON block)"
        }

# --- 6. Core Processing Function (reverted to OpenAI) ---
def process_item(item, prompt_template, error_types_str, openai_kwargs):
    """Processes a single data item, calls the OpenAI API, and returns evaluation."""
    
    # Prepare inputs (same as before)
    problem_text = item.get("question", "")
    solution_lines = item.get("transformed_solution", "").split('\n')
    solution_steps = "\n".join(
        f"Step {i+1}: {line.strip()}" 
        for i, line in enumerate(solution_lines) 
        if line.strip()
    )
    
    # Ground Truth labels (same as before)
    gt_step = item.get("wrong_step", None)
    gt_type = item.get("wrong_type", "No Error Found")
    if gt_step is None or gt_step == 0:
        gt_step = None 
        gt_type = "No Error Found"

    # Fill the prompt
    filled_prompt = prompt_template.format_map({
        "error_type_count": len(ERROR_CATEGORIES),
        "error_types": "\n".join(f"- {cat}" for cat in ERROR_CATEGORIES),
        "problem_text": problem_text,
        "solution_steps": solution_steps
    })

    # Call OpenAI API
    try:
        chatml = prompt_to_chatml(filled_prompt)
        client = create_OpenAIclient(dict(api_key=os.getenv("OpenAI_API_KEY")))
        response_dict = openai_completion(client, chatml, openai_kwargs)
        
        llm_raw_text = response_dict["response"]
        api_cost = response_dict["cost"]
        
        llm_output = parse_llm_response(llm_raw_text)
        
        # Predicted labels
        pred_found = llm_output.get("found_error", False)
        pred_step = llm_output.get("step_number", None) if pred_found else None
        pred_type = llm_output.get("error_type", "No Error Found")

        if pred_type not in ERROR_CATEGORIES:
            pred_type = "Invalid Type" 

    except Exception as e:
        # --- 详细的错误打印 ---
        print(f"\n--- DETAILED ERROR (process_item) ---")
        print(f"Error Type: {type(e)}")
        print(f"Error Details: {e}")

        # 导入 traceback 库来打印完整的错误堆栈
        import traceback
        traceback.print_exc() 

        print(f"--- END DETAILED ERROR ---")
        # --- 详细的错误打印结束 ---

        pred_step = -1
        pred_type = "API Error"
        api_cost = 0.0
        llm_raw_text = str(e)
        
    # Compare results
    step_is_correct = (gt_step == pred_step)
    type_is_correct = (gt_type == pred_type)

    return {
        "problem_id": item.get("trace_id", "N/A"),
        "ground_truth": {"step": gt_step, "type": gt_type},
        "predicted": {"step": pred_step, "type": pred_type},
        "step_is_correct": step_is_correct,
        "type_is_correct": type_is_correct,
        "llm_raw_output": llm_raw_text,
        "cost": api_cost  # Store the cost for this item
    }

# --- 7. Main Function ---
def main():
    # 1. Load assets
    prompt_template = load_prompt(args.prompt_file)
    dataset = load_data(args.input_file)
    
    # 2. Define OpenAI Config
    OPENAI_KWARGS = {
        "model": args.model,
        "max_tokens": 1024,
        "temperature": 0.0,
        "seed": 0,
    }

    # 3. Prepare for multiprocessing
    # Pass OPENAI_KWARGS to the worker function
    worker_function = functools.partial(
        process_item,
        prompt_template=prompt_template,
        error_types_str="\n".join(f"- {cat}" for cat in ERROR_CATEGORIES),
        openai_kwargs=OPENAI_KWARGS
    )
    
    # 4. Run evaluation
    results = []
    print(f"Starting evaluation with {args.num_procs} workers...")
    with multiprocessing.Pool(args.num_procs) as pool:
        for result in tqdm(pool.imap(worker_function, dataset), total=len(dataset)):
            results.append(result)
            
    print("Evaluation complete. Calculating metrics...")

    # 5. Calculate accuracy and cost
    if not results:
        print("Error: No results to process.")
        return

    total_items = len(results)
    step_correct_count = sum(1 for res in results if res["step_is_correct"])
    type_correct_count = sum(1 for res in results if res["type_is_correct"])
    total_cost = sum(res.get("cost", 0.0) for res in results) # Sum the cost
    
    step_accuracy = (step_correct_count / total_items) * 100
    type_accuracy = (type_correct_count / total_items) * 100

    # 6. Print report
    print("\n--- Evaluation Report ---")
    print(f"Total Items: {total_items}")
    print(f"Total Cost: ${total_cost:.6f}")
    print(f"Error Location (Step) Accuracy: {step_accuracy:.2f}% ({step_correct_count}/{total_items})")
    print(f"Error Classification (Type) Accuracy: {type_accuracy:.2f}% ({type_correct_count}/{total_items})")
    print("-------------------------\n")

    # 7. Save detailed results
    print(f"Saving detailed results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()