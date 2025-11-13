import os
import json
import argparse
import functools
import multiprocessing
import re
import time
from tqdm import tqdm
from collections import Counter
import google.generativeai as genai

# Prices for Gemini 1.5 Flash (per 1 million tokens)
PRICE_INPUT_PER_1M_TOKENS = 0.35
PRICE_OUTPUT_PER_1M_TOKENS = 1.05

# Global model for worker processes
g_model = None

parser = argparse.ArgumentParser(description="Classify error descriptions and count them.")
parser.add_argument("--input_file", type=str, default="qwen-4b-annotation.json")
parser.add_argument("--output_file", type=str, default="qwen-4b-direct-classification-gemini-2.5-flash-v2.json")
parser.add_argument("--rubric_file", type=str, default="rubric.json")
parser.add_argument("--num_procs", type=int, default=1)
parser.add_argument("--model", type=str, default="gemini-2.5-flash")
args = parser.parse_args()


def init_worker(api_key, google_kwargs):
    """
    Initializer function for each worker process.
    Configures the API key and creates a single, reusable model instance.
    """
    global g_model
    try:
        print(f"[Worker {os.getpid()}] Initializing...")
        
        # --- THIS IS THE CORRECT API PATTERN ---
        genai.configure(api_key=api_key)
        
        generation_config = genai.GenerationConfig(
            max_output_tokens=google_kwargs.get("max_output_tokens"),
            temperature=google_kwargs.get("temperature"),
        )
        g_model = genai.GenerativeModel(
            model_name=google_kwargs.get("model"),
            generation_config=generation_config
        )
        # --- END CORRECT API PATTERN ---

        print(f"[Worker {os.getpid()}] Initialized successfully.")
    except Exception as e:
        print(f"[Worker {os.getpid()}] CRITICAL INIT FAILED: {e}")
        g_model = None # Ensure model is None if init fails


def load_data(filepath):
    print(f"Loading error descriptions from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Error: {filepath} must contain a JSON list (of strings).")

    if data and not isinstance(data[0], str):
        print(f"Warning: Input file does not seem to be a list of strings. Converting items to string...")
        data = [str(item) for item in data]
        
    print(f"Loaded {len(data)} descriptions.")
    return data

def load_prompt(filepath):
    print(f"Loading prompt from {filepath}...")
    with open(filepath, 'r') as f:
        return f.read()

def parse_llm_response(response_text, valid_codes):
    match = re.search(r'```json\s*([\s\S]+?)\s*```', response_text)
    
    if match:
        json_str = match.group(1)
        try:
            parsed_json = json.loads(json_str.strip())
            code = parsed_json.get("error_code")
            if code in valid_codes:
                return code, parsed_json
            else:
                return "Invalid Type", parsed_json
        except json.JSONDecodeError:
            return "Parsing Error", {"error": "Failed to parse JSON", "raw": json_str}
    else:
        return "Parsing Error", {"error": "No JSON block found", "raw": response_text}

def process_item(error_description, prompt_template, rubric, valid_codes):
    
    global g_model
    
    if g_model is None:
        return {
            "input_description": error_description,
            "classification_code": "API Error",
            "llm_json_output": {"error": "Worker initialization failed. Check API Key or model name."},
            "cost": 0.0
        }

    filled_prompt = prompt_template.format_map({
        "rubric": rubric,
        "error_description": error_description
    })
    
    try:
        response = g_model.generate_content(filled_prompt)
        
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        
        input_cost = (input_tokens / 1_000_000) * PRICE_INPUT_PER_1M_TOKENS
        output_cost = (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_1M_TOKENS
        api_cost = input_cost + output_cost

        if not response.candidates:
            pf_block_reason = "N/A"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 pf_block_reason = response.prompt_feedback.block_reason.name
            
            return {
                "input_description": error_description,
                "classification_code": "API Error (Prompt Blocked)",
                "llm_json_output": {"error": f"Prompt was blocked by API. Reason: {pf_block_reason}"},
                "cost": api_cost 
            }

        # 3. 获取第一个 candidate 并检查 finish_reason
        candidate = response.candidates[0]
        
        llm_raw_text = ""
        
        if candidate.finish_reason in [1, 2]: 
            if candidate.content.parts:
                llm_raw_text = "".join(part.text for part in candidate.content.parts)
            else:
                llm_raw_text = ""
        else: 
            safety_ratings = "N/A"
            if candidate.safety_ratings:
                 safety_ratings = [str(rating) for rating in candidate.safety_ratings]
            
            return {
                "input_description": error_description,
                "classification_code": "API Error (Response Blocked)",
                "llm_json_output": {
                    "error": "Response was blocked by API.",
                    "finish_reason": candidate.finish_reason.name,
                    "safety_ratings": safety_ratings
                },
                "cost": api_cost
            }
        
        # --- [END] 修复代码 ---

        classification_code, llm_json = parse_llm_response(llm_raw_text, valid_codes)

        return {
            "input_description": error_description,
            "classification_code": classification_code,
            "llm_json_output": llm_json,
            "cost": api_cost
        }
        
    except Exception as e:
        # 捕获其他所有 API 错误 (例如超时, ResourceExhausted 等)
        print(f"\n[Worker {os.getpid()}] API call failed: {e}")
        return {
            "input_description": error_description,
            "classification_code": "API Error",
            "llm_json_output": {"error": str(e)},
            "cost": 0.0
        }

def main():
    # Use GEMINI_API_KEY as requested
    print("Checking for GEMINI_API_KEY...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("CRITICAL ERROR: GEMINI_API_KEY environment variable is NOT SET.")
        print("Please run: export GEMINI_API_KEY=\"your_key_here\"")
        print("Aborting.")
        return
    print(f"API Key found (ending in: ...{api_key[-4:]}).")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Loading rubric from {args.rubric_file}...")
    with open(args.rubric_file, 'r') as f:
        rubric = json.load(f)
    
    rubric_string = ""
    valid_codes = []
    main_categories = []
    
    for i, category in enumerate(rubric["error_categories"]):
        rubric_string += f"  {i+1}. **{category['label']} ({category['code']})**: {category['definition']}\n"
        valid_codes.append(category['code'])
        main_categories.append(category['code'])
        
    valid_codes.extend(["Parsing Error", "API Error", "Invalid Type", "API Error (RateLimit)"])
    
    prompt_file_path = os.path.join(script_dir, "prompt_classify.txt")
    prompt_template = load_prompt(prompt_file_path)
    dataset = load_data(args.input_file)

    if not dataset:
        print("CRITICAL ERROR: Input file is empty. No data to process.")
        print("Aborting.")
        return

    GOOGLE_KWARGS = {
        "model": args.model,
        "max_output_tokens": 4096,
        "temperature": 0.0,
    }

    initializer_func = functools.partial(
        init_worker, 
        api_key=api_key,
        google_kwargs=GOOGLE_KWARGS
    )
    
    worker_function = functools.partial(
        process_item,
        prompt_template=prompt_template,
        rubric=rubric_string,
        valid_codes=valid_codes
    )
    
    results = []
    print(f"Starting classification with {args.num_procs} workers...")
    
    with multiprocessing.Pool(args.num_procs, initializer=initializer_func) as pool:
        for result in tqdm(pool.imap(worker_function, dataset), total=len(dataset)):
            results.append(result)
            
    print("Classification complete. Counting results...")

    if not results:
        print("Error: No results to process.")
        return

    total_items = len(results)
    total_cost = sum(res.get("cost", 0.0) for res in results)
    
    classification_counts = Counter(res["classification_code"] for res in results)

    print("\n--- Classification Report ---")
    print(f"Total Items Processed: {total_items}")
    print(f"Total Cost: ${total_cost:.6f}")
    print("\n--- Category Counts ---")
    
    for category_code in main_categories:
        count = classification_counts.get(category_code, 0)
        print(f"  {category_code.ljust(4)}: {count} ({count / total_items:.2%})")

    print("\n--- Other Counts ---")
    for category_code, count in classification_counts.items():
        if category_code not in main_categories:
            print(f"  {category_code.ljust(15)}: {count} ({count / total_items:.2%})")
    print("-------------------------\n")

    print(f"Saving detailed classification results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()