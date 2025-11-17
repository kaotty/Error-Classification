import json
import random
import re
import time
import fire
import os
import ast
import logging
import google.generativeai as genai

def askAPI(prompt, model_name):
    generation_config = {"temperature": 0}
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"), transport="rest")
    model = genai.GenerativeModel(model_name="gemini-2.5-pro",generation_config=generation_config)
    response = model.generate_content(prompt)
    return response.text

def main(
        expected_cases: int = 5,
        model_name: str = "gemini-2.5-pro",
        selected_type: str = "VE",
        dataset: str = "GSM8K",
        rubric_filename: str = "rubric.json",
        prompt_path: str = "generation_prompt.txt"
):
    print(f"selected_type: {selected_type}")
    global type
    num_of_iteration = 0
    num_of_cases = 0
    transformed_cases = []
    transformed_cases1 = []

    save_dir = f"generated_cases_{dataset}/{model_name}/{selected_type}/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    original_cases_dir = f"./step_data_{dataset}/total.jsonl"
    with open(original_cases_dir, 'r', encoding='utf-8') as file:
        original_cases = [json.loads(line) for line in file]
        print("original_question loaded")
    
    with open(rubric_filename, 'r', encoding='utf-8') as f:
        rubric_data = json.load(f)
    if rubric_data and isinstance(rubric_data, dict):
        if "error_categories" in rubric_data:
            categories_list = rubric_data["error_categories"]
            print(f"Successfully loaded {len(categories_list)} categories.\n")
            
            for category in categories_list:
                code = category.get("code")
                if code == selected_type:
                    label = category.get("label")
                    definition = category.get("definition")
                    examples = category.get("examples") # should be a list
                    example1_str = json.dumps(examples[0], indent=2)
                    example2_str = json.dumps(examples[1], indent=2)
                    break
        else:
            print(f"Error in finding error categories.")
    else:
        print("Error in processing rubric.")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # cases_dir = f"./in_context_learning/{selected_type}/initial_cases.json"
    # with open(cases_dir, 'r', encoding="utf8") as file:
    #     initial_cases = json.load(file)
    #     print("initial cases loaded")

    # templates_dir = f"./in_context_learning/{selected_type}/in_context_learning.json"
    # with open(templates_dir, 'r', encoding="utf8") as file:
    #     template = json.load(file)[0]

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(save_dir + 'info.log', mode='w')

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    while num_of_cases < expected_cases:
        logger.info("###########################")
        logger.info(f"selected_type:{selected_type}")
        logger.info(f"iteration:{num_of_iteration}")
        logger.info(f"total cases:{num_of_cases}/{expected_cases}")
        logger.info(f"len of original_cases:{len(original_cases)}")

        num_of_iteration += 1
        # num_of_cases += 1

        start_time = time.time()
        
        original_case_chosen=random.sample(original_cases, 1)
        final_prompt = prompt_template.format(
                    label=label,
                    code=code, 
                    definition=definition,
                    Example1=example1_str,
                    Example2=example2_str,
                    question=original_case_chosen[0]['question'],
                    original_solution=original_case_chosen[0]['answer']
                )
        text = askAPI(final_prompt, model_name)
        match = re.search(r'\{(.*?)\}', text, re.DOTALL)
        if not match:
            logger.warning(f"JSON object not found.")
            continue

        json_text = "{" + match.group(1) + "}"
        parsed_json = json.loads(json_text)
        original = parsed_json.get("original_answer")
        transformed = parsed_json.get("transformed_answer")
        # print(json_text)
        # try:
        #     parsed_json = ast.literal_eval(json_text)
        # except Exception as e:
        #     logger.error(f"Fail to parse JSON.")
        #     continue
        
        if original != transformed:
            with open(os.path.join(save_dir, 'generated_cases_clean.jsonl'), 'a', encoding="utf8") as file:
                json.dump(parsed_json, file, ensure_ascii=False)
                file.write("\n")
            
            original_cases.remove(original_case_chosen[0])
            num_of_cases += 1
            logger.info(f"Successfully generated {num_of_cases}/{expected_cases}.")
        else:
            logger.warning("Getting the same answer, discarded.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info("execution time:{:.2f}s\n".format(elapsed_time))

if __name__ == "__main__":
    fire.Fire(main)
