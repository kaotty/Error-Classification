"""
测试版本 - 只提取第一个 solution 图片的文本
"""
import json
import os
import base64
from pathlib import Path
from openai import OpenAI

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 配置
SOLUTION_DIR = "/Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/test/solution"
OUTPUT_FILE = "/Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/test/solutions_test.jsonl"

# 配置 OpenRouter API
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("ERROR: OPENROUTER_API_KEY environment variable is not set!")
    import sys
    sys.exit(1)

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    timeout=1200.0
)


def encode_image_to_base64(image_path):
    """将图片编码为 base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_text_from_image(image_path, problem_id):
    """使用 OpenRouter API 从图片中提取文本"""
    print(f"\nProcessing problem {problem_id}...")
    print(f"Image: {image_path}")

    # 编码图片
    base64_image = encode_image_to_base64(image_path)

    # 构建 prompt
    prompt = """Please extract ALL the text content from this mathematical solution image.

Requirements:
1. Extract all text exactly as it appears
2. Preserve mathematical notation (use LaTeX format where appropriate)
3. Maintain the structure and formatting
4. Include all steps, equations, and explanations
5. If there are multiple solutions or cases, include all of them

Please provide the complete text content in a clear, readable format."""

    print("Calling API...")

    # 调用 API
    response = openrouter_client.chat.completions.create(
        model="openai/gpt-4o",  # 使用支持视觉的模型
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=16000,
        temperature=0.1,
    )

    extracted_text = response.choices[0].message.content
    print(f"✓ Successfully extracted {len(extracted_text)} characters")
    return extracted_text


def main():
    """主函数 - 只处理第一个文件"""
    print("="*80)
    print("Solution Text Extraction - TEST MODE")
    print("="*80)

    # 获取第一个 PNG 文件
    solution_files = sorted(Path(SOLUTION_DIR).glob("p_*.png"))

    if not solution_files:
        print("No solution files found!")
        return

    # 只处理第一个
    file_path = solution_files[0]
    filename = file_path.stem
    problem_id = int(filename.split('_')[1])

    print(f"\nProcessing file: {file_path.name}")
    print(f"Problem ID: {problem_id}")

    try:
        # 提取文本
        extracted_text = extract_text_from_image(str(file_path), problem_id)

        if extracted_text:
            result = {
                "problem_id": problem_id,
                "solution_text": extracted_text,
                "source_file": file_path.name
            }

            # 保存结果
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

            print(f"\n✓ Saved to {OUTPUT_FILE}")

            # 显示提取的内容
            print("\n" + "="*80)
            print("Extracted Text:")
            print("="*80)
            print(extracted_text)
            print("="*80)

            print(f"\nText length: {len(extracted_text)} characters")
            print("\n✓ Test successful! You can now run extract_solutions.py for all files.")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
