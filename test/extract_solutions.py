"""
从 solution 图片中提取文本内容，保存到 JSONL 文件
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
OUTPUT_FILE = "/Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/test/solutions.jsonl"

# 配置 OpenRouter API
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("="*80)
    print("ERROR: OPENROUTER_API_KEY environment variable is not set!")
    print("="*80)
    print("\nPlease set your API key:")
    print('   export OPENROUTER_API_KEY="your_api_key_here"')
    print("\n" + "="*80)
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
    """
    使用 OpenRouter API 从图片中提取文本
    """
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

    try:
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
            temperature=0.1,  # 低温度以获得更准确的提取
        )

        extracted_text = response.choices[0].message.content
        print(f"✓ Successfully extracted {len(extracted_text)} characters")
        return extracted_text

    except Exception as e:
        print(f"✗ Error extracting text: {e}")
        return None


def main():
    """主函数"""
    print("="*80)
    print("Solution Text Extraction")
    print("="*80)
    print(f"Solution directory: {SOLUTION_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print("="*80)

    # 获取所有 PNG 文件
    solution_files = sorted(Path(SOLUTION_DIR).glob("p_*.png"))

    if not solution_files:
        print("No solution files found!")
        return

    print(f"\nFound {len(solution_files)} solution files")

    # 处理每个文件
    results = []
    success_count = 0
    error_count = 0

    for file_path in solution_files:
        # 从文件名提取问题ID
        # 例如: p_1.png -> problem_id = 1
        filename = file_path.stem  # p_1
        problem_id = int(filename.split('_')[1])

        try:
            # 提取文本
            extracted_text = extract_text_from_image(str(file_path), problem_id)

            if extracted_text:
                result = {
                    "problem_id": problem_id,
                    "solution_text": extracted_text,
                    "source_file": file_path.name
                }
                results.append(result)
                success_count += 1

                # 实时保存结果（防止中断丢失数据）
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')

                print(f"✓ Saved to {OUTPUT_FILE}")
            else:
                error_count += 1

        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {e}")
            error_count += 1
            continue

        print("-" * 80)

    # 最终汇总
    print("\n" + "="*80)
    print("Extraction Complete")
    print("="*80)
    print(f"Total files: {len(solution_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Output file: {OUTPUT_FILE}")
    print("="*80)

    # 显示预览
    if results:
        print("\nPreview of first solution:")
        print("-" * 80)
        first_result = results[0]
        print(f"Problem ID: {first_result['problem_id']}")
        print(f"Text length: {len(first_result['solution_text'])} characters")
        print(f"First 200 characters:")
        print(first_result['solution_text'][:200] + "...")
        print("-" * 80)


if __name__ == "__main__":
    main()
