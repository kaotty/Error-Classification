#!/usr/bin/env python3
"""
重新提取 predicted_answer 从 reasoning_trace 中的 \boxed{} 内容
规则:
1. 提取 \boxed{answer} 中 \boxed{} 内部的整数
2. 如果没有匹配到 \boxed{}, 则 predicted_answer=0
"""

import json
import re


def extract_boxed_answer(text):
    """
    从文本中提取 \boxed{} 内的整数

    Args:
        text: 包含 \boxed{} 的文本

    Returns:
        提取到的整数,如果没有找到则返回 0
    """
    # 查找所有 \boxed{...} 模式
    # 使用正则表达式匹配 \boxed{内容}
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)

    if not matches:
        return 0

    # 取最后一个匹配(通常最后一个是最终答案)
    last_match = matches[-1]

    # 从匹配的内容中提取整数
    # 查找所有数字(包括负数)
    numbers = re.findall(r'-?\d+', last_match)

    if numbers:
        # 返回第一个数字(转换为整数)
        return int(numbers[0])
    else:
        return 0


def main():
    input_file = '/Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/test/qwen-4b_aime2025.jsonl'
    output_file = '/Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/test/qwen-4b_aime2025_updated.jsonl'

    processed_count = 0
    updated_count = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            # 解析JSON
            data = json.loads(line)

            # 提取 reasoning_trace
            reasoning_trace = data.get('reasoning_trace', '')

            # 提取答案
            new_predicted_answer = extract_boxed_answer(reasoning_trace)

            # 记录是否更新
            old_predicted_answer = data.get('predicted_answer')
            if old_predicted_answer != new_predicted_answer:
                updated_count += 1

            # 更新 predicted_answer
            data['predicted_answer'] = new_predicted_answer

            # 写入新文件
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')

            processed_count += 1

            # 每处理100行打印一次进度
            if processed_count % 100 == 0:
                print(f"已处理 {processed_count} 行...")

    print(f"\n处理完成!")
    print(f"总共处理: {processed_count} 行")
    print(f"更新了: {updated_count} 行")
    print(f"输出文件: {output_file}")


if __name__ == '__main__':
    main()
