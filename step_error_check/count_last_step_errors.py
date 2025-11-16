#!/usr/bin/env python3
"""
统计last step的error_type
"""
import json
import os
from collections import Counter
from pathlib import Path


def count_last_step_errors(check_dir):
    """
    统计指定目录中所有JSON文件的last step error_type

    Args:
        check_dir: 包含检查结果JSON文件的目录路径

    Returns:
        Counter对象，包含各error_type的统计
    """
    check_path = Path(check_dir)

    if not check_path.exists():
        print(f"错误: 目录不存在: {check_dir}")
        return Counter()

    # 收集所有error_type
    error_types = []
    file_count = 0
    error_count = 0

    # 遍历所有JSON文件
    for json_file in sorted(check_path.glob("step_check_*.json")):
        file_count += 1
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 获取check_results列表
            check_results = data.get('check_results', [])

            if not check_results:
                print(f"警告: {json_file.name} 没有check_results")
                continue

            # 获取最后一个step
            last_step = check_results[-1]

            # 提取error_type
            error_type = last_step.get('error_type', '')

            # 记录信息
            step_number = last_step.get('step_number', '?')
            correctness = last_step.get('correctness', None)

            error_types.append(error_type)

            # 如果有错误，增加计数
            if error_type:
                error_count += 1
                print(f"{json_file.name}: Step {step_number}, Error Type: '{error_type}', Correctness: {correctness}")
            else:
                print(f"{json_file.name}: Step {step_number}, Error Type: '(无错误/空)', Correctness: {correctness}")
        except Exception as e:
            print(f"处理 {json_file.name} 时出错: {e}")

    # 统计
    counter = Counter(error_types)

    print(f"\n{'='*60}")
    print(f"总共处理文件数: {file_count}")
    print(f"有错误的last step数: {error_count}")
    print(f"无错误的last step数: {file_count - error_count}")
    print(f"{'='*60}\n")

    return counter


def print_statistics(counter):
    """
    打印统计结果

    Args:
        counter: Counter对象
    """
    print("Last Step Error Type 统计:")
    print(f"{'='*60}")

    # 按出现次数排序
    for error_type, count in counter.most_common():
        if error_type == '':
            error_type_display = '(无错误/空)'
        else:
            error_type_display = error_type

        percentage = (count / sum(counter.values())) * 100
        print(f"{error_type_display:20s}: {count:4d} ({percentage:5.2f}%)")

    print(f"{'='*60}")
    print(f"总计: {sum(counter.values())}")


if __name__ == "__main__":
    # 指定check目录路径
    check_dir = "/Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/step_error_check/deepseek_r1_aime2025_v4/check"

    print(f"开始统计: {check_dir}\n")

    # 统计
    counter = count_last_step_errors(check_dir)

    # 打印结果
    print()
    print_statistics(counter)
