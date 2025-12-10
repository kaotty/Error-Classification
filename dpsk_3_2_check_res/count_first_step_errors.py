#!/usr/bin/env python3
"""
统计first incorrect step的error_type
"""
import json
import os
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib


def count_first_step_errors(check_dir):
    """
    统计指定目录中所有JSON文件的first incorrect step error_type

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
    no_error_count = 0

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

            # 查找第一个incorrect step
            first_incorrect_step = None
            for step in check_results:
                correctness = step.get('correctness', None)
                if correctness is False:
                    first_incorrect_step = step
                    break

            # 如果没有找到incorrect step，分类为'no error'
            if first_incorrect_step is None:
                error_types.append('no error')
                no_error_count += 1
                print(f"{json_file.name}: 无错误步骤 (all steps correct)")
            else:
                # 提取error_type
                error_type = first_incorrect_step.get('error_type', '')
                step_number = first_incorrect_step.get('step_number', '?')

                # 如果error_type为空，用空字符串表示
                error_types.append(error_type if error_type else '')

                if error_type:
                    error_count += 1
                    print(f"{json_file.name}: First Incorrect Step {step_number}, Error Type: '{error_type}'")
                else:
                    print(f"{json_file.name}: First Incorrect Step {step_number}, Error Type: '(空)'")

        except Exception as e:
            print(f"处理 {json_file.name} 时出错: {e}")

    # 统计
    counter = Counter(error_types)

    print(f"\n{'='*60}")
    print(f"总共处理文件数: {file_count}")
    print(f"有first incorrect step的数量: {error_count + (counter.get('', 0) or 0)}")
    print(f"无错误步骤的数量 (no error): {no_error_count}")
    print(f"{'='*60}\n")

    return counter


def print_statistics(counter):
    """
    打印统计结果

    Args:
        counter: Counter对象
    """
    print("First Incorrect Step Error Type 统计:")
    print(f"{'='*60}")

    # 按出现次数排序
    for error_type, count in counter.most_common():
        if error_type == '':
            error_type_display = '(空/未标注)'
        elif error_type == 'no error':
            error_type_display = 'no error (完全正确)'
        else:
            error_type_display = error_type

        percentage = (count / sum(counter.values())) * 100
        print(f"{error_type_display:30s}: {count:4d} ({percentage:5.2f}%)")

    print(f"{'='*60}")
    print(f"总计: {sum(counter.values())}")


def plot_error_pie_chart(counter, save_path=None):
    """
    绘制有错误的error type饼状图（排除'no error'）

    Args:
        counter: Counter对象
        save_path: 保存图片的路径，如果为None则显示图片
    """
    # 过滤掉'no error'，只保留有错误的类型
    error_counter = Counter({k: v for k, v in counter.items() if k in ['A-1', 'A-2', 'G-1', 'G-2', 'G-3', 'C-1', 'C-2', 'C-3']})

    if not error_counter:
        print("没有错误数据可以绘制")
        return

    # 设置中文字体支持
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 准备数据
    labels = []
    sizes = []
    for error_type, count in error_counter.most_common():
        if error_type == '':
            label = '(空/未标注)'
        else:
            label = error_type
        labels.append(label)
        sizes.append(count)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制饼图
    colors = plt.cm.Set3(range(len(labels)))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10}
    )

    # 美化百分比文本
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

    # 添加标题
    total_errors = sum(sizes)
    ax.set_title(f'Error Distribution - DeepSeek R1',
                 fontsize=14, fontweight='bold', pad=20)

    # 添加图例
    # ax.legend(wedges, [f'{label}: {size}' for label, size in zip(labels, sizes)],
    #           title="Error Types",
    #           loc="center left",
    #           bbox_to_anchor=(1, 0, 0.5, 1),
    #           fontsize=9)

    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n饼状图已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # 指定check目录路径
    check_dir = "/Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/dpsk_3_2_check_res/deepseek_r1_ds/check"

    print(f"开始统计: {check_dir}\n")

    # 统计
    counter = count_first_step_errors(check_dir)

    # 打印结果
    print()
    print_statistics(counter)

    # 绘制饼状图（排除'no error'）
    # 可以指定保存路径，例如：
    # save_path = os.path.join(os.path.dirname(check_dir), "first_step_errors_pie_chart.png")
    # plot_error_pie_chart(counter, save_path=save_path)

    # 或者直接显示图片
    plot_error_pie_chart(counter)
