# Solution Extraction Guide

## 概述

这些脚本用于从 solution 图片（PNG格式）中提取数学解答文本，并保存为 JSONL 格式。

## 文件说明

| 文件 | 用途 |
|------|------|
| `extract_solutions.py` | 提取所有15个solution图片的文本 |
| `extract_solutions_test.py` | 测试版本，只提取第一个图片 |
| `solutions.jsonl` | 输出文件（包含所有提取的文本） |

## 使用方法

### 1. 确保环境变量已设置

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

或使用 `.env` 文件（推荐）：

```bash
cd /Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/test
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

### 2. 测试提取（推荐先执行）

```bash
cd /Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/test
python extract_solutions_test.py
```

这将：
- 只处理第一个图片 (`p_1.png`)
- 显示提取的完整文本
- 保存到 `solutions_test.jsonl`
- 验证 API 配置是否正确

### 3. 提取所有 solution

如果测试成功，运行：

```bash
python extract_solutions.py
```

这将：
- 处理所有15个图片文件
- 实时保存结果（防止中断丢失数据）
- 生成 `solutions.jsonl` 文件

## 输出格式

JSONL 文件格式（每行一个 JSON 对象）：

```json
{
  "problem_id": 1,
  "solution_text": "Complete solution text with LaTeX notation...",
  "source_file": "p_1.png"
}
```

## 使用的模型

- **模型**: `openai/gpt-4o`
- **原因**: 支持视觉输入，能够识别图片中的文字和数学符号
- **温度**: 0.1（低温度以提高准确性）
- **最大 tokens**: 16000

## 处理时间估算

- 单个图片: 约 10-30 秒
- 15个图片: 约 3-8 分钟
- 实际时间取决于图片复杂度和API响应速度

## 注意事项

### 1. API 成本

- 使用视觉模型（gpt-4o）成本较高
- 建议先运行测试版本验证效果
- 每个图片大约消耗 1000-3000 tokens

### 2. 图片要求

- 格式: PNG
- 命名: `p_{problem_id}.png`（例如 `p_1.png`, `p_2.png`）
- 清晰度: 越清晰提取效果越好

### 3. 文本质量

- LaTeX 公式会尽可能保留格式
- 手写内容可能识别不准确
- 复杂的数学符号可能需要人工校对

## 故障排除

### 问题: API 认证失败

```
ERROR: OPENROUTER_API_KEY environment variable is not set!
```

**解决方案**: 设置环境变量或创建 `.env` 文件

### 问题: 找不到图片文件

```
No solution files found!
```

**解决方案**:
- 检查路径是否正确
- 确认图片命名格式为 `p_*.png`

### 问题: 提取内容不完整

**解决方案**:
- 检查图片是否清晰
- 可以修改 prompt 增加更详细的指示
- 尝试增加 `max_tokens` 参数

### 问题: API 超时

**解决方案**:
- 增加 `timeout` 参数（默认1200秒）
- 检查网络连接
- 图片文件过大时考虑压缩

## 后续处理

提取完成后，可以：

### 1. 查看提取的文本

```python
import json

with open('solutions.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        print(f"Problem {data['problem_id']}:")
        print(data['solution_text'][:200] + "...\n")
```

### 2. 验证提取质量

```python
import json

with open('solutions.jsonl', 'r', encoding='utf-8') as f:
    solutions = [json.loads(line) for line in f]

print(f"Total solutions: {len(solutions)}")
for sol in solutions:
    print(f"Problem {sol['problem_id']}: {len(sol['solution_text'])} characters")
```

### 3. 合并到主数据集

可以将提取的 solution 添加到 `qwen-4b_aime2025.jsonl` 中：

```python
import json

# 读取 solutions
solutions = {}
with open('solutions.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        solutions[data['problem_id']] = data['solution_text']

# 读取并更新主数据集
updated_records = []
with open('qwen-4b_aime2025.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        problem_id = record['problem_id']
        if problem_id in solutions:
            record['reference_solution'] = solutions[problem_id]
        updated_records.append(record)

# 保存更新后的数据
with open('qwen-4b_aime2025_with_solutions.jsonl', 'w', encoding='utf-8') as f:
    for record in updated_records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
```

## 示例输出

```json
{
  "problem_id": 1,
  "solution_text": "To solve this problem, we need to find all integer bases $b > 9$ for which $17_b$ divides $97_b$.\n\nStep 1: Convert to base 10\n$17_b = 1 \\cdot b + 7 = b + 7$\n$97_b = 9 \\cdot b + 7 = 9b + 7$\n\nStep 2: Set up divisibility condition\nWe need $(b + 7) | (9b + 7)$\n\n...",
  "source_file": "p_1.png"
}
```

## 高级配置

### 修改提取精度

编辑脚本中的 `temperature` 参数：

```python
temperature=0.1,  # 降低以提高准确性（0-1之间）
```

### 修改 prompt

可以在脚本中修改 `prompt` 变量以改进提取效果：

```python
prompt = """Custom prompt here...
- Add specific instructions
- Mention specific notation to preserve
- Add examples if needed
"""
```

### 使用不同模型

如果需要使用其他模型：

```python
model="anthropic/claude-3.5-sonnet",  # 或其他支持视觉的模型
```

## 性能优化

### 并行处理

如果需要更快处理，可以修改脚本支持并行：

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(extract_text_from_image, f, pid)
               for f, pid in files]
```

**注意**: 并行处理会增加 API 调用速率，可能触发限流。

## 质量检查

提取完成后建议进行质量检查：

1. 检查字符数是否合理（不应太短或太长）
2. 验证 LaTeX 公式是否正确
3. 对比原图片确认关键步骤是否完整
4. 检查特殊符号是否正确识别

## 相关文件

- 主数据集: `qwen-4b_aime2025.jsonl`
- Solution 图片目录: `solution/`
- 输出文件: `solutions.jsonl`
- 环境配置: `.env`
