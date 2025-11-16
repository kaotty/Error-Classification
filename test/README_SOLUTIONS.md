# Solution Extraction - Quick Start

## 快速开始

### 1. 测试提取第一个图片

```bash
cd /Users/cusgadmin/Desktop/Project/LLM/repos/Error-Classification/test
python extract_solutions_test.py
```

### 2. 提取所有15个图片

```bash
python extract_solutions.py
```

## 输出文件

- `solutions_test.jsonl` - 测试输出（只有problem 1）
- `solutions.jsonl` - 完整输出（所有15个problems）

## 文件结构

```
test/
├── solution/              # 输入：PNG图片
│   ├── p_1.png
│   ├── p_2.png
│   └── ...
├── extract_solutions.py   # 主脚本
├── extract_solutions_test.py  # 测试脚本
├── solutions.jsonl        # 输出文件
└── .env                   # API密钥配置
```

## JSONL格式

每行一个JSON对象：

```json
{"problem_id": 1, "solution_text": "...", "source_file": "p_1.png"}
{"problem_id": 2, "solution_text": "...", "source_file": "p_2.png"}
```

## 查看结果

```bash
# 查看第一条记录
head -1 solutions.jsonl | python -m json.tool

# 统计提取的文本长度
cat solutions.jsonl | while read line; do
    echo $line | python -c "import sys, json; d=json.load(sys.stdin); print(f\"Problem {d['problem_id']}: {len(d['solution_text'])} chars\")"
done
```

## 常见命令

```bash
# 检查处理了多少个文件
wc -l solutions.jsonl

# 提取特定问题的solution
cat solutions.jsonl | grep '"problem_id": 5' | python -m json.tool

# 导出为单独的文本文件
python -c "
import json
with open('solutions.jsonl') as f:
    for line in f:
        d = json.loads(line)
        with open(f'solution_p{d[\"problem_id\"]}.txt', 'w') as out:
            out.write(d['solution_text'])
"
```

## 注意事项

- 使用 gpt-4o 模型，支持视觉输入
- 每个图片处理时间: 10-30秒
- 15个图片总计约: 3-8分钟
- 结果会实时保存，可随时中断和恢复

详细说明请查看: [SOLUTION_EXTRACTION.md](SOLUTION_EXTRACTION.md)
