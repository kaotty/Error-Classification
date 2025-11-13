# Step Error Check Pipeline

This directory contains a three-stage pipeline for analyzing and checking reasoning steps in mathematical problem-solving traces.

## Overview

The pipeline processes reasoning traces through three sequential stages:

1. **Rule Segment** - Segments and labels reasoning steps with high-level functions
2. **Merge Steps** - Merges related steps according to predefined rules
3. **Step Check** - Validates correctness and identifies errors in each step

## Pipeline Workflow

### Stage 1: Rule Segment (`rule_segment.py`)

**Purpose**: Segment a reasoning trace into individual steps and classify each step with a high-level function label.

**Functionality**:
- Reads reasoning traces from JSONL files
- Splits traces using keyword-based segmentation (e.g., "Wait,", "Alternatively,", "Let me verify")
- Uses GPT model to classify each step into one of 7 high-level functions:
  1. Understanding the Problem
  2. Setting Problem Solving Strategy
  3. Execute Solving Strategy and Calculate
  4. Obtaining Intermediate Results
  5. Review Previous Steps
  6. Exploring Alternative Approach
  7. Finalize and Present the Answer

**Output**: `labeled_steps_problem_{problem_id}.json` containing labeled reasoning steps

**Configuration**:
- Set `problem_id` variable to specify which problem to process
- Configure API key for OpenRouter client
- Adjust input/output file paths as needed

### Stage 2: Merge Steps (`merge_steps.py`)

**Purpose**: Reduce redundancy by merging related reasoning steps based on predefined rules.

**Functionality**:
- Reads labeled steps from Stage 1 output
- Applies merging rules:
  - "Execute Solving Strategy and Calculate" steps merge with previous steps
  - "Setting Problem Solving Strategy" after "Exploring Alternative Approach" merges
  - Consecutive "Review Previous Steps" merge together
  - Consecutive "Obtaining Intermediate Results" merge together
- Tracks merge metadata (original indices, labels, merge reasons)

**Output**: `merged_steps_problem_{problem_id}.json` with:
- Metadata (compression statistics)
- Merged steps with detailed merge information
- Simplified step list for easy access

**Configuration**:
- Set `problem_id` to match Stage 1
- Input file should be the output from Stage 1
- Output file will be used as input for Stage 3

### Stage 3: Step Check (`step_check.py`)

**Purpose**: Validate the correctness of each merged step and identify error types.

**Functionality**:
- Reads merged steps from Stage 2 output
- Compares each step against a reference solution
- For each step, validates:
  - **Correctness**: Whether the step is mathematically correct
  - **Method Consistency**: Whether the approach aligns with reference solution
  - **Error Type**: Classifies errors into 8 categories (FE, CE, CA, CS, MS, MC, HA, VE)
  - **Dependencies**: Identifies parent steps and relationship types (Progressive/Review/Corrective)
  - **Summary**: Generates concise description for visualization

**Error Types**:
- **FE** (Formula Error): Wrong formula or misapplied theorem
- **CE** (Conceptual Error): Misunderstanding constraints/conditions
- **CA** (Calculation Error): Arithmetic or algebraic mistakes
- **CS** (Contradictory Step): Inconsistent reasoning
- **MS** (Missing Step): Skipped critical intermediate reasoning
- **MC** (Missing Case): Incomplete case analysis
- **HA** (Hallucination Error): Introducing non-existent facts
- **VE** (Verification Error): Failing to check against constraints

**Output**: `step_check_problem_{problem_id}_v1.json` containing:
- Metadata (statistics on correct/incorrect steps)
- Detailed check results for each step
- Information flow graph data

**Configuration**:
- Set `problem_id` to match previous stages
- Provide reference solution in `refer_answer` variable
- Configure model temperature (default: 0.3)

## Usage Instructions

### Prerequisites

```bash
pip install openai
```

Ensure you have:
- Valid OpenRouter API key configured
- Input data files (reasoning traces and problem datasets)
- Sufficient API quota for the selected model

### Step-by-Step Execution

#### 1. Run Rule Segment

```bash
python rule_segment.py
```

Before running:
- Set `problem_id` (line 13)
- Configure API key (line 9)
- Verify input paths (lines 15, 20)
- Adjust keywords list if needed (lines 32-88)

Output: `labeled_steps_problem_{problem_id}.json`

#### 2. Run Merge Steps

```bash
python merge_steps.py
```

Before running:
- Set `problem_id` to match Stage 1 (line 3)
- Verify input file path points to Stage 1 output (line 4)
- Check output file path (line 5)

Output: `merged_steps_problem_{problem_id}.json`

#### 3. Run Step Check

```bash
python step_check.py
```

Before running:
- Set `problem_id` to match previous stages (line 12)
- Provide reference solution in `refer_answer` (lines 46-112)
- Verify input file path points to Stage 2 output (line 13)
- Configure API key (line 7)

Output: `step_check_problem_{problem_id}_v1.json`

## File Paths Configuration

All three scripts use absolute file paths. Update the following paths according to your environment:

**rule_segment.py**:
- Line 15: Input reasoning trace path
- Line 20: Problem questions path
- Line 154: Output labeled steps path

**merge_steps.py**:
- Line 4: Input labeled steps path (from Stage 1)
- Line 5: Output merged steps path

**step_check.py**:
- Line 13: Input merged steps path (from Stage 2)
- Line 14: Output check results path

## Output Files

### Stage 1 Output (`labeled_steps_problem_{id}.json`)
```json
{
  "problem_id": 13,
  "problem": "...",
  "total_steps": 50,
  "processed_steps": 50,
  "labeled_steps": [...]
}
```

### Stage 2 Output (`merged_steps_problem_{id}.json`)
```json
{
  "metadata": {
    "problem_id": 13,
    "original_steps_count": 50,
    "merged_steps_count": 35,
    "reduction_rate": "30.00%"
  },
  "merged_steps": [...],
  "simple_steps": [...]
}
```

### Stage 3 Output (`step_check_problem_{id}_v1.json`)
```json
{
  "metadata": {
    "problem_id": 13,
    "total_steps": 35,
    "correct_steps": 28,
    "incorrect_steps": 7
  },
  "check_results": [...]
}
```

## Model Configuration

All scripts use the OpenRouter API with:
- Model: `openai/gpt-5-mini` (configured in each script)
- Timeout: 1200.0 seconds
- Temperature:
  - Stage 1 & 3: 0.8 / 0.3 (can be adjusted)
  - Stage 2: No model usage (rule-based)

## Notes

- Each stage saves results incrementally to prevent data loss
- API errors are caught and logged with partial results saved
- Streaming output provides real-time feedback during processing
- The pipeline is designed to handle long reasoning traces with multiple steps
- Ensure sufficient API quota before processing large datasets

## Troubleshooting

**API Key Issues**: Verify the API key is set correctly in `openrouter_client` initialization

**File Path Errors**: Ensure all absolute paths exist and are accessible

**JSON Parsing Errors**: Check that the model output contains properly formatted `<output>` tags

**Memory Issues**: For very long traces, consider processing in batches by adjusting the step range
