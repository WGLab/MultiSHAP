# MultiSHAP VQA Analysis

A Python tool for analyzing cross-modal interactions in Vision-Language models using Shapley Interaction Index. This implementation focuses on Visual Question Answering (VQA) tasks with ViLT models.

## Features

- **Cross-modal Interaction Analysis**: Quantify synergistic and suppressive interactions between image patches and text tokens
- **Multiple Sampling Methods**: Choose between exact computation, Monte Carlo sampling, or stratified sampling
- **Model Agnostic**: Works with any Hugging Face ViLT model
- **Rich Visualizations**: Generate heatmaps showing interaction patterns
- **Flexible Configuration**: Comprehensive command-line interface for various experimental setups

## Installation

```bash
pip install torch transformers datasets matplotlib opencv-python pillow numpy tqdm
```

## Quick Start

### Basic Usage

```bash
# Analyze 5 samples with default settings
python multishap_vqa.py --num-samples 5 --visualize-first 3

# Save results and visualizations
python multishap_vqa.py \
    --num-samples 10 \
    --output-dir ./results \
    --save-results ./results/analysis.json
```

### Advanced Usage

```bash
# High-quality analysis with stratified sampling
python multishap_vqa.py \
    --num-samples 20 \
    --n-iccs-samples 256 \
    --visualize-first 5 \
    --output-dir ./experiments/high_quality

# Exact computation (slower but precise)
python multishap_vqa.py \
    --num-samples 3 \
    --exact \
    --output-dir ./experiments/exact

# Custom model and dataset
python multishap_vqa.py \
    --model-name "your-model-name" \
    --dataset-name "your-dataset" \
    --device cuda \
    --num-samples 15
```

## Command Line Arguments

### Core Arguments
- `--num-samples`: Number of dataset samples to analyze (default: 10)
- `--model-name`: Hugging Face model identifier (default: "dandelin/vilt-b32-finetuned-vqa")
- `--device`: Computing device - auto/cpu/cuda/mps (default: auto)

### Sampling Options
- `--n-iccs-samples`: Monte Carlo samples for ICCS computation (default: 128)
- `--exact`: Use exact Shapley computation instead of Monte Carlo
- `--no-stratified`: Use uniform sampling instead of stratified sampling

### Visualization
- `--visualize-first`: Number of samples to visualize (default: 3)
- `--visualize-average-only`: Only show average token interactions
- `--output-dir`: Directory to save visualizations

### Data Selection
- `--sample-indices`: Specific dataset indices to analyze
- `--random-seed`: Seed for reproducible sampling (default: 42)
- `--dataset-name`: Hugging Face dataset name (default: "HuggingFaceM4/VQAv2")
- `--dataset-split`: Dataset split to use (default: "validation")

### Output Options
- `--save-results`: Path to save analysis results as JSON
- `--verbose`: Enable verbose logging

## Sampling Methods

### Monte Carlo (Default)

### Stratified Sampling (Recommended)

### Exact Computation

## License

This project is licensed under the MIT License.