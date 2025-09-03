# BLIP-Tool

A comprehensive toolkit for image analysis using BLIP-2 (Bootstrapped Language-Image Pre-training) model. This repository provides tools for image captioning, visual question answering, and advanced image composition/color analysis.

## Features

- **Image Captioning**: Generate natural language descriptions of images
- **Visual Question Answering (VQA)**: Ask questions about images and get AI-powered answers
- **Composition Analysis**: Analyze visual arrangement, balance, and compositional techniques
- **Color Tone Analysis**: Evaluate color palettes, mood, and emotional impact
- **Multi-language Support**: Available in English and Chinese

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mr0bean/BLIP-tool.git
cd BLIP-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Image Analysis

Run general-purpose image analysis with captioning and VQA:

```bash
python run_blip2.py --image path/to/image.jpg --mode both
```

Options:
- `--image`: Path to local image or image URL
- `--mode`: Analysis mode (`caption`, `vqa`, or `both`)
- `--question`: Custom question for VQA mode

### Advanced Composition and Color Analysis

For detailed artistic and compositional analysis:

```bash
python analyze_image.py --image path/to/image.jpg
```

This generates a comprehensive report including:
- Overall image description
- Composition analysis (visual layout, techniques, focal points)
- Color tone analysis (palette, mood, harmony, emotional impact)

### Chinese Version

For Chinese language analysis:

```bash
python analyze_image_cn.py --image path/to/image.jpg
```

## Model Information

This toolkit uses the **Salesforce/blip2-opt-2.7b** model from Hugging Face, which combines:
- Vision Transformer for image encoding
- Q-Former for vision-language alignment
- OPT-2.7B language model for text generation

## Project Structure

```
BLIP-tool/
├── run_blip2.py           # Main script for captioning and VQA
├── analyze_image.py       # Advanced composition and color analysis
├── analyze_image_cn.py    # Chinese version of analysis tool
├── test_mps.py           # Apple Silicon MPS compatibility test
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Performance Notes

- **GPU (CUDA)**: Recommended for best performance, uses FP16 for efficiency
- **CPU**: Fully supported, uses FP32
- **Apple Silicon (MPS)**: Experimental support, CPU fallback available

## Examples

### Image Captioning
```bash
python run_blip2.py --image demo.jpg --mode caption
```

### Visual Question Answering
```bash
python run_blip2.py --image demo.jpg --mode vqa --question "What is the main subject?"
```

### Composition Analysis
```bash
python analyze_image.py --image landscape.jpg
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Salesforce Research](https://github.com/salesforce/LAVIS) for BLIP-2 model
- [Hugging Face](https://huggingface.co/) for model hosting and transformers library

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{blip_tool_2024,
  author = {Mr0bean},
  title = {BLIP-Tool: Image Analysis Toolkit using BLIP-2},
  year = {2024},
  url = {https://github.com/Mr0bean/BLIP-tool}
}
```