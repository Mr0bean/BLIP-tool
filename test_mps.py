#!/usr/bin/env python3
"""
BLIP2 图像分析脚本 - 专注于构图和色调分析
"""
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import argparse
import os

def setup_device():
    """设置运行设备"""
    if torch.cuda.is_available():
        device = "cuda"
        print("使用 NVIDIA GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = "cpu"
        print("使用 CPU (MPS 可能有兼容性问题)")
    else:
        device = "cpu"
        print("使用 CPU")
    return device

def load_model(device):
    """加载BLIP2模型"""
    print("\n正在加载 BLIP2 模型...")
    
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    if device == "cuda":
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            low_cpu_mem_usage=True
        ).to(device)
    
    print("✅ 模型加载完成！")
    return processor, model

def load_image(image_path):
    """加载本地图片"""
    if os.path.exists(image_path):
        print(f"从本地加载图片: {image_path}")
        image = Image.open(image_path).convert('RGB')
    else:
        raise ValueError(f"无法找到图片: {image_path}")
    return image

def generate_caption(processor, model, image, device):
    """生成图片的总体描述"""
    inputs = processor(image, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=5,
        temperature=0.8,
        do_sample=True
    )
    
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption

def analyze_with_prompt(processor, model, image, prompt, device, max_tokens=50):
    """使用特定提示词分析图片"""
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=4,
        temperature=0.7,
        do_sample=True
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
    else:
        answer = generated_text.strip()
    
    return answer

def analyze_composition(processor, model, image, device):
    """分析图片构图"""
    composition_prompts = [
        "Question: Describe the composition of this image in terms of visual arrangement and balance. Answer:",
        "Question: What compositional technique is used in this photograph? Answer:",
        "Question: How are the elements arranged in this image? Answer:",
        "Question: Describe the rule of thirds, symmetry, or leading lines in this image. Answer:",
        "Question: What is the focal point and how does the composition guide the viewer's eye? Answer:"
    ]
    
    results = []
    for prompt in composition_prompts:
        answer = analyze_with_prompt(processor, model, image, prompt, device, max_tokens=80)
        results.append(answer)
    
    return results

def analyze_color_tone(processor, model, image, device):
    """分析图片色调"""
    color_prompts = [
        "Question: What are the dominant colors and color palette in this image? Answer:",
        "Question: Describe the mood created by the color tones (warm, cool, neutral)? Answer:",
        "Question: What time of day does the lighting and color suggest? Answer:",
        "Question: How would you describe the color harmony and contrast in this image? Answer:",
        "Question: What emotions do the colors in this image evoke? Answer:"
    ]
    
    results = []
    for prompt in color_prompts:
        answer = analyze_with_prompt(processor, model, image, prompt, device, max_tokens=60)
        results.append(answer)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='BLIP2 图像构图与色调分析')
    parser.add_argument(
        '--image', 
        type=str, 
        default='MinnesotaWaters.jpg',
        help='图片路径'
    )
    
    args = parser.parse_args()
    
    device = setup_device()
    processor, model = load_model(device)
    image = load_image(args.image)
    
    print("\n" + "="*80)
    print("📸 图像分析报告")
    print("="*80)
    
    print("\n📝 【总体描述】")
    print("-"*60)
    caption = generate_caption(processor, model, image, device)
    print(f"图像内容: {caption}")
    
    print("\n🎨 【构图分析】")
    print("-"*60)
    composition_analysis = analyze_composition(processor, model, image, device)
    composition_labels = [
        "视觉布局",
        "构图技巧",
        "元素排列",
        "构图规则",
        "视觉引导"
    ]
    
    for label, analysis in zip(composition_labels, composition_analysis):
        print(f"• {label}: {analysis}")
    
    print("\n🌈 【色调分析】")
    print("-"*60)
    color_analysis = analyze_color_tone(processor, model, image, device)
    color_labels = [
        "主色调",
        "色温氛围",
        "时间感知",
        "色彩和谐",
        "情感色彩"
    ]
    
    for label, analysis in zip(color_labels, color_analysis):
        print(f"• {label}: {analysis}")
    
    print("\n" + "="*80)
    print("✅ 分析完成！")

if __name__ == "__main__":
    main()