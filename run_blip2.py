#!/usr/bin/env python3
"""
BLIP2 模型运行脚本
支持本地图片和URL图片的描述生成和视觉问答
"""
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import requests
import argparse
import os

def setup_device():
    """设置运行设备"""
    if torch.cuda.is_available():
        device = "cuda"
        print("使用 NVIDIA GPU (CUDA)")
    elif torch.backends.mps.is_available():
        # MPS 可能有兼容性问题，可选择使用
        device = "cpu"  # 改为 "mps" 如果想尝试使用 Apple Silicon GPU
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
        # GPU上可以使用半精度
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
    else:
        # CPU上使用全精度
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            low_cpu_mem_usage=True
        ).to(device)
    
    print("✅ 模型加载完成！")
    return processor, model

def load_image(image_source):
    """加载图片（支持URL和本地文件）"""
    if image_source.startswith('http'):
        print(f"从URL加载图片: {image_source}")
        image = Image.open(requests.get(image_source, stream=True).raw).convert('RGB')
    elif os.path.exists(image_source):
        print(f"从本地加载图片: {image_source}")
        image = Image.open(image_source).convert('RGB')
    else:
        raise ValueError(f"无法加载图片: {image_source}")
    return image

def generate_caption(processor, model, image, device):
    """生成图片描述"""
    print("\n生成图片描述...")
    inputs = processor(image, return_tensors="pt").to(device)
    
    # 使用采样参数来获得更多样的输出
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=5,
        temperature=0.8
    )
    
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption

def answer_question(processor, model, image, question, device):
    """回答关于图片的问题"""
    # 为问题添加前缀，这对BLIP2很重要
    prompt = f"Question: {question} Answer:"
    
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=30,
        num_beams=3,
        temperature=0.7
    )
    
    # 只获取生成的部分（去掉输入部分）
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 提取答案部分
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
    else:
        answer = generated_text.strip()
    
    return answer

def main():
    parser = argparse.ArgumentParser(description='运行 BLIP2 模型')
    parser.add_argument(
        '--image', 
        type=str, 
        default='https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg',
        help='图片路径或URL'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['caption', 'vqa', 'both'],
        default='both',
        help='运行模式：caption(描述), vqa(问答), both(两者都运行)'
    )
    parser.add_argument(
        '--question',
        type=str,
        default=None,
        help='要问的问题（VQA模式）'
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = setup_device()
    
    # 加载模型
    processor, model = load_model(device)
    
    # 加载图片
    image = load_image(args.image)
    
    print("\n" + "="*60)
    
    # # 生成描述
    # if args.mode in ['caption', 'both']:
    #     caption = generate_caption(processor, model, image, device)
    #     print(f"📝 图片描述: {caption}")
    #     print("="*60)
    
    # 视觉问答
    if args.mode in ['vqa', 'both']:
        if args.question:
            # 使用用户提供的问题
            answer = answer_question(processor, model, image, args.question, device)
            print(f"❓ 问题: {args.question}")
            print(f"💬 回答: {answer}")
        else:
            # 默认问题集
            print("\n视觉问答演示:")
            print("-"*40)
            
            # demo_questions = [
            #     "What is in this image?",
            #     "How many people are there?",
            #     "What is the main color?",
            #     "What is happening in this picture?"
            # ]
            
            
            demo_questions = [
                # "what is the composition of the image?"
                "what is the composition of the image {Rule of Thirds} or {Symmetrical Composition} or {Diagonal Composition} or {Leading Lines Composition} or {Framing Composition} or {Central Composition} or {Golden Ratio / Golden Spiral} or {Negative Space Composition} or {Repetition & Pattern Composition} or {Asymmetrical Balance} or {Radial Composition} or {Layering Composition}"
            ]

            for q in demo_questions:
                answer = answer_question(processor, model, image, q, device)
                print(f"\n❓ 问题: {q}")
                print(f"💬 回答: {answer}")
        
        print("\n" + "="*60)
    
    print("\n✅ 完成！")

if __name__ == "__main__":
    main()