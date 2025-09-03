#!/usr/bin/env python3
"""
BLIP2 图像分析脚本 - 中文版，专注于构图和色调分析
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
        width, height = image.size
        print(f"图片尺寸: {width}x{height}")
    else:
        raise ValueError(f"无法找到图片: {image_path}")
    return image

def analyze_with_prompt(processor, model, image, prompt, device, max_tokens=60):
    """使用特定提示词分析图片"""
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=5,
        do_sample=True,
        top_p=0.95
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
    else:
        answer = generated_text.replace(prompt, "").strip()
    
    return answer

def comprehensive_analysis(processor, model, image, device):
    """综合分析图片的构图和色调"""
    
    print("\n开始分析图片...")
    results = {}
    
    # 基础描述
    base_prompt = "This is a photograph showing"
    base_inputs = processor(image, base_prompt, return_tensors="pt").to(device)
    base_ids = model.generate(**base_inputs, max_new_tokens=80, num_beams=5, do_sample=True, top_p=0.9)
    base_desc = processor.batch_decode(base_ids, skip_special_tokens=True)[0].replace(base_prompt, "").strip()
    results['基础描述'] = base_desc
    
    # 构图分析 - 使用更具体的提示词
    composition_prompts = {
        '构图类型': "Question: Is this image using rule of thirds, centered composition, symmetrical, diagonal lines, or leading lines? Answer:",
        '前景中景背景': "Question: Describe the foreground, middle ground, and background elements. Answer:",
        '视觉平衡': "Question: Is the visual weight balanced or asymmetrical in this photo? Answer:",
        '水平线位置': "Question: Where is the horizon line positioned in the frame? Answer:",
        '视觉焦点': "Question: What is the main subject or focal point? Answer:"
    }
    
    for key, prompt in composition_prompts.items():
        answer = analyze_with_prompt(processor, model, image, prompt, device, max_tokens=40)
        results[key] = answer
    
    # 色调分析 - 使用更具体的提示词  
    color_prompts = {
        '主要颜色': "Question: What are the main colors visible? List the top 3 colors. Answer:",
        '色彩氛围': "Question: Are the colors warm (reds/oranges/yellows) or cool (blues/greens/purples)? Answer:",
        '光线特征': "Question: What type of lighting is present - sunrise, sunset, midday, overcast, or golden hour? Answer:",
        '色彩对比': "Question: Is there high contrast or low contrast between colors? Answer:",
        '天空特征': "Question: If there's a sky, describe its colors and cloud patterns. Answer:"
    }
    
    for key, prompt in color_prompts.items():
        answer = analyze_with_prompt(processor, model, image, prompt, device, max_tokens=40)
        results[key] = answer
    
    return results

def interpret_results(results):
    """将英文结果解释为中文分析"""
    print("\n" + "="*80)
    print("📸 图像专业分析报告")
    print("="*80)
    
    print("\n【📝 场景概述】")
    print("-"*60)
    base_desc = results.get('基础描述', 'N/A')
    print(f"场景内容: {base_desc}")
    
    print("\n【🎨 构图分析】")
    print("-"*60)
    
    # 构图类型判断
    comp_type = results.get('构图类型', '').lower()
    if 'rule of thirds' in comp_type:
        print("• 构图法则: 三分法构图")
    elif 'centered' in comp_type or 'center' in comp_type:
        print("• 构图法则: 中心构图")
    elif 'symmetr' in comp_type:
        print("• 构图法则: 对称构图")
    elif 'diagonal' in comp_type:
        print("• 构图法则: 对角线构图")
    elif 'leading lines' in comp_type:
        print("• 构图法则: 引导线构图")
    else:
        print(f"• 构图特征: {comp_type}")
    
    # 空间层次
    layers = results.get('前景中景背景', '')
    if layers:
        print(f"• 空间层次: {layers}")
    
    # 视觉平衡
    balance = results.get('视觉平衡', '').lower()
    if 'balanced' in balance and 'asymmetrical' not in balance:
        print("• 视觉平衡: 平衡构图，左右视觉重量相当")
    elif 'asymmetrical' in balance:
        print("• 视觉平衡: 非对称构图，创造动感")
    else:
        print(f"• 视觉平衡: {balance}")
    
    # 地平线
    horizon = results.get('水平线位置', '').lower()
    if 'top' in horizon or 'upper' in horizon:
        print("• 地平线: 位于画面上三分之一，强调前景")
    elif 'bottom' in horizon or 'lower' in horizon:
        print("• 地平线: 位于画面下三分之一，强调天空")
    elif 'middle' in horizon or 'center' in horizon:
        print("• 地平线: 位于画面中央")
    else:
        print(f"• 地平线位置: {horizon}")
    
    # 焦点
    print(f"• 视觉焦点: {results.get('视觉焦点', 'N/A')}")
    
    print("\n【🌈 色调分析】")
    print("-"*60)
    
    # 主色调
    colors = results.get('主要颜色', '')
    print(f"• 主要色彩: {colors}")
    
    # 色温
    temp = results.get('色彩氛围', '').lower()
    if 'warm' in temp:
        print("• 色温: 暖色调 (偏红橙黄)，营造温暖、活力的氛围")
    elif 'cool' in temp:
        print("• 色温: 冷色调 (偏蓝绿紫)，营造宁静、清爽的氛围")
    else:
        print(f"• 色温特征: {temp}")
    
    # 光线
    light = results.get('光线特征', '').lower()
    if 'sunrise' in light:
        print("• 光线时刻: 日出时分，柔和的晨光")
    elif 'sunset' in light:
        print("• 光线时刻: 日落时分，温暖的夕阳光")
    elif 'golden hour' in light:
        print("• 光线时刻: 黄金时刻，最佳拍摄光线")
    elif 'midday' in light:
        print("• 光线时刻: 正午，强烈的日光")
    elif 'overcast' in light:
        print("• 光线时刻: 阴天，柔和均匀的散射光")
    else:
        print(f"• 光线特征: {light}")
    
    # 对比度
    contrast = results.get('色彩对比', '').lower()
    if 'high contrast' in contrast:
        print("• 对比度: 高对比度，画面层次分明")
    elif 'low contrast' in contrast:
        print("• 对比度: 低对比度，画面柔和统一")
    else:
        print(f"• 对比特征: {contrast}")
    
    # 天空
    sky = results.get('天空特征', '')
    if sky and sky != 'N/A':
        print(f"• 天空描述: {sky}")
    
    print("\n【💡 专业建议】")
    print("-"*60)
    
    # 基于分析给出建议
    if 'sunset' in light or 'sunrise' in light:
        print("• 这是黄金时刻拍摄，光线条件极佳")
    
    if 'rule of thirds' in comp_type:
        print("• 构图遵循三分法，画面平衡感良好")
    
    if 'warm' in temp and ('sunset' in light or 'sunrise' in light):
        print("• 暖色调与黄金时刻完美配合，营造温馨氛围")
    
    if 'reflection' in base_desc.lower() or 'water' in base_desc.lower():
        print("• 水面倒影增加了画面的对称美和空间深度")

def main():
    parser = argparse.ArgumentParser(description='BLIP2 图像构图与色调专业分析')
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
    
    # 执行综合分析
    results = comprehensive_analysis(processor, model, image, device)
    
    # 解释结果
    interpret_results(results)
    
    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("\n提示: BLIP2 模型的分析基于视觉理解，可能存在一定偏差。")
    print("建议结合人工判断以获得最准确的图像分析。")

if __name__ == "__main__":
    main()