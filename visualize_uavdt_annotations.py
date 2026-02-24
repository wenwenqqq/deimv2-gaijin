# /mnt/e/Experiment/DEIMv2-main/visualize_uavdt_annotations.py

import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import numpy as np

# UAVDT类别映射
UAVDT_CATEGORIES = {
    1: 'car',
    2: 'truck', 
    3: 'bus'
}

# 类别颜色映射
CATEGORY_COLORS = {
    1: 'red',    # car
    2: 'blue',   # truck
    3: 'green'   # bus
}

def visualize_annotations(json_file, img_root, num_images=10, save_dir='visualization_results'):
    """
    可视化UAVDT数据集的标注框
    
    Args:
        json_file: COCO格式的标注文件路径
        img_root: 图像根目录路径
        num_images: 要可视化的图像数量
        save_dir: 保存可视化结果的目录
    """
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载标注文件
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # 构建图像ID到图像信息的映射
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # 构建图像ID到标注的映射
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)
    
    # 随机选择图像ID
    all_image_ids = list(image_id_to_info.keys())
    selected_image_ids = random.sample(all_image_ids, min(num_images, len(all_image_ids)))
    
    print(f"从 {len(all_image_ids)} 张图像中随机选择了 {len(selected_image_ids)} 张进行可视化")
    
    # 可视化每张图像
    for i, image_id in enumerate(selected_image_ids):
        image_info = image_id_to_info[image_id]
        annotations = image_id_to_annotations.get(image_id, [])
        
        # 构建完整的图像路径
        img_path = os.path.join(img_root, image_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"警告: 图像文件不存在: {img_path}")
            continue
        
        print(f"可视化第 {i+1} 张图像: {image_info['file_name']}")
        print(f"  图像尺寸: {image_info['width']}x{image_info['height']}")
        print(f"  标注数量: {len(annotations)}")
        
        # 加载图像
        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
        except Exception as e:
            print(f"  错误: 无法加载图像 {img_path}: {e}")
            continue
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_array)
        ax.set_title(f"Image: {image_info['file_name']}\nSize: {image_info['width']}x{image_info['height']}, Annotations: {len(annotations)}")
        
        # 绘制标注框
        for ann in annotations:
            category_id = ann['category_id']
            bbox = ann['bbox']  # [x, y, width, height]
            is_crowd = ann.get('iscrowd', 0)
            
            # 跳过忽略区域（iscrowd=1）
            if is_crowd == 1:
                continue
            
            x, y, width, height = bbox
            
            # 创建矩形框
            color = CATEGORY_COLORS.get(category_id, 'yellow')
            linestyle = '--' if is_crowd == 1 else '-'
            alpha = 0.7 if is_crowd == 1 else 1.0
            
            rect = patches.Rectangle(
                (x, y), width, height,
                linewidth=2, edgecolor=color, facecolor='none',
                linestyle=linestyle, alpha=alpha
            )
            ax.add_patch(rect)
            
            # 添加类别标签
            category_name = UAVDT_CATEGORIES.get(category_id, f'class_{category_id}')
            ax.text(x, y-5, f"{category_name} (ID:{category_id})", 
                   color=color, fontsize=10, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.axis('off')
        plt.tight_layout()
        
        # 保存结果
        base_name = os.path.basename(image_info['file_name'])
        save_path = os.path.join(save_dir, f"visualized_{i+1:03d}_{base_name}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存到: {save_path}")
        
        plt.close(fig)
    
    print(f"\n可视化完成！结果保存在: {save_dir}")
    print(f"共可视化了 {len(selected_image_ids)} 张图像")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化UAVDT数据集的标注框')
    parser.add_argument('--json_file', type=str, default='uavdt_train.json',
                       help='COCO格式的标注文件路径')
    parser.add_argument('--img_root', type=str, default='/mnt/e/Experiment/OpenDataLab___UAVDT/raw/UAV-benchmark-M',
                       help='图像根目录路径')
    parser.add_argument('--num_images', type=int, default=10,
                       help='要可视化的图像数量')
    parser.add_argument('--save_dir', type=str, default='visualization_results',
                       help='保存可视化结果的目录')
    
    args = parser.parse_args()
    
    visualize_annotations(args.json_file, args.img_root, args.num_images, args.save_dir)

if __name__ == '__main__':
    main()

    import json
    # 加载你的标注文件
    with open('/mnt/e/Experiment/DEIMv2-main/uavdt_val.json', 'r') as f:
        data = json.load(f)
    # 检查前几个标注
    for ann in data['annotations'][:5]:
        print(f"bbox: {ann['bbox']}")  # 应该是 [x, y, width, height]
        print(f"area: {ann['area']}")  # 应该等于 width * height
        print("---")