import os
import json
from PIL import Image
import argparse
import random

# ========== 核心修改1：类别ID直接映射为0开始的连续ID ==========
# 原始UAVDT类别ID → 模型训练用ID（0/1/2）
UAVDT_CATEGORY_MAP = {
    1: 0,  # car → 0
    2: 1,  # truck → 1
    3: 2   # bus → 2
}

# 类别名称映射（ID为0/1/2）
CATEGORY_NAMES = {
    0: 'car',
    1: 'truck',
    2: 'bus'
}

# 基于M_attr文件夹的官方序列划分
OFFICIAL_TRAIN_SEQUENCES = [
    'M0101', 'M0201', 'M0202', 'M0204', 'M0206', 'M0207', 'M0210', 'M0301',
    'M0401', 'M0402', 'M0501', 'M0603', 'M0604', 'M0605', 'M0702', 'M0703',
    'M0704', 'M0901', 'M0902', 'M1002', 'M1003', 'M1005', 'M1006', 'M1008',
    'M1102', 'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
]

OFFICIAL_TEST_SEQUENCES = [
    'M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606',
    'M0701', 'M0801', 'M0802', 'M1001', 'M1004', 'M1007', 'M1009', 'M1101',
    'M1301', 'M1302', 'M1303', 'M1401'
]

def get_sequences_for_split(split, val_ratio=0.2, random_seed=42):
    """
    根据split类型返回对应的序列列表
    
    Args:
        split: 'train', 'val', 或 'test'
        val_ratio: 从训练集中划分验证集的比例
        random_seed: 随机种子，确保可重复性
    """
    if split == 'test':
        return OFFICIAL_TEST_SEQUENCES
    elif split == 'train' or split == 'val':
        # 从官方训练集中划分train/val
        random.seed(random_seed)
        train_sequences = OFFICIAL_TRAIN_SEQUENCES.copy()
        random.shuffle(train_sequences)
        
        val_size = int(len(train_sequences) * val_ratio)
        
        if split == 'val':
            return train_sequences[:val_size]
        else:  # train
            return train_sequences[val_size:]
    else:
        raise ValueError(f"Unknown split: {split}")

def convert_uavdt_to_coco(image_root, annotation_root, output_file, split="train", val_ratio=0.2):
    """
    将UAVDT数据集转换为COCO格式（类别ID直接转为0开始）
    
    Args:
        image_root: UAV-benchmark-M目录路径
        annotation_root: UAV-benchmark-MOTD_v1.0/GT目录路径
        output_file: 输出的COCO标注文件路径
        split: 数据集分割（train/val/test）
        val_ratio: 验证集比例（仅当split为train或val时有效）
    """
    # 获取当前split对应的序列列表
    target_sequences = get_sequences_for_split(split, val_ratio)
    print(f"Processing {split} split with {len(target_sequences)} sequences: {target_sequences}")
    
    # 初始化COCO格式数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # ========== 核心修改2：类别信息使用0开始的ID ==========
    # 添加类别信息（ID为0/1/2）
    for category_id, category_name in CATEGORY_NAMES.items():
        coco_data["categories"].append({
            "id": category_id,
            "name": category_name,
            "supercategory": "vehicle"
        })
    
    # 遍历目标序列目录
    image_id = 0
    annotation_id = 0
    
    for seq_name in target_sequences:
        seq_dir = os.path.join(image_root, seq_name)
        if not os.path.isdir(seq_dir):
            print(f"Warning: Sequence directory not found: {seq_dir}, skipping...")
            continue
        
        print(f"Processing sequence: {seq_name}")
        
        # 检查标注文件是否存在
        det_anno_file = os.path.join(annotation_root, f"{seq_name}_gt_whole.txt")
        ignore_anno_file = os.path.join(annotation_root, f"{seq_name}_gt_ignore.txt")
        
        if not os.path.exists(det_anno_file):
            print(f"Warning: Detection annotation file not found for {seq_name}, skipping...")
            continue
        
        # 读取图像文件列表
        image_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])
        
        # 构建帧到图像文件的映射
        frame_to_image = {}
        for img_file in image_files:
            frame_idx = int(img_file[3:9])  # 从img000001.jpg提取帧索引
            frame_to_image[frame_idx] = img_file
        
        # 处理检测标注
        with open(det_anno_file, 'r') as f:
            det_lines = f.readlines()
        
        # 按帧分组标注
        frame_annotations = {}
        for line in det_lines:
            parts = line.strip().split(',')
            frame_idx = int(parts[0])
            target_id = int(parts[1])
            bbox_left = float(parts[2])
            bbox_top = float(parts[3])
            bbox_width = float(parts[4])
            bbox_height = float(parts[5])
            out_of_view = int(parts[6])
            occlusion = int(parts[7])
            category_id = int(parts[8])
            
            # ========== 核心修改3：转换为0开始的类别ID ==========
            # 仅处理映射表中的类别（1/2/3 → 0/1/2）
            if category_id in UAVDT_CATEGORY_MAP:
                coco_category_id = UAVDT_CATEGORY_MAP[category_id]
            else:
                print(f"Warning: Unknown category ID {category_id} in {seq_name}, frame {frame_idx}, skipping...")
                continue
            
            if frame_idx not in frame_annotations:
                frame_annotations[frame_idx] = []
            
            frame_annotations[frame_idx].append({
                "target_id": target_id,
                "bbox": [bbox_left, bbox_top, bbox_width, bbox_height],
                "category_id": coco_category_id,  # 使用转换后的ID
                "out_of_view": out_of_view,
                "occlusion": occlusion
            })
        
        # 处理忽略区域标注
        ignore_annotations = {}
        if os.path.exists(ignore_anno_file):
            with open(ignore_anno_file, 'r') as f:
                ignore_lines = f.readlines()
            
            for line in ignore_lines:
                parts = line.strip().split(',')
                frame_idx = int(parts[0])
                target_id = int(parts[1])
                bbox_left = float(parts[2])
                bbox_top = float(parts[3])
                bbox_width = float(parts[4])
                bbox_height = float(parts[5])
                
                if frame_idx not in ignore_annotations:
                    ignore_annotations[frame_idx] = []
                
                ignore_annotations[frame_idx].append({
                    "target_id": target_id,
                    "bbox": [bbox_left, bbox_top, bbox_width, bbox_height]
                })
        
        # 为每个图像创建COCO标注
        for frame_idx, img_file in frame_to_image.items():
            img_path = os.path.join(seq_dir, img_file)
            
            # 获取图像尺寸
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                continue
            
            # 添加图像信息
            image_info = {
                "id": image_id,
                "file_name": os.path.join(seq_name, img_file),
                "width": width,
                "height": height,
                "date_captured": "2018-08-01",
                "license": 1,
                "coco_url": "",
                "flickr_url": ""
            }
            coco_data["images"].append(image_info)
            
            # 添加检测标注
            if frame_idx in frame_annotations:
                for ann in frame_annotations[frame_idx]:
                    bbox_left, bbox_top, bbox_width, bbox_height = ann["bbox"]
                    
                    # 计算面积
                    area = bbox_width * bbox_height
                    
                    # 确保边界框在图像范围内（修复越界问题）
                    x1 = max(0, bbox_left)
                    y1 = max(0, bbox_top)
                    x2 = min(width, bbox_left + bbox_width)
                    y2 = min(height, bbox_top + bbox_height)
                    
                    # 更新边界框（确保width/height为正）
                    bbox_width_fixed = max(1e-6, x2 - x1)
                    bbox_height_fixed = max(1e-6, y2 - y1)
                    bbox = [x1, y1, bbox_width_fixed, bbox_height_fixed]
                    area = bbox_width_fixed * bbox_height_fixed
                    
                    # 添加标注（使用转换后的类别ID）
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": ann["category_id"],  # 0/1/2
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [],
                        "ignore": 0
                    }
                    coco_data["annotations"].append(annotation)
                    annotation_id += 1
            
            # 添加忽略区域标注
            if frame_idx in ignore_annotations:
                for ann in ignore_annotations[frame_idx]:
                    bbox_left, bbox_top, bbox_width, bbox_height = ann["bbox"]
                    
                    # 计算面积
                    area = bbox_width * bbox_height
                    
                    # 确保边界框在图像范围内
                    x1 = max(0, bbox_left)
                    y1 = max(0, bbox_top)
                    x2 = min(width, bbox_left + bbox_width)
                    y2 = min(height, bbox_top + bbox_height)
                    
                    # 更新边界框（确保width/height为正）
                    bbox_width_fixed = max(1e-6, x2 - x1)
                    bbox_height_fixed = max(1e-6, y2 - y1)
                    bbox = [x1, y1, bbox_width_fixed, bbox_height_fixed]
                    area = bbox_width_fixed * bbox_height_fixed
                    
                    # ========== 核心修改4：忽略区域类别ID也用0（避免越界） ==========
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 0,  # 用0（有效类别ID），避免未知ID
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 1,  # 标记为忽略区域
                        "segmentation": [],
                        "ignore": 1
                    }
                    coco_data["annotations"].append(annotation)
                    annotation_id += 1
            
            image_id += 1
    
    # 保存COCO标注文件
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Conversion completed!")
    print(f"Number of images: {len(coco_data['images'])}")
    print(f"Number of annotations: {len(coco_data['annotations'])}")
    print(f"COCO annotations saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert UAVDT dataset to COCO format (0-start category ID)')
    parser.add_argument('--image_root', type=str, default='/mnt/e/Experiment/OpenDataLab___UAVDT/raw/UAV-benchmark-M',
                        help='Path to UAV-benchmark-M directory')
    parser.add_argument('--annotation_root', type=str, default='/mnt/e/Experiment/OpenDataLab___UAVDT/raw/UAV-benchmark-MOTD_v1.0/GT',
                        help='Path to UAV-benchmark-MOTD_v1.0/GT directory')
    parser.add_argument('--output_file', type=str, default='uavdt_train.json',
                        help='Path to output COCO annotation file')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Dataset split')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Ratio of validation set from training set (default: 0.2)')
    
    args = parser.parse_args()
    
    convert_uavdt_to_coco(args.image_root, args.annotation_root, args.output_file, 
                         args.split, args.val_ratio)

if __name__ == '__main__':
    main()