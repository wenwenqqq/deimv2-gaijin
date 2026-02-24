# 保存为 check_uavdt_data.py
import json
import os

def check_annotation_file(json_path, img_folder):
    with open(json_path) as f:
        data = json.load(f)
    
    print(f"Checking {json_path}")
    print(f"Images: {len(data['images'])}")
    print(f"Annotations: {len(data['annotations'])}")
    print(f"Categories: {len(data['categories'])}")
    
    # 检查类别ID是否连续
    cat_ids = [cat['id'] for cat in data['categories']]
    print(f"Category IDs: {sorted(cat_ids)}")
    
    # 检查边界框
    invalid_bboxes = 0
    for ann in data['annotations']:
        bbox = ann['bbox']
        if len(bbox) != 4 or any(v < 0 for v in bbox):
            invalid_bboxes += 1
            if invalid_bboxes <= 5:
                print(f"Invalid bbox: {bbox}")
    
    print(f"Invalid bboxes: {invalid_bboxes}")
    
    # 检查图像文件是否存在
    missing_images = 0
    for img_info in data['images'][:10]:  # 只检查前10个
        img_path = os.path.join(img_folder, img_info['file_name'])
        if not os.path.exists(img_path):
            missing_images += 1
            print(f"Missing image: {img_path}")
    
    print(f"Missing images (sample check): {missing_images}")

# 检查所有标注文件
for split in ['train', 'val', 'test']:
    json_path = f'/mnt/e/Experiment/DEIMv2-main/uavdt_{split}.json'
    check_annotation_file(json_path, '/mnt/e/Experiment/OpenDataLab___UAVDT/raw/UAV-benchmark-M')