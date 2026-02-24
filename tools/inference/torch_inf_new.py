"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE ( `https://github.com/Peterande/D-FINE)` 
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""
 
import os
import sys
 
import cv2  # Added for video processing
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig
from engine.data.dataset import mscoco_label2category, mscoco_category2name
 
 
def draw(images, labels, boxes, scores, thrh=0.45):
    # ===== 字体设置（调这里的 size 即可）=====
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        size=15   # 👈 字体大小，建议 24~36
    )

    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        keep = scr > thrh
        lab = labels[i][keep]
        box = boxes[i][keep]
        scrs = scr[keep]

        for j, b in enumerate(box):
            # 画检测框
            draw.rectangle(list(b), outline="red", width=3)

            # ===== 类别名 =====
            label_id = lab[j].item()
            category_id = mscoco_label2category[label_id]
            category_name = mscoco_category2name[category_id]
            score = scrs[j].item()

            text = f"{category_name} {score:.2f}"

            # ===== 计算文字尺寸 =====
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # ===== 文字背景 =====
            draw.rectangle(
                [
                    b[0],
                    b[1] - text_h - 2,
                    b[0] + text_w + 2,
                    b[1]
                ],
                fill="red"
            )

            # ===== 写文字 =====
            draw.text(
                (b[0] + 3, b[1] - text_h - 3),
                text,
                fill="white",
                font=font
            )

        im.save("torch_example.jpg")

 
 
def process_image(model, device, file_path, size=(640, 640), vit_backbone=False):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
 
    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                if vit_backbone else T.Lambda(lambda x: x)
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)
 
    output = model(im_data, orig_size)
    labels, boxes, scores = output
 
    draw([im_pil], labels, boxes, scores)
 
 
def process_video(model, device, file_path, size=(640, 640), vit_backbone=False):
    cap = cv2.VideoCapture(file_path)
 
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('torch_results.mp4', fourcc, fps, (orig_w, orig_h))
 
    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                if vit_backbone else T.Lambda(lambda x: x)
    ])
 
    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
 
        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
 
        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
 
        im_data = transforms(frame_pil).unsqueeze(0).to(device)
 
        output = model(im_data, orig_size)
        labels, boxes, scores = output
 
        # Draw detections on the frame
        draw([frame_pil], labels, boxes, scores)
 
        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
 
        # Write the frame
        out.write(frame)
        frame_count += 1
 
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")
 
    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'torch_results.mp4'.")
 
 
def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)
 
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
 
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
 
    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)
 
    class Model(nn.Module):
 
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
 
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
 
    device = args.device
    model = Model().to(device)
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    vit_backbone = cfg.yaml_cfg.get('DINOv3STAs', False)
 
    # Check if the input file is an image or a video
    file_path = args.input
    if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process as image
        process_image(model, device, file_path, img_size, vit_backbone)
        print("Image processing complete.")
    else:
        # Process as video
        process_video(model, device, file_path, img_size, vit_backbone)
 
 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str, required=True)
    # parser.add_argument('-r', '--resume', type=str, required=True)
    # parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, default='/mnt/e/Experiment/DEIMv2-main/configs/deimv2/deimv2_hgnetv2_n_coco.yml')
    parser.add_argument('-r', '--resume', type=str, default='/mnt/e/Experiment/DEIMv2-main/weight/hgnetv2/deimv2_hgnetv2_n_coco.pth')
    parser.add_argument('-i', '--input', type=str, default='/mnt/e/Experiment/DEIMv2-main/example.jpg')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    args = parser.parse_args()
 
    main(args)