import argparse
import os

import numpy as np
import torch
from timm.models import create_model
from torchvision import transforms

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
from dataset.loader import get_video_loader


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


def resize(vid, size, interpolation='bilinear'):
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)


class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


def get_args():
    parser = argparse.ArgumentParser(
        'Extract TAD features using the videomae model', add_help=False)

    parser.add_argument(
        '--data_set',
        default='SumMe',
        choices=['THUMOS14', 'FINEACTION', 'SumMe'],
        type=str,
        help='dataset')
    parser.add_argument(
        '--data_path',
        default='./data/SumMe/videos',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--save_path',
        default='./output/videomae_embeddings',
        type=str,
        help='path for saving features')
    parser.add_argument(
        '--model',
        default='pretrain_videomae_giant_patch14_224',
        type=str,
        metavar='MODEL',
        help='Name of model')
    parser.add_argument(
        '--ckpt_path',
        default='./checkpoints/vit_g_hybrid_pt_1200e.pth',
        help='load from checkpoint')

    return parser.parse_args()


def get_start_idx_range(data_set):
    def thumos14_range(num_frames):
        return range(0, num_frames - 15, 4)

    def fineaction_range(num_frames):
        return range(0, num_frames - 15, 16)

    def summe_range(num_frames):
        return range(0, num_frames, 15)

    if data_set == 'THUMOS14':
        return thumos14_range
    elif data_set == 'FINEACTION':
        return fineaction_range
    elif data_set == 'SumMe':
        return summe_range
    else:
        raise NotImplementedError()


def extract_feature(args):
    # Preparation
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    video_loader = get_video_loader()
    start_idx_range = get_start_idx_range(args.data_set)

    transform = transforms.Compose([
        ToFloatTensorInZeroOne(),
        Resize((224, 224))
    ])

    # Get video path
    vid_list = [f for f in os.listdir(args.data_path) if f.endswith('.mp4')]

    # Get model & load ckpt
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=0,
        all_frames=8,
        tubelet_size=2,
        drop_path_rate=0.0
    )

    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    for model_key in ['model', 'module']:
        if model_key in ckpt:
            ckpt = ckpt[model_key]
            break
            
    model_dict = model.state_dict()
    # Only retain encoder weights
    pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict and "encoder" in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    model.cuda()

    # Extract feature
    num_videos = len(vid_list)
    clip_len = 8          # Frame length per clip
    frame_interval = 2    # Frame sampling interval

    for idx, vid_name in enumerate(vid_list):
        url = os.path.join(args.save_path, vid_name.split('.')[0] + '.npy')
        if os.path.exists(url):
            continue

        video_path = os.path.join(args.data_path, vid_name)
        print(f"[{idx + 1}/{num_videos}] Processing video: {vid_name}")
        
        vr = video_loader(video_path)
        feature_list = []
        num_frames = len(vr)

        for start_idx in start_idx_range(num_frames):
            end_idx = start_idx + clip_len

            # Adjust start index if close to the end of the video
            if end_idx > num_frames:
                start_idx = max(0, num_frames - clip_len)

            # Calculate sampling indices based on interval
            frame_indices = np.arange(start_idx, start_idx + frame_interval * clip_len, frame_interval)

            # Pad frames backwards if exceeding video length
            if frame_indices[-1] >= num_frames:
                frame_indices = np.arange(max(0, num_frames - frame_interval * clip_len), num_frames, frame_interval)
                # If frames are still insufficient after padding, repeat the last frame
                if len(frame_indices) < clip_len:
                    padding = np.full(clip_len - len(frame_indices), num_frames - 1)
                    frame_indices = np.concatenate([frame_indices, padding])

            # Get frame data and convert to tensor
            data = vr.get_batch(frame_indices).asnumpy()
            frame = torch.from_numpy(data)  
            frame_q = transform(frame)      
            input_data = frame_q.unsqueeze(0).cuda()

            # Create full-visible mask
            B = input_data.size(0)
            num_patches = model.encoder.patch_embed.num_patches
            mask = torch.zeros(B, num_patches, dtype=torch.bool, device=input_data.device) 

            with torch.no_grad():
                feature = model.encoder(input_data, mask)
                feature_list.append(feature.cpu().numpy()[0])

        # Save features as [N, C]
        np.save(url, np.stack(feature_list))
        print(f'Saved feature to: {url}')


if __name__ == '__main__':
    args = get_args()
    extract_feature(args)
