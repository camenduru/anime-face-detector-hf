#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import subprocess
import tarfile

if os.environ.get('SYSTEM') == 'spaces':
    import mim

    mim.uninstall('mmcv-full', confirm_yes=True)
    mim.install('mmcv-full==1.3.16', is_yes=True)

    subprocess.call('pip uninstall -y opencv-python'.split())
    subprocess.call('pip uninstall -y opencv-python-headless'.split())
    subprocess.call('pip install opencv-python-headless'.split())

import anime_face_detector
import cv2
import gradio as gr
import huggingface_hub
import numpy as np
import torch

TITLE = 'hysts/anime-face-detector'
DESCRIPTION = 'This is a demo for https://github.com/hysts/anime-face-detector.'
ARTICLE = '<center><img src="https://visitor-badge.glitch.me/badge?page_id=hysts.anime-face-detector" alt="visitor badge"/></center>'

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


def load_sample_image_paths() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        dataset_repo = 'hysts/sample-images-TADNE'
        path = huggingface_hub.hf_hub_download(dataset_repo,
                                               'images.tar.gz',
                                               repo_type='dataset',
                                               use_auth_token=TOKEN)
        with tarfile.open(path) as f:
            f.extractall()
    return sorted(image_dir.glob('*'))


def detect(
        image: np.ndarray, detector_name: str, face_score_threshold: float,
        landmark_score_threshold: float,
        detectors: dict[str,
                        anime_face_detector.LandmarkDetector]) -> np.ndarray:
    detector = detectors[detector_name]
    # RGB -> BGR
    image = image[:, :, ::-1]
    preds = detector(image)

    res = image.copy()
    for pred in preds:
        box = pred['bbox']
        box, score = box[:4], box[4]
        if score < face_score_threshold:
            continue
        box = np.round(box).astype(int)

        line_width = max(2, int(3 * (box[2:] - box[:2]).max() / 256))

        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0),
                      line_width)

        pred_pts = pred['keypoints']
        for *pt, score in pred_pts:
            if score < landmark_score_threshold:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            pt = np.round(pt).astype(int)
            cv2.circle(res, tuple(pt), line_width, color, cv2.FILLED)
    return res[:, :, ::-1]


def main():
    args = parse_args()
    device = torch.device(args.device)

    detector_names = ['faster-rcnn', 'yolov3']
    detectors = {
        detector_name: anime_face_detector.create_detector(detector_name,
                                                           device=device)
        for detector_name in detector_names
    }

    func = functools.partial(detect, detectors=detectors)
    func = functools.update_wrapper(func, detect)

    image_paths = load_sample_image_paths()
    examples = [[path.as_posix(), 'yolov3', 0.5, 0.3] for path in image_paths]

    gr.Interface(
        func,
        [
            gr.inputs.Image(type='numpy', label='Input'),
            gr.inputs.Radio(detector_names,
                            type='value',
                            default='yolov3',
                            label='Detector'),
            gr.inputs.Slider(
                0, 1, step=0.05, default=0.5, label='Face Score Threshold'),
            gr.inputs.Slider(
                0, 1, step=0.05, default=0.3,
                label='Landmark Score Threshold'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
