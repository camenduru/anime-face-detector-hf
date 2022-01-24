#!/usr/bin/env python

from __future__ import annotations

import mim

mim.uninstall('mmcv-full', confirm_yes=True)
mim.install('mmcv-full==1.3.16', is_yes=True)

import argparse
import functools
import os
import pathlib
import tarfile

import anime_face_detector
import cv2
import gradio as gr
import huggingface_hub
import numpy as np
import PIL.Image
import torch

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--detector', type=str, default='yolov3')
    parser.add_argument('--face-score-slider-step', type=float, default=0.05)
    parser.add_argument('--face-score-threshold', type=float, default=0.5)
    parser.add_argument('--landmark-score-slider-step',
                        type=float,
                        default=0.05)
    parser.add_argument('--landmark-score-threshold', type=float, default=0.3)
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


def detect(image, face_score_threshold: float, landmark_score_threshold: float,
           detector: anime_face_detector.LandmarkDetector) -> np.ndarray:
    image = cv2.imread(image.name)
    preds = detector(image)

    res = image.copy()
    for pred in preds:
        box = pred['bbox']
        box, score = box[:4], box[4]
        if score < face_score_threshold:
            continue
        box = np.round(box).astype(int)

        lt = max(2, int(3 * (box[2:] - box[:2]).max() / 256))

        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), lt)

        pred_pts = pred['keypoints']
        for *pt, score in pred_pts:
            if score < landmark_score_threshold:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            pt = np.round(pt).astype(int)
            cv2.circle(res, tuple(pt), lt, color, cv2.FILLED)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    return res


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    image_paths = load_sample_image_paths()
    examples = [[
        path.as_posix(), args.face_score_threshold,
        args.landmark_score_threshold
    ] for path in image_paths]

    detector = anime_face_detector.create_detector(args.detector,
                                                   device=device)
    func = functools.partial(detect, detector=detector)
    func = functools.update_wrapper(func, detect)

    repo_url = 'https://github.com/hysts/anime-face-detector'
    title = 'hysts/anime-face-detector'
    description = f'A demo for {repo_url}'
    article = None

    gr.Interface(
        func,
        [
            gr.inputs.Image(type='file', label='Input'),
            gr.inputs.Slider(0,
                             1,
                             step=args.face_score_slider_step,
                             default=args.face_score_threshold,
                             label='Face Score Threshold'),
            gr.inputs.Slider(0,
                             1,
                             step=args.landmark_score_slider_step,
                             default=args.landmark_score_threshold,
                             label='Landmark Score Threshold'),
        ],
        gr.outputs.Image(label='Output'),
        theme=args.theme,
        title=title,
        description=description,
        article=article,
        examples=examples,
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
