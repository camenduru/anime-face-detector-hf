#!/usr/bin/env python

import mim

mim.uninstall('mmcv-full', confirm_yes=True)
mim.install('mmcv-full==1.3.16', is_yes=True)

import functools
import pathlib

import cv2
import gradio as gr
import numpy as np
import PIL.Image
import torch

import anime_face_detector


def detect(img, face_score_threshold: float, landmark_score_threshold: float,
           detector: anime_face_detector.LandmarkDetector) -> PIL.Image.Image:
    image = cv2.imread(img.name)
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

    image_pil = PIL.Image.fromarray(res)
    return image_pil


def main():
    sample_path = pathlib.Path('input.jpg')
    if not sample_path.exists():
        torch.hub.download_url_to_file(
            'https://raw.githubusercontent.com/hysts/anime-face-detector/main/assets/input.jpg',
            sample_path.as_posix())

    detector_name = 'yolov3'
    device = 'cpu'
    score_slider_step = 0.05
    face_score_threshold = 0.5
    landmark_score_threshold = 0.3
    live = False

    detector = anime_face_detector.create_detector(detector_name,
                                                   device=device)
    func = functools.partial(detect, detector=detector)
    func = functools.update_wrapper(func, detect)

    title = 'hysts/anime-face-detector'
    description = 'Demo for hysts/anime-face-detector. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below.'
    article = "<a href='https://github.com/hysts/anime-face-detector'>GitHub Repo</a>"

    gr.Interface(
        func,
        [
            gr.inputs.Image(type='file', label='Input'),
            gr.inputs.Slider(0,
                             1,
                             step=score_slider_step,
                             default=face_score_threshold,
                             label='Face Score Threshold'),
            gr.inputs.Slider(0,
                             1,
                             step=score_slider_step,
                             default=landmark_score_threshold,
                             label='Landmark Score Threshold'),
        ],
        gr.outputs.Image(type='pil', label='Output'),
        title=title,
        description=description,
        article=article,
        examples=[
            [
                sample_path.as_posix(),
                face_score_threshold,
                landmark_score_threshold,
            ],
        ],
        enable_queue=True,
        live=live,
    ).launch()


if __name__ == '__main__':
    main()
