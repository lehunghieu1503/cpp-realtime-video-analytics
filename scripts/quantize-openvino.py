#!/usr/bin/env python3
"""Quantize a model to INT8 for the OpenVINO backend (needed for good NPU speed).

  scripts/quantize-openvino.py models/yolo11s.onnx data/coco/val2017
  build/src/video_analytics --backend openvino:NPU --model models/yolo11s_int8.xml ...

Writes models/yolo11s_int8.xml and .bin next to the model.
"""
import argparse
import os
import sys

try:
    import cv2
    import nncf
    import numpy as np
    import openvino as ov
except ImportError:
    sys.exit("Install dependencies: pip install nncf openvino opencv-python-headless")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="ONNX model")
    parser.add_argument("images", help="directory of .jpg calibration images")
    parser.add_argument("-n", type=int, default=300, help="images to use (default: 300)")
    args = parser.parse_args()

    model = ov.Core().read_model(args.model)
    _, _, height, width = model.input().shape
    files = sorted(f for f in os.listdir(args.images) if f.endswith(".jpg"))[: args.n]

    def preprocess(name):
        # Same as src/inference/pre-process.cpp: stretch, BGR → RGB, [0, 1], NCHW
        image = cv2.resize(cv2.imread(os.path.join(args.images, name)), (width, height))
        return image[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    quantized = nncf.quantize(
        model, nncf.Dataset(files, preprocess), subset_size=len(files)
    )
    output = os.path.splitext(args.model)[0] + "_int8.xml"
    ov.save_model(quantized, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
