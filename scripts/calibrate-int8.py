#!/usr/bin/env python3
"""Generate the INT8 calibration table TensorRT needs for --precision int8.

  scripts/calibrate-int8.py models/yolo11s.onnx data/coco/val2017

Writes models/yolo11s.calibration (for the tensorrt backend) and
models/yolo11s.calibration.cache (for tensorrt-native) next to the model.
scripts/deploy-jetson.sh copies both into the board's models/trt-cache/,
where the backends look for them. The table holds value ranges only, so it is generated
once on a PC and works on any device.
"""
import argparse
import os
import sys
import tempfile

try:
    import cv2
    import numpy as np
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        create_calibrator,
        write_calibration_table,
    )
    import onnxruntime as ort
except ImportError:
    sys.exit("Install dependencies: pip install onnxruntime opencv-python-headless")


class ImageReader(CalibrationDataReader):
    """Feeds images through the same pre-processing as src/inference/pre-process.cpp"""

    def __init__(self, model, image_dir, count):
        session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.name = session.get_inputs()[0].name
        _, _, self.height, self.width = session.get_inputs()[0].shape
        files = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))
        self.paths = iter(os.path.join(image_dir, f) for f in files[:count])

    def get_next(self):
        path = next(self.paths, None)
        if path is None:
            return None
        # Stretch to the input size, BGR → RGB, scale to [0, 1], HWC → NCHW
        image = cv2.resize(cv2.imread(path), (self.width, self.height))
        blob = image[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return {self.name: blob}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="ONNX model")
    parser.add_argument("images", help="directory of .jpg calibration images")
    parser.add_argument("-n", type=int, default=200, help="images to use (default: 200)")
    args = parser.parse_args()

    output = os.path.splitext(args.model)[0] + ".calibration"
    with tempfile.TemporaryDirectory() as work:
        calibrator = create_calibrator(
            args.model,
            [],
            augmented_model_path=os.path.join(work, "augmented.onnx"),
            calibrate_method=CalibrationMethod.MinMax,
        )
        calibrator.collect_data(ImageReader(args.model, args.images, args.n))
        # Writes calibration.flatbuffers (the format the TensorRT execution
        # provider reads) and two more files into the current directory
        previous = os.getcwd()
        os.chdir(work)
        try:
            write_calibration_table(calibrator.compute_data())
        finally:
            os.chdir(previous)
        os.replace(os.path.join(work, "calibration.flatbuffers"), output)
        # The same ranges in TensorRT's own cache format, for tensorrt-native
        os.replace(os.path.join(work, "calibration.cache"), output + ".cache")
    print(f"Saved {output} and {output}.cache")


if __name__ == "__main__":
    main()
