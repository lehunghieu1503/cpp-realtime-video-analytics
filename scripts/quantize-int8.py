#!/usr/bin/env python3
"""Make an INT8 model with quantize/dequantize nodes, the only INT8 TensorRT 11 builds.

  scripts/quantize-int8.py --exclude '^/model\.2/' models/stall-bev-town05.onnx data/stalls-bev/images/train
  scripts/quantize-int8.py --fp8 models/stall-bev-town05.onnx data/stalls-bev/images/train

Writes models/stall-bev-town05_int8.onnx (_fp8 with --fp8), the files
simulation/run.sh --int8 (--fp8) runs. Run it with `precision: fp16`
in the config: the file itself says which layers are INT8, and `int8` makes
TensorRT 11 stop (no calibrator any more). The calibration images must not be
the ones the model is evaluated on. Keeping /model.2/ out of INT8 is what
makes the bird's-eye-view stall model class stalls as FP32 does.

  --tool ort       ONNX Runtime quantize_static (pip install onnxruntime
                   onnxconverter-common); Conv and MatMul in INT8, the rest
                   converted to FP16
  --tool modelopt  NVIDIA ModelOpt (pip install "nvidia-modelopt[onnx]"); puts
                   the nodes where TensorRT fuses them, the rest in FP16
  --fp8            FP8 (E4M3) instead of INT8, with ModelOpt; needs a GPU with
                   FP8 tensor cores (Ada, Hopper, Blackwell), not the Jetson's
"""
import argparse
import itertools
import os
import re
import shutil
import sys
import tempfile

import cv2
import numpy as np
import onnx


def calibration_images(model, image_dir, count):
    """count images spread over the sorted directory, pre-processed as
    src/inference/pre-process.cpp does: stretched to the input size, BGR → RGB,
    [0, 1], NCHW"""
    _, _, height, width = (d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)
    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))
    if len(files) < count:
        sys.exit(f"{image_dir} has {len(files)} .jpg images, fewer than -n {count}")
    files = files[:: len(files) // count][:count]
    images = np.empty((count, 3, height, width), np.float32)
    for i, f in enumerate(files):
        image = cv2.resize(cv2.imread(os.path.join(image_dir, f)), (width, height))
        images[i] = image[:, :, ::-1].transpose(2, 0, 1) / 255.0
    return images


def quantize_ort(model_path, output, images, method, exclude):
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                          QuantFormat, QuantType, quantize_static)
    from onnxruntime.quantization.calibrate import HistogramCalibrater

    class Reader(CalibrationDataReader):
        def __init__(self, batches):
            self.batches = iter(batches)

        def get_next(self):
            return next(self.batches, None)

    # The histogram calibrators keep every tensor of every image until the end
    # (24 GB for 200 images of this size): feed them 10 at a time, their
    # histograms add up across calls
    collect = HistogramCalibrater.collect_data

    def collect_in_chunks(self, reader):
        batches = iter(reader.get_next, None)
        while chunk := list(itertools.islice(batches, 10)):
            collect(self, Reader(chunk))

    HistogramCalibrater.collect_data = collect_in_chunks
    name = onnx.load(model_path, load_external_data=False).graph.input[0].name
    batches = ({name: images[i:i + 1]} for i in range(len(images)))

    quantize_static(
        model_path, output, Reader(batches),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv", "MatMul"],
        nodes_to_exclude=exclude,
        per_channel=True,
        # TensorRT takes only signed INT8 with a zero point of 0, and no Int32 bias
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        calibrate_method={"max": CalibrationMethod.MinMax,
                          "entropy": CalibrationMethod.Entropy}[method],
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                       "QuantizeBias": False})

    # TensorRT 11 runs what is not quantized in the type the file gives it:
    # FP16 instead of FP32, which needs opset 19 for FP16 quantize nodes
    from onnx import version_converter
    from onnxconverter_common import float16
    model = version_converter.convert_version(onnx.load(output), 19)
    onnx.save(float16.convert_float_to_float16(
        model, keep_io_types=True, op_block_list=float16.DEFAULT_OP_BLOCK_LIST), output)


def quantize_modelopt(model_path, output, images, method, exclude, mode="int8"):
    from modelopt.onnx.quantization import quantize

    # ModelOpt saves the model it is given back over its path: give it a copy
    with tempfile.TemporaryDirectory() as work:
        copy = shutil.copy(model_path, work)
        quantize(copy, quantize_mode=mode, calibration_data=images,
                 calibration_method=method, output_path=output,
                 high_precision_dtype="fp16", nodes_to_exclude=exclude)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="FP32 ONNX model")
    parser.add_argument("images", help="directory of .jpg calibration images")
    parser.add_argument("--tool", choices=["ort", "modelopt"], default="ort")
    parser.add_argument("--method", choices=["entropy", "max"], default="entropy",
                        help="how a tensor's range is chosen (default: entropy)")
    parser.add_argument("-n", type=int, default=200, help="images to use (default: 200)")
    parser.add_argument("--exclude", metavar="REGEX",
                        help="nodes whose name matches stay unquantized, e.g. '/model.23/cv3'")
    parser.add_argument("--fp8", action="store_true", help="FP8 instead of INT8, with ModelOpt")
    parser.add_argument("-o", "--output", help="default: MODEL_int8.onnx, MODEL_fp8.onnx with --fp8")
    args = parser.parse_args()

    mode = "fp8" if args.fp8 else "int8"
    output = args.output or f"{os.path.splitext(args.model)[0]}_{mode}.onnx"
    model = onnx.load(args.model, load_external_data=False)
    exclude = [n.name for n in model.graph.node
               if args.exclude and re.search(args.exclude, n.name)]
    if args.exclude and not exclude:
        sys.exit(f"no node name matches {args.exclude}")
    images = calibration_images(model, args.images, args.n)
    if args.fp8:
        quantize_modelopt(args.model, output, images, args.method, exclude, "fp8")
    else:
        {"ort": quantize_ort, "modelopt": quantize_modelopt}[args.tool](
            args.model, output, images, args.method, exclude)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
