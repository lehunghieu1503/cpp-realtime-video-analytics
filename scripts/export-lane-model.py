#!/usr/bin/env python3
"""Export the lane model to ONNX: Ultra Fast Lane Detection, ResNet-18, TuSimple.

The authors publish a PyTorch checkpoint and a TorchScript exporter, not an
ONNX file. This script loads the checkpoint into the authors' model class,
exports it with a fixed input and its one output, and checks the result:

    input   float32 [1, 3, 288, 800]   RGB, scaled to [0, 1], then
                                       (x - mean) / std with the ImageNet values
    output  float32 [1, 101, 56, 4]    100 column bins + "no lane",
                                       56 row anchors, 4 lanes

Usage:
    git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection ufld
    # tusimple_18.pth: the link in that repository's README
    scripts/export-lane-model.py --repo ufld --checkpoint tusimple_18.pth \
        --output models/ufld-tusimple18.onnx

Next to the ONNX file it writes <output>.json with where the model came
from: repository commit, checkpoint and ONNX checksums, tool versions and the
measured difference between PyTorch and ONNX Runtime. The repository's code
is MIT licensed; the checkpoint's own terms are not stated by the authors,
so the checkpoint and the ONNX file are not committed here.

Needs torch, onnx and onnxruntime; they are not needed to run the model.
"""

import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np

INPUT_SHAPE = (1, 3, 288, 800)
OUTPUT_SHAPE = (1, 101, 56, 4)
# Keys of the auxiliary segmentation head, used in training only
AUX_PREFIXES = ("aux_header2.", "aux_header3.", "aux_header4.", "aux_combine.")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(repo, checkpoint):
    import torch

    sys.path.insert(0, repo)
    from model.model import parsingNet  # the authors' class

    net = parsingNet(pretrained=False, backbone="18", cls_dim=OUTPUT_SHAPE[1:],
                     use_aux=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    # Saved from DataParallel: every key starts with "module."
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    unused = [k for k in state if k.startswith(AUX_PREFIXES)]
    for key in unused:
        del state[key]
    # strict: a missing or unknown weight is an error, not a silent default
    net.load_state_dict(state, strict=True)
    return net.eval(), len(unused)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", required=True,
                        help="clone of cfzd/Ultra-Fast-Lane-Detection")
    parser.add_argument("--checkpoint", required=True, help="tusimple_18.pth")
    parser.add_argument("--output", default="models/ufld-tusimple18.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import onnx
    import onnxruntime
    import torch

    net, unused = load_model(args.repo, args.checkpoint)
    example = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    torch.onnx.export(net, example, args.output, input_names=["image"],
                      output_names=["logits"], opset_version=args.opset,
                      dynamo=False)

    model = onnx.load(args.output)
    onnx.checker.check_model(model)

    def dims(value):
        return tuple(d.dim_value for d in value.type.tensor_type.shape.dim)

    inputs = [i for i in model.graph.input
              if i.name not in {w.name for w in model.graph.initializer}]
    assert len(inputs) == 1 and len(model.graph.output) == 1, "one input, one output"
    assert dims(inputs[0]) == INPUT_SHAPE, dims(inputs[0])
    assert dims(model.graph.output[0]) == OUTPUT_SHAPE, dims(model.graph.output[0])
    for value in (inputs[0], model.graph.output[0]):
        assert value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    # Same tensors through PyTorch and ONNX Runtime on the CPU
    session = onnxruntime.InferenceSession(args.output,
                                           providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(4):
        x = rng.standard_normal(INPUT_SHAPE).astype(np.float32)
        with torch.no_grad():
            expected = net(torch.from_numpy(x)).numpy()
        actual = session.run(None, {"image": x})[0]
        assert actual.shape == OUTPUT_SHAPE and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
        worst = max(worst, float(np.abs(actual - expected).max()))

    commit = subprocess.run(["git", "-C", args.repo, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    info = {
        "model": "Ultra Fast Lane Detection, ResNet-18, TuSimple, use_aux=False",
        "repository": "https://github.com/cfzd/Ultra-Fast-Lane-Detection",
        "repository_commit": commit,
        "repository_license": "MIT",
        "checkpoint_license": "not stated by the authors",
        "checkpoint_sha256": sha256(args.checkpoint),
        "onnx_sha256": sha256(args.output),
        "auxiliary_head_weights_dropped": unused,
        "input": {"name": "image", "shape": INPUT_SHAPE, "dtype": "float32"},
        "output": {"name": "logits", "shape": OUTPUT_SHAPE, "dtype": "float32"},
        "opset": args.opset,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "pytorch_vs_onnxruntime_max_abs_difference": worst,
    }
    with open(args.output + ".json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
