"""Export MoGe-2 (ViT-S, with normals) to ONNX for the browser.

Writes into --out (default ./models):

  moge-2-vits-normal-<version>-fp32.onnx   full fp32 export, kept for reference
  moge-2-vits-normal-<version>.onnx        the file the app loads: fp16 weights, fp32 compute

The shipped file stores every large weight as fp16 and casts it back to fp32
when the session loads. That halves the download without running any op in
half precision, so one file serves WebGPU and the WASM fallback alike and
accuracy stays at the fp32 level (the --verify-image report shows the gap).

Inputs: `image` (1, 3, H, W) in [0, 1] and `num_tokens` (int64 scalar).
Outputs, the raw forward pass: `points` (1, H, W, 3) affine point map,
`normal` (1, H, W, 3), `mask` (1, H, W) and `metric_scale` (1,). Focal and
shift recovery happen in TypeScript (src/geometry/moge-post.ts).

Setup, CPU is enough:

  python -m venv .venv && . .venv/bin/activate
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install "git+https://github.com/microsoft/MoGe.git" onnx onnxruntime onnxslim pillow
  python scripts/export-moge.py --verify-image some-photo.jpg

Then upload the shipped file and the ORT runtime to R2 (see README).
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["XFORMERS_DISABLED"] = "1"

import numpy as np
import torch

from moge.model.v2 import MoGeModel

OUTPUTS = ["points", "normal", "mask", "metric_scale"]


def export(model: MoGeModel, path: Path, opset: int) -> None:
    sample = torch.rand(1, 3, 518, 686)
    torch.onnx.export(
        model,
        (sample, torch.tensor(1800)),
        str(path),
        input_names=["image", "num_tokens"],
        output_names=OUTPUTS,
        dynamic_axes={"image": {2: "height", 3: "width"}},
        opset_version=opset,
        dynamo=False,
    )


def slim(path: Path) -> None:
    try:
        import onnxslim
    except ImportError:
        print("onnxslim not installed, skipping graph simplification")
        return
    import onnx

    onnx.save(onnxslim.slim(onnx.load(str(path))), str(path))


def fp16_weights(src: Path, dst: Path, min_elements: int = 1024) -> None:
    """Store large fp32 initializers as fp16, each followed by a Cast back to fp32."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(src))
    graph = model.graph
    casts, keep = [], []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims) < min_elements:
            keep.append(init)
            continue
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
        keep.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name], name=init.name + "__cast", to=TensorProto.FLOAT))
    del graph.initializer[:]
    graph.initializer.extend(keep)
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def load_image(path: Path, num_tokens: int) -> np.ndarray:
    """Resize like the app: height on the token grid, width keeps the aspect ratio."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    base_h = round((num_tokens / (w / h)) ** 0.5)
    height = base_h * 14
    width = round(height * w / h)
    img = img.resize((width, height), Image.LANCZOS)
    return (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]


def verify(model: MoGeModel, files: list[Path], image: Path, num_tokens: int) -> None:
    import onnxruntime as ort

    x = load_image(image, num_tokens)
    with torch.no_grad():
        ref = {k: v.numpy() for k, v in model(torch.from_numpy(x), num_tokens).items()}
    valid = ref["mask"][0] > 0.5
    for f in files:
        sess = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
        feed = {"image": x, "num_tokens": np.array(num_tokens, dtype=np.int64)}
        out = dict(zip(OUTPUTS, sess.run(None, feed)))
        z_ref, z = ref["points"][0, ..., 2][valid], out["points"][0, ..., 2][valid]
        rel = float(np.median(np.abs(z - z_ref) / np.abs(z_ref)))
        cos = np.clip((out["normal"][0][valid] * ref["normal"][0][valid]).sum(-1), -1, 1)
        ang = float(np.degrees(np.median(np.arccos(cos))))
        agree = float(((out["mask"][0] > 0.5) == valid).mean())
        scale = float(out["metric_scale"][0] / ref["metric_scale"][0])
        print(f"{f.name}: z rel err {rel:.2e}, normal err {ang:.3f} deg, mask agree {agree:.4f}, scale ratio {scale:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Ruicheng/moge-2-vits-normal")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify-image", type=Path, help="compare ONNX outputs against PyTorch on this image")
    ap.add_argument("--num-tokens", type=int, default=1800)
    ap.add_argument("--reuse-fp32", action="store_true", help="skip the export if the fp32 file exists")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fp32 = args.out / f"moge-2-vits-normal-{args.version}-fp32.onnx"
    shipped = args.out / f"moge-2-vits-normal-{args.version}.onnx"

    model = MoGeModel.from_pretrained(args.model).eval()
    model.onnx_compatible_mode = True

    if not (args.reuse_fp32 and fp32.exists()):
        print(f"exporting {fp32}")
        export(model, fp32, args.opset)
        slim(fp32)
    print(f"writing {shipped}")
    fp16_weights(fp32, shipped)
    for f in (fp32, shipped):
        print(f"{f.name}: {f.stat().st_size / 2**20:.1f} MiB")

    if args.verify_image:
        verify(model, [fp32, shipped], args.verify_image, args.num_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
