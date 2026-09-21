"""Export v2 or legacy v1 checkpoints with an explicit preprocessing contract."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from training.models import build_model, read_checkpoint


def export_model(model_path, output_path=None, verify=True):
    checkpoint = read_checkpoint(model_path)
    config = checkpoint["config"]
    version = config["model_version"]
    if output_path is None:
        output_path = f"model/orientation_model{'_v2' if version == 'v2' else ''}.onnx"
    output = Path(output_path)
    if version != "v2" and output.name == "orientation_model_v2.onnx":
        raise ValueError("Refusing to label a legacy v1 checkpoint as v2")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.onnx")
    model = build_model(config["backbone"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    size = config["input_size"]
    sample = torch.randn(1, 3, size, size)
    torch.onnx.export(model, sample, str(temporary), export_params=True, opset_version=17,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      dynamo=False)
    exported = onnx.load(str(temporary))
    onnx.helper.set_model_props(exported, {
        "model_version": version, "backbone": config["backbone"],
        "input_size": str(size), "resize_mode": config["resize_mode"],
        "angles_ccw": json.dumps(config["angles_ccw"]),
        "normalization": "imagenet_rgb", "manifest_sha256": config.get("manifest_sha256", "legacy"),
    })
    onnx.checker.check_model(exported)
    onnx.save(exported, str(temporary))
    if verify:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(temporary), sess_options=options, providers=["CPUExecutionProvider"])
        for batch in (1, 2):
            inputs = torch.randn(batch, 3, size, size)
            with torch.inference_mode():
                expected = model(inputs).numpy()
            actual = session.run(None, {"input": inputs.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
        del session
        print("ONNX parity verified for batch sizes 1 and 2.")
    temporary.replace(output)
    print(f"Exported {version}: {output} ({output.stat().st_size / 1024**2:.1f} MiB)")
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--verify", action="store_true", help="Parity verification is always enabled.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    export_model(args.model_path, args.output_path, verify=True)


if __name__ == "__main__":
    main()
