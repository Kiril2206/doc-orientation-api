import argparse
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Export Document Orientation Classifier to ONNX")
    parser.add_argument('--model-path', type=str, required=True, help="Path to best_model.pth")
    parser.add_argument('--output-path', type=str, default=os.path.join('model', 'orientation_model.onnx'), help="Path to save ONNX model")
    parser.add_argument('--verify', action='store_true', help="Verify ONNX model with onnxruntime")
    return parser.parse_args()

def export_model(model_path: str, output_path: str = os.path.join('model', 'orientation_model.onnx'), verify: bool = False):
    """Export trained PyTorch model to ONNX format."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    # Load Model
    print(f"Loading PyTorch model from {model_path}...")
    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Create dummy input: Dynamic batch size, 3 channels, 224x224
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

    # Export
    print(f"Exporting to ONNX: {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model saved to {output_path}! Size: {file_size_mb:.2f} MB")

    if verify:
        print("\nVerifying ONNX model...")
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime is not installed. Please install it to verify the model: pip install onnxruntime")
            return

        # Get PyTorch output
        with torch.no_grad():
            torch_out = model(dummy_input)

        # Get ONNX Runtime output
        ort_session = ort.InferenceSession(output_path)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
        ort_outs = ort_session.run(None, ort_inputs)

        # Compare
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Verification successful! PyTorch and ONNX Runtime outputs match within tolerance.")

        # Test dynamic batch size
        print("Testing dynamic batch size (batch=4)...")
        dummy_input_batch = torch.randn(4, 3, 224, 224)
        ort_inputs_batch = {ort_session.get_inputs()[0].name: to_numpy(dummy_input_batch)}
        ort_outs_batch = ort_session.run(None, ort_inputs_batch)
        print(f"Dynamic batch size test passed. Output shape: {ort_outs_batch[0].shape}")

def main():
    args = get_args()
    export_model(args.model_path, args.output_path, verify=args.verify)

if __name__ == "__main__":
    main()
