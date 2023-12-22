import onnxruntime as ort
import onnx
import torch

model_path = "swin_encoder.onnx"

session = ort.InferenceSession(model_path)

sample_input = torch.randn(1, 3, 896, 672)

ort_inputs = {'input': sample_input.cpu().numpy()}
ort_outs = session.run(None, ort_inputs)

print(ort_outs[0].shape)

