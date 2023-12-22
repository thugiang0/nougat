import torch.onnx

from nougat.model import SwinEncoder

swin_encoder = SwinEncoder(
    input_size=[896, 672],
    align_long_axis=False,
    window_size=7,
    encoder_layer=[2, 2, 14, 2],
    patch_size=4,
    embed_dim=128,
    num_heads=[4, 8, 16, 32],
)
swin_encoder.eval()
swin_encoder.cuda()

sample_input = torch.randn(1, 3, 896, 672).cuda()  

onnx_path = "swin_encoder_2.onnx"
torch.onnx.export(
    swin_encoder,
    sample_input,
    onnx_path,
    verbose=True,
    input_names=["input"],
    output_names=["output"],
    opset_version=13
)

print("swin_encoder.onnx")