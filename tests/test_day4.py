from pathlib import Path

import torch
from torchvision import io, transforms

from gpu_100days import gray_scale, grey_scale_triton


def test_grey_scale_triton():
    pwd = Path(__file__).parent
    image = io.read_image(str(pwd / "image" / "puppy.png"), mode=io.ImageReadMode.RGB)
    image = image.to(device="cuda", dtype=torch.uint8)
    out = grey_scale_triton(image).cpu()
    out_path = pwd / "image" / "puppy_gray_scale_triton.png"
    io.write_png(out, str(out_path))

    gray_scaler = transforms.Grayscale()
    out_pytorch = gray_scaler(image).cpu()
    out_pytorch_path = pwd / "image" / "puppy_gray_scale_pytorch.png"
    io.write_png(out_pytorch, str(out_pytorch_path))
    assert torch.allclose(out, out_pytorch, rtol=1e1, atol=1e1)


def test_gray_scale():
    pwd = Path(__file__).parent
    image = io.read_image(str(pwd / "image" / "puppy.png"), mode=io.ImageReadMode.RGB)
    image = image.to(device="cuda", dtype=torch.uint8)
    out = gray_scale(image).cpu()
    out_path = pwd / "image" / "puppy_gray_scale_cuda.png"
    io.write_png(out, str(out_path))

    gray_scaler = transforms.Grayscale()
    out_pytorch = gray_scaler(image).cpu()
    out_pytorch_path = pwd / "image" / "puppy_gray_scale_pytorch.png"
    io.write_png(out_pytorch, str(out_pytorch_path))
    assert torch.allclose(out, out_pytorch, rtol=1e1, atol=1e1)
