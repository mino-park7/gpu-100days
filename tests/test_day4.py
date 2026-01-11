import torch
from torchvision import io, transforms

from gpu_100days import gray_scale, grey_scale_triton


def test_grey_scale_triton():
    image = io.read_image("tests/image/puppy.png", mode=io.ImageReadMode.RGB)
    image = image.to(device="cuda", dtype=torch.uint8)
    out = grey_scale_triton(image).cpu()

    io.write_png(out, "tests/image/puppy_gray_scale_triton.png")

    gray_scaler = transforms.Grayscale()
    out_pytorch = gray_scaler(image).cpu()
    io.write_png(out_pytorch, "tests/image/puppy_gray_scale_pytorch.png")
    assert torch.allclose(out, out_pytorch, rtol=1e1, atol=1e1)


def test_gray_scale():
    image = io.read_image("tests/image/puppy.png", mode=io.ImageReadMode.RGB)
    image = image.to(device="cuda", dtype=torch.uint8)
    out = gray_scale(image).cpu()
    io.write_png(out, "tests/image/puppy_gray_scale_cuda.png")

    gray_scaler = transforms.Grayscale()
    out_pytorch = gray_scaler(image).cpu()
    io.write_png(out_pytorch, "tests/image/puppy_gray_scale_pytorch.png")
    assert torch.allclose(out, out_pytorch, rtol=1e1, atol=1e1)
