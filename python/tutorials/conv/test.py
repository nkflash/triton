import torch
import triton
import conv

def bench_op(
    # Tensor dimensions
    BATCH,
    IN_C,
    IN_H,
    IN_W,
    KERNEL_N,
    KERNEL_H,
    KERNEL_W,
    # provider
    provider,
    # parameters of conv
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    dtype=torch.float32,
    layout="nhwc",
    warmup=25,
    rep=75,
):

    # allocate inputs, nchw
    x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype, device="cuda")
    w = torch.randn(
        (KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W), dtype=dtype, device="cuda"
    )
    bias = torch.randn((KERNEL_N), dtype=dtype, device="cuda")
    if layout == "nhwc":
        x = x.to(memory_format=torch.channels_last)
        w = w.to(memory_format=torch.channels_last)
    OUT_H = (
        IN_H + 2 * padding[0] - dilation[0] * (KERNEL_H - 1) - 1 + stride[0]
    ) // stride[0]
    OUT_W = (
        IN_W + 2 * padding[1] - dilation[1] * (KERNEL_W - 1) - 1 + stride[1]
    ) // stride[1]

    tflops = (
        lambda ms: 2.0
        * BATCH
        * OUT_H
        * OUT_W
        * IN_C
        * KERNEL_H
        * KERNEL_W
        * KERNEL_N
        / ms
        * 1e-9
    )
    if provider == "cublas":

        def fn():
            return torch.conv2d(x, w, bias, stride, padding, dilation, groups)

    elif provider == "triton":

        def fn():
            return conv.conv2d(
                x, w, bias, stride, padding, dilation, False, (0, 0), groups
            )

    fn()

bench_op(1, 3, 224, 224, 64, 7, 7, "triton")
