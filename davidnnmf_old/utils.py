import torch

def calculate_output_size(
    value: list[int],
    kernel_size: list[int],
    stride: list[int],
    dilation: list[int],
    padding: list[int],
) -> torch.Tensor:
    assert len(value) == 2
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert len(dilation) == 2
    assert len(padding) == 2

    return torch.tensor(
        [
            int(
                (
                    float(value[0])
                    + float(padding[0])
                    - float(dilation[0]) * (float(kernel_size[0]) - 1.0)
                    - 1.0
                )
                / float(stride[0])
                + 1.0
            ),
            int(
                (
                    float(value[1])
                    + float(padding[1])
                    - float(dilation[1]) * (float(kernel_size[1]) - 1.0)
                    - 1.0
                )
                / float(stride[1])
                + 1.0
            ),
        ],
        dtype=torch.int64,
    )
