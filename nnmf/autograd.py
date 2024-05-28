import torch

class FunctionalNNMFLinear(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        h: torch.Tensor,
        number_of_iterations: int,
        reconstruct_fun: callable,
        forward_fun: callable,
    ) -> torch.Tensor:

        for _ in range(number_of_iterations):
            reconstruction = reconstruct_fun(h, weight=weight)
            h *= forward_fun((input / (reconstruction + 1e-20)), weight=weight)
            torch.nn.functional.normalize(h, dim=-1, p=1, out=h, eps=1e-20)

        ctx.save_for_backward(
            input,
            weight,
            h,
        )

        return h

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore
        (
            input,
            weight,
            output,
        ) = ctx.saved_tensors

        grad_input = grad_weights = None

        backprop_r: torch.Tensor = weight.unsqueeze(0) * output.unsqueeze(-1)
        backprop_bigr: torch.Tensor = backprop_r.sum(dim=1)
        backprop_z: torch.Tensor = backprop_r * (
            1.0 / (backprop_bigr.unsqueeze(1) + 1e-20)
        )

        grad_input = torch.bmm(grad_output.unsqueeze(1), backprop_z).squeeze(1)

        backprop_f: torch.Tensor = output.unsqueeze(2) * (
            input / (backprop_bigr**2 + 1e-20)
        ).unsqueeze(1)

        result_omega: torch.Tensor = backprop_bigr.unsqueeze(1) * grad_output.unsqueeze(
            -1
        )
        result_omega -= torch.bmm(grad_output.unsqueeze(1), backprop_r)
        result_omega *= backprop_f

        grad_weights = result_omega.sum(0)

        return grad_input, grad_weights, None, None, None, None


class FunctionalNNMFConv2d(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        h: torch.Tensor,
        number_of_iterations: int,
        reconstruct_fun: callable,
        forward_fun: callable,
        convolution_contribution_map: torch.Tensor,
        stride: int,
        padding: int,
    ) -> torch.Tensor:

        input *= convolution_contribution_map
        for _ in range(number_of_iterations):
            reconstruction = reconstruct_fun(h, weight=weight)
            h *= forward_fun((input / (reconstruction + 1e-20)), weight=weight)
            torch.nn.functional.normalize(h, dim=-1, p=1, out=h, eps=1e-20)

        ctx.save_for_backward(
            input,
            weight,
            h,
            reconstruction,
            torch.tensor(stride),
            torch.tensor(padding),
        )

        return h

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore
        (
            input,
            weight,
            h,
            reconstruction,
            stride,
            padding,
        ) = ctx.saved_tensors

        positive_weights = weight # TODO: add exp weight
        grad_input = grad_weights = None

        # grad input (needed also for the grad weights)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            big_r: torch.Tensor = torch.nn.functional.conv_transpose2d(
                h,
                positive_weights,
                stride=stride.item(),
                padding=padding.item(),
                dilation=1,
            )
            factor_x_div_r: torch.Tensor = input / (big_r + 10e-20)
            # TODO: stride, padding, dilation
            grad_input = torch.nn.functional.conv_transpose2d(
                (h * grad_output),
                positive_weights,
                stride=stride.item(),
                padding=padding.item(),
                dilation=1,
            ) / (big_r + 10e-20)
            del big_r

        # grad weights
        if ctx.needs_input_grad[1]:
            grad_weight = - torch.nn.functional.conv2d(
                (factor_x_div_r * grad_input).movedim(0, 1), h.movedim(0, 1)
            )
            grad_weight += torch.nn.functional.conv2d(
                factor_x_div_r.movedim(0, 1), (h * grad_output).movedim(0, 1)
            )
            grad_weight = grad_weight.movedim(0, 1)
            # grad_weight *= positive_weights
            # grad_weight -= positive_weights * grad_weight.sum(dim=0, keepdim=True)

        return grad_input, grad_weights, None, None, None, None, None, None, None

