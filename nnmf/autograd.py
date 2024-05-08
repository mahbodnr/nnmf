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
            reconstruction = reconstruct_fun(h, input=input, weight=weight)
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


# class FunctionalNNMFConv2d(torch.autograd.Function):
#     @staticmethod
#     def forward(  # type: ignore
#         ctx,
#         input: torch.Tensor,
#         weight: torch.Tensor,
#         output_size: torch.Tensor,
#         convolution_contribution_map: torch.Tensor,
#         iterations: int,
#         stride: tuple[int, int],
#         padding: str | tuple[int, int],
#         dilation: tuple[int, int],
#         beta: float | None,
#     ) -> torch.Tensor:

#         # Prepare h
#         output_size[0] = input.shape[0]
#         h = torch.full(
#             output_size.tolist(),
#             1.0 / float(output_size[1]),
#             device=input.device,
#             dtype=input.dtype,
#         )

#         # Prepare the non-negative version of the weigths
#         if beta is not None:
#             positive_weights = torch.exp(beta * weight)
#         else:
#             positive_weights = torch.exp(weight)

#         positive_weights /= positive_weights.sum((1, 2, 3), keepdim=True)

#         # Prepare input
#         input /= input.sum((1, 2, 3), keepdim=True) + 10e-20
#         input *= convolution_contribution_map

#         for _ in range(0, iterations):
#             # TODO: stride, padding, dilation
#             factor_x_div_r: torch.Tensor = (
#                 input
#                 / torch.nn.functional.conv_transpose2d(
#                     h,
#                     positive_weights,
#                     stride=1,
#                     padding=0,
#                     dilation=1,
#                 )
#                 + 10e-20
#             )

#             h *= torch.nn.functional.conv2d(
#                 factor_x_div_r,
#                 positive_weights,
#                 stride=stride,
#                 padding=padding,
#                 dilation=dilation,
#             )

#             h /= h.sum(1, keepdim=True) + 10e-20

#         # ###########################################################
#         # Save the necessary data for the backward pass
#         # ###########################################################
#         ctx.save_for_backward(
#             input,
#             positive_weights,
#             h,
#         )

#         ctx.beta = beta

#         return h

#     @staticmethod
#     @torch.autograd.function.once_differentiable
#     def backward(ctx, grad_output: torch.Tensor) -> tuple[  # type: ignore
#         torch.Tensor | None,
#         torch.Tensor | None,
#         None,
#         None,
#         None,
#         None,
#         None,
#         None,
#         None,
#     ]:

#         # ##############################################
#         # Default values
#         # ##############################################
#         grad_input: torch.Tensor | None = None
#         grad_weight: torch.Tensor | None = None
#         grad_output_size: None = None
#         grad_convolution_contribution_map: None = None
#         grad_iterations: None = None
#         grad_stride: None = None
#         grad_padding: None = None
#         grad_dilation: None = None
#         grad_beta: None = None

#         # ##############################################
#         # Get the variables back
#         # ##############################################
#         (
#             input,
#             positive_weights,
#             h,
#         ) = ctx.saved_tensors

#         big_r: torch.Tensor = torch.nn.functional.conv_transpose2d(
#             h,
#             positive_weights,
#             stride=1,
#             padding=0,
#             dilation=1,
#         )

#         factor_x_div_r: torch.Tensor = input / (big_r + 10e-20)

#         # TODO: stride, padding, dilation
#         grad_input = torch.nn.functional.conv_transpose2d(
#             (h * grad_output),
#             positive_weights,
#             stride=1,
#             padding=0,
#             dilation=1,
#         ) / (big_r + 10e-20)

#         del big_r

#         grad_weight = torch.nn.functional.conv2d(
#             (factor_x_div_r * grad_input).movedim(0, 1), h.movedim(0, 1)
#         )
#         grad_weight += torch.nn.functional.conv2d(
#             factor_x_div_r.movedim(0, 1), (h * grad_output).movedim(0, 1)
#         )
#         grad_weight = grad_weight.movedim(0, 1)

#         # Reverse the positive_weights transformation for the weight gradient
#         # exp(u_{si})/sum_k exp(u_{sk})
#         # d w_{si} / d u_{sj} = w_{si} (delta_{ij} - w_{sj})
#         # \Delta w^{not positive}_{sj} = \sum_i \Delta w^{positive}_{si} w_{si} (delta_{ij} - w_{sj})
#         # = \sum_i \Delta w^{positive}_{si} w_{si} (delta_{ij} - w_{sj})
#         # \Delta v^{positive}_{si} = \Delta w^{positive}_{si} w_{si}
#         grad_weight *= positive_weights
#         # \Delta w^{not positive}_{sj} = \Delta v^{positive}_{sj} - w_{si} \sum_i \Delta v^{positive}_{si}
#         grad_weight -= positive_weights * grad_weight.sum(dim=0, keepdim=True)
#         # If you really want to, you have to multiply it with beta
#         # grad_weight *= ctx.beta

#         return (
#             grad_input,
#             grad_weight,
#             grad_output_size,
#             grad_convolution_contribution_map,
#             grad_iterations,
#             grad_stride,
#             grad_padding,
#             grad_dilation,
#             grad_beta,
#         )




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
        )

        return h

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore
        (
            input,
            weight,
            h,
            reconstruction,
        ) = ctx.saved_tensors

        positive_weights = weight # TODO: add exp weight
        grad_input = grad_weights = None

        # grad input (needed also for the grad weights)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            big_r: torch.Tensor = torch.nn.functional.conv_transpose2d(
                h,
                positive_weights,
                stride=1,
                padding=0,
                dilation=1,
            )
            factor_x_div_r: torch.Tensor = input / (big_r + 10e-20)
            # TODO: stride, padding, dilation
            grad_input = torch.nn.functional.conv_transpose2d(
                (h * grad_output),
                positive_weights,
                stride=1,
                padding=0,
                dilation=1,
            ) / (big_r + 10e-20)
            del big_r

        # grad weights
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv2d(
                (factor_x_div_r * grad_input).movedim(0, 1), h.movedim(0, 1)
            )
            grad_weight += torch.nn.functional.conv2d(
                factor_x_div_r.movedim(0, 1), (h * grad_output).movedim(0, 1)
            )
            grad_weight = grad_weight.movedim(0, 1)
            grad_weight *= positive_weights
            grad_weight -= positive_weights * grad_weight.sum(dim=0, keepdim=True)

        return grad_input, grad_weights, None, None, None, None, None

