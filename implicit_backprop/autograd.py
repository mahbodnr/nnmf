import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

SAFE_DIVISION_EPSILON = 1e-20


def div(a, b):
    return a / (b + SAFE_DIVISION_EPSILON)


class ImplicitGradient(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        h: torch.Tensor,
        n_iterations: int,
        nnmf_iteration: callable,
    ):
        soft_weight = torch.nn.functional.softmax(weight, dim=1)
        for i in range(n_iterations):
            h, reconstruction, nnmf_update = nnmf_iteration(input, h, soft_weight)
        ctx.save_for_backward(
            input, weight, soft_weight, h, reconstruction, nnmf_update
        )
        return h

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: b, o
        """ """
        input, weight, soft_weight, h, reconstruct, nnmf_update = ctx.saved_tensors
        grad_input = grad_weight = grad_h = None
        reconstruct_square = reconstruct.square()
        # not_converged = nnmf_update < 0.95
        not_converged = h < 1e-3
        # not_converged = (h < 0)

        # print(not_converged.sum()/not_converged.numel())
        # plt.subplot(2, 1, 1)
        # plt.hist(nnmf_update.cpu().detach().numpy().flatten(), bins=100)
        # nnmf_update[not_converged] = 0
        # plt.subplot(2, 1, 2)
        # plt.hist(nnmf_update.cpu().detach().numpy().flatten(), bins=100)
        # plt.show()

        A = div(soft_weight.unsqueeze(0), reconstruct.unsqueeze(1))  # B,o,i
        term1 = div(input, reconstruct_square)
        term2 = soft_weight.t().unsqueeze(0) * term1.unsqueeze(-1)  # B,i,o
        B = -torch.einsum("bio,si->bso", term2, soft_weight)  # B,o,o #TODO bos VS bso
        # B = - term2.transpose(1, 2) @ soft_weight.t() # B,o,o
        delta_reconstruct = reconstruct.unsqueeze(-1).unsqueeze(-1) * torch.eye(
            h.shape[-1], device=reconstruct.device
        )
        C = term1.unsqueeze(-1).unsqueeze(-1) * (
            delta_reconstruct - torch.einsum("oi,bj->bioj", soft_weight, h)
        )  # B,i,o[weight(k)],o[h(i*)]
        # term3 = soft_weight.t().unsqueeze(-1).unsqueeze(0) * h.unsqueeze(1).unsqueeze(1)
        # C = term1.unsqueeze(-1).unsqueeze(-1) * (delta_reconstruct - term3) # B,i,o[weight(k)],o[h(i*)]
        # A
        A[not_converged] = 0
        # B
        B[not_converged] = 0
        B = B.permute(0, 2, 1)
        B[not_converged] = 0
        B = B.permute(0, 2, 1)
        # C
        # c: b, i, o1, o2
        C = C.permute(0, 3, 1, 2)
        # c: b, o2, i, o1
        C[not_converged] = 0
        C = C.permute(0, 2, 3, 1)
        # c: b, i, o1, o2

        B_inv = torch.linalg.pinv(B)
        if ctx.needs_input_grad[0]:
            try:
                # dh_dx = torch.linalg.solve(B, A)
                dh_dx = B_inv @ A
            except torch.linalg.LinAlgError:
                print("Singular matrix in input gradient")
                dh_dx = torch.linalg.solve(
                    B + torch.eye(B.shape[-1], device=B.device).unsqueeze(0) * 1e-6, A
                )

            grad_input = torch.einsum("boi,bo->bi", dh_dx, grad_output)
            # grad_input = (dh_dx * grad_output.unsqueeze(-1)).sum(dim=1)

        if ctx.needs_input_grad[1]:
            try:
                # dh_dw = torch.linalg.solve(B.unsqueeze(1), C) # B,i,o,o
                dh_dw = B_inv.unsqueeze(1) @ C  # B,i,o
            except torch.linalg.LinAlgError:
                print("Singular matrix in weight gradient")
                dh_dw = torch.linalg.solve(
                    B.unsqueeze(1)
                    + torch.eye(B.shape[-1], device=B.device).unsqueeze(0).unsqueeze(0)
                    * 1e-6,
                    C,
                )

            grad_weight = soft_weight * torch.einsum("biso,bo->si", dh_dw, grad_output)
            # grad_weight = soft_weight * (dh_dw * grad_output.unsqueeze(1).unsqueeze(1)).sum(dim=[0, -1]).t()

        plt.subplot(2, 1, 1)
        plt.hist(grad_weight.cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label="grad_weight")
        plt.legend()
        if ctx.needs_input_grad[0]:
            plt.subplot(2, 1, 2)
            plt.hist(grad_input.cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label="grad_input")
        plt.legend()
        plt.show()

        return grad_input, grad_weight, grad_h, None, None


# class ImplicitGradient(torch.autograd.Function):

#     @staticmethod
#     def forward(
#         ctx,
#         input: torch.Tensor,
#         weight: torch.Tensor,
#         h: torch.Tensor,
#         n_iterations: int,
#         nnmf_iteration: callable,
#         ):
#         soft_weight = torch.nn.functional.softmax(weight, dim=1)
#         for i in range(n_iterations):
#             h, reconstruction, nnmf_update = nnmf_iteration(input, h, weight)
#         ctx.save_for_backward(input, weight, h, reconstruction, nnmf_update)
#         return h

#     @staticmethod
#     def backward(ctx, grad_output):
#         # grad_output: b, o
#         """

#         """
#         input, weight, h, reconstruct, nnmf_update = ctx.saved_tensors
#         grad_input = grad_weight = grad_h = None
#         reconstruct_square = reconstruct.square()
#         # not_converged = nnmf_update < 0.95
#         not_converged = (h < 1e-3)
#         # not_converged = (h < 0)


#         A = div(weight.unsqueeze(0), reconstruct.unsqueeze(1)) # B,o,i
#         term1 = div(input, reconstruct_square)
#         term2 = weight.t().unsqueeze(0) * term1.unsqueeze(-1) # B,i,o
#         B = - torch.einsum("bio,si->bso", term2, weight) # B,o,o #TODO bos VS bso
#         delta_reconstruct = reconstruct.unsqueeze(-1).unsqueeze(-1) * torch.eye(h.shape[-1], device=reconstruct.device)
#         C = term1.unsqueeze(-1).unsqueeze(-1) * (delta_reconstruct - torch.einsum("oi,bj->bioj", weight, h)) # B,i,o[weight(k)],o[h(i*)]
#         # A
#         A[not_converged] = 0
#         # B
#         B[not_converged] = 0
#         B = B.permute(0, 2, 1)
#         B[not_converged] = 0
#         B = B.permute(0, 2, 1)
#         #C
#         # c: b, i, o1, o2
#         C = C.permute(0, 3, 1, 2)
#         # c: b, o2, i, o1
#         C[not_converged] = 0
#         C = C.permute(0, 2, 3, 1)
#         # c: b, i, o1, o2

#         B_inv = torch.linalg.pinv(B)
#         if ctx.needs_input_grad[0]:
#             try:
#                 dh_dx = B_inv @ A
#             except torch.linalg.LinAlgError:
#                 print("Singular matrix in input gradient")
#                 dh_dx = torch.linalg.solve(B + torch.eye(B.shape[-1], device=B.device).unsqueeze(0) * 1e-6, A)


#             grad_input = torch.einsum("boi,bo->bi", dh_dx, grad_output)
#             # print(f"grad_input min: {grad_input.min()}, max: {grad_input.max()}")


#         if ctx.needs_input_grad[1]:
#             try:
#                 dh_dw = B_inv.unsqueeze(1) @ C # B,i,o
#             except torch.linalg.LinAlgError:
#                 print("Singular matrix in weight gradient")
#                 dh_dw = torch.linalg.solve(B.unsqueeze(1) + torch.eye(B.shape[-1], device=B.device).unsqueeze(0).unsqueeze(0) * 1e-6, C)

#             grad_weight = weight * torch.einsum("biso,bo->si", dh_dw, grad_output)
#             # print(f"grad_weight min: {grad_weight.min()}, max: {grad_weight.max()}")

#         return grad_input, grad_weight, grad_h, None, None


class ImplicitGradient(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        h: torch.Tensor,
        n_iterations: int,
        nnmf_iteration: callable,
    ):
        # soft_weight = torch.nn.functional.softmax(weight, dim=1)
        soft_weight = weight
        for i in range(n_iterations):
            h, reconstruction, nnmf_update = nnmf_iteration(input, h, soft_weight)
        ctx.save_for_backward(
            input, weight, soft_weight, h, reconstruction, nnmf_update
        )
        return h

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: b, o
        """ """
        input, weight, soft_weight, h, reconstruct, nnmf_update = ctx.saved_tensors
        grad_input = grad_weight = grad_h = None
        B = (div(input, reconstruct.square()) @ soft_weight.t().square()).unsqueeze(-1)  # B, o, 1
        weight_h = h.unsqueeze(-1) * soft_weight.unsqueeze(0)  # B, o, i

        #
        # not_converged = nnmf_update < 0.95
        # not_converged = h < 1e-3
        # print(not_converged.sum() / not_converged.numel())
        # grad_output[not_converged] = 0

        if ctx.needs_input_grad[0]:
            dh_dx = div(soft_weight, (weight_h * B))

            grad_input = torch.einsum("boi,bo->bi", dh_dx , grad_output * h)
            # grad_input = torch.clamp(grad_input, -1e3, 1e3)
            # grad_input = (dh_dx * grad_output.unsqueeze(-1)).sum(dim=1)

        if ctx.needs_input_grad[1]:
            dh_dw = (
                div(input, reconstruct).unsqueeze(-2)
                * (
                    torch.eye(h.shape[-1], input.shape[-1], device=reconstruct.device)
                    - div(weight_h, reconstruct.unsqueeze(-2))
                )
                / B
            )

            grad_weight = - torch.einsum("boi,bo->oi", dh_dw, grad_output) * soft_weight
            # grad_weight = torch.clamp(grad_weight, -1e3, 1e3)


        # plt.subplot(3, 1, 1)
        # plt.hist(grad_output.cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label="grad_output")
        # plt.legend()
        # plt.subplot(3, 1, 2)
        # plt.hist(grad_weight.cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label="grad_weight")
        # plt.legend()
        # if ctx.needs_input_grad[0]:
        #     plt.subplot(3, 1, 3)
        #     plt.hist(grad_input.cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label="grad_input")
        # plt.legend()
        # plt.show()


        return grad_input, grad_weight, grad_h, None, None


class ImplicitGradient(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        h: torch.Tensor,
        n_iterations: int,
        nnmf_iteration: callable,
    ):
        # soft_weight = torch.nn.functional.softmax(weight, dim=1)
        soft_weight = weight
        for i in range(n_iterations):
            h, reconstruction, nnmf_update = nnmf_iteration(input, h, soft_weight)
        ctx.save_for_backward(
            input, weight, soft_weight, h, reconstruction, nnmf_update
        )
        return h

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: b, o
        """ """
        input, weight, soft_weight, h, reconstruct, nnmf_update = ctx.saved_tensors
        grad_input = grad_weight = grad_h = None
        weight_h = h.unsqueeze(-1) * soft_weight.unsqueeze(0)  # B, o, i

        if ctx.needs_input_grad[0]:
            dh_dx = div(weight_h, reconstruct.unsqueeze(-2)) - h.unsqueeze(-1)

            # grad_input = torch.einsum("boi,bo->bi", dh_dx , grad_output)
            grad_input = torch.bmm(grad_output.unsqueeze(-2), dh_dx).squeeze(-2)
            # grad_input = (dh_dx * grad_output.unsqueeze(-1)).sum(dim=1)

        if ctx.needs_input_grad[1]:
            dh_dw = (
                div(input, reconstruct).unsqueeze(-2)
                * (
                    torch.eye(h.shape[-1], input.shape[-1], device=reconstruct.device)
                    - div(weight_h, reconstruct.unsqueeze(-2))
                )
                # - (div(h, weight.sum(-1)).unsqueeze(-1) * weight)
            )

            grad_weight = - torch.einsum("boi,bo->oi", dh_dw, grad_output)# * soft_weight
            # grad_weight = - (dh_dw * grad_output.unsqueeze(-1)).sum(dim=0)

        return grad_input, grad_weight, grad_h, None, None



        # input, weight, soft_weight, output, reconstruct, nnmf_update = ctx.saved_tensors

        # backprop_r: torch.Tensor = weight.unsqueeze(0) * output.unsqueeze(-1)
        # backprop_bigr: torch.Tensor = backprop_r.sum(dim=1)
        # backprop_z: torch.Tensor = backprop_r * (
        #     1.0 / (backprop_bigr.unsqueeze(1) + 1e-20)
        # )

        # grad_input = torch.bmm(grad_output.unsqueeze(1), backprop_z).squeeze(1)

        # backprop_f: torch.Tensor = output.unsqueeze(2) * (
        #     input / (backprop_bigr**2 + 1e-20)
        # ).unsqueeze(1)

        # result_omega: torch.Tensor = backprop_bigr.unsqueeze(1) * grad_output.unsqueeze(
        #     -1
        # )
        # result_omega -= torch.bmm(grad_output.unsqueeze(1), backprop_r)
        # result_omega *= backprop_f

        # grad_weights = result_omega.sum(0)

        # return grad_input, grad_weights, None, None, None