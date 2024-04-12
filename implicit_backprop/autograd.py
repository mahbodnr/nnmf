import torch

SAFE_DIVISION_EPSILON = 1e-12

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
        # self.convergence = []
        # self.reconstruction_mse = []
        soft_weight = torch.nn.functional.softmax(weight, dim=1)
        for i in range(n_iterations):
            new_h, reconstruction = nnmf_iteration(input, soft_weight)
            # self.convergence.append(F.mse_loss(new_h, self.h))
            # self.reconstruction_mse.append(F.mse_loss(self.reconstruction, input))
            h = new_h
            # if (
            #     self.convergence_threshold > 0
            #     and self.convergence[-1] < self.convergence_threshold
            # ):
            #     break
        ctx.save_for_backward(input, weight, soft_weight, h, reconstruction)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: b, o
        """

        """
        input, weight, soft_weight, h, reconstruct = ctx.saved_tensors
        grad_input = grad_weight = grad_h = None
        reconstruct_square = reconstruct.square()

        A = div(soft_weight.unsqueeze(0), reconstruct.unsqueeze(1)) # B,o,i
        term1 = - div(input, reconstruct_square)
        term2 = soft_weight.t().unsqueeze(0) * term1.unsqueeze(-1) # B,i,o
        # B = torch.bmm(term2.transpose(1, 2).unsqueeze(-2).unsqueeze(-2), soft_weight.unsqueeze(0).unsqueeze(-1).squeeze(-1)).sum(-1).sum(-1) # B,i,i
        B = torch.einsum("bio,si->bos", term2, soft_weight) # B,o,o
        # B = - div(torch.einsum("bs,is,ks->bsik", input, soft_weight, soft_weight), reconstruct_square.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) # B,i,i
        delta_reconstruct = reconstruct.unsqueeze(-1).unsqueeze(-1) * torch.eye(h.shape[-1], device=reconstruct.device)
        C = div(input, reconstruct_square).unsqueeze(-1).unsqueeze(-1) * (delta_reconstruct - torch.einsum("oi,bj->bioj", soft_weight, h)) # B,i,o[weight],o[h]

        if ctx.needs_input_grad[0]:
            try:
                # dh/dx = B⁻¹ A
                dh_dx = torch.linalg.solve(B, A)
            except torch.linalg.LinAlgError:
                print("Singular matrix in input gradient")
                dh_dx = torch.linalg.solve(B + torch.eye(B.shape[-1], device=B.device).unsqueeze(0) * 1e-6, A)
            grad_input = torch.einsum("boi,bo->bi", dh_dx, grad_output)

        if ctx.needs_input_grad[1]:
            try:
                # dh/dw = B⁻¹ C
                dh_dw = torch.linalg.solve(B.unsqueeze(1), C).sum(-1) # B,i,o

            except torch.linalg.LinAlgError:
                print("Singular matrix in weight gradient")
                dh_dw = torch.linalg.solve(B.unsqueeze(1) + torch.eye(B.shape[-1], device=B.device).unsqueeze(0).unsqueeze(0) * 1e-6, C).sum(-1)
            grad_weight = - soft_weight * torch.einsum("bio,bo->i", dh_dw, grad_output).unsqueeze(0)

        return grad_input, grad_weight, grad_h, None, None