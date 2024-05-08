import torch


class NNMFLinear(torch.nn.Module):
    weight: torch.nn.parameter.Parameter
    _number_of_neurons: int
    _number_of_input_neurons: int
    _input_size: list[int]
    _number_of_iterations: int
    _threshold_clamp: float

    def __init__(
        self,
        number_of_input_neurons: int,
        number_of_neurons: int,
        number_of_iterations: int,
        weight_noise_range: list[float] = [0.0, 1.0],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        threshold_clamp: float = 1e-5,  # Warning!!!! Needs to be divided by the number of H neurons
    ) -> None:
        super().__init__()

        self._number_of_input_neurons = int(number_of_input_neurons)
        self._number_of_neurons = int(number_of_neurons)

        self._number_of_iterations = int(number_of_iterations)
        self._threshold_clamp = threshold_clamp

        assert len(weight_noise_range) == 2
        weight = torch.empty(
            (
                int(self._number_of_neurons),
                int(self._number_of_input_neurons),
            ),
            dtype=dtype,
            device=device,
        )

        torch.nn.init.uniform_(
            weight,
            a=float(weight_noise_range[0]),
            b=float(weight_noise_range[1]),
        )
        self.weight = torch.nn.Parameter(weight)

        self.functional_nnmf_linear = FunctionalNNMFLinear.apply

    ####################################################################
    # Forward                                                          #
    ####################################################################

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:

        input = input.requires_grad_(True)
        input = torch.nn.functional.normalize(input, dim=-1, p=1, eps=1e-20)

        with torch.no_grad():
            torch.nn.functional.normalize(
                self.weight, dim=1, p=1, eps=1e-20, out=self.weight
            )
            torch.clamp(
                self.weight.data,
                min=float(self._threshold_clamp),
                max=None,
                out=self.weight.data,
            )
            torch.nn.functional.normalize(
                self.weight, dim=1, p=1, eps=1e-20, out=self.weight
            )

        h = self.functional_nnmf_linear(
            input,
            self.weight,
            self._number_of_iterations,
        )

        return h


class FunctionalNNMFLinear(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        number_of_iterations: int,
    ) -> torch.Tensor:

        h = torch.full(
            (input.shape[0], weight.shape[0]),
            float(1.0 / weight.shape[0]),
            device=weight.device,
            dtype=weight.dtype,
        )

        for _ in range(0, number_of_iterations):
            reconstruction = torch.nn.functional.linear(h, weight.T)
            reconstruction += 1e-20
            h *= torch.nn.functional.linear((input / reconstruction), weight)
            torch.nn.functional.normalize(h, dim=-1, p=1, out=h, eps=1e-20)

        ctx.save_for_backward(
            input,
            weight,
            h,
        )

        return h

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore

        # ##############################################
        # Get the variables back
        # ##############################################
        (
            input,
            weight,
            output,
        ) = ctx.saved_tensors

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

        return (grad_input, grad_weights, None)
