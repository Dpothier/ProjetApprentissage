import torch
import torch.autograd as autograd

class GeneratedLinearFunction(autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, state, generator, bias_generator, input, generator_bias=None, bias_generator_bias =None):
        ctx.save_for_backward(state, generator, input, generator_bias)

        layer_input_dim = input.shape[-1]
        batch_size = input.shape[0]

        layer = state.mm(generator.t())
        bias = state.m(bias_generator.t())
        if generator_bias is not None:
            layer += generator_bias.unsqueeze(0).expand_as(layer)

        if bias_generator_bias is not None:
            bias += generator_bias.unsqueeze(0).expand_as(layer)

        layer = layer.view(batch_size, layer_input_dim, -1)

        output = input.matmul(layer.permute(0, 2, 1))
        if bias_generator is not None:
            output += bias.unsqueeze(0).expand_as(layer)

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


if __name__ == "__main__":
    GeneratedLinear = GeneratedLinearFunction.apply

    d_state = 5
    d_model = 4

    state = torch.randn(d_state).unsqueeze(0)
    input = torch.randn(d_model).unsqueeze(0)

    generator = torch.randn(d_state, d_model * d_model)
    bias_generator = torch.randn(d_state, d_model)

    generator_bias = torch.randn(d_model * d_model)
    bias_generator_bias = torch.randn(d_model)


