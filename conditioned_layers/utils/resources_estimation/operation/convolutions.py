from utils.resources_estimation.operation.estimator import Estimator
from utils.resources_estimation.operation.basics import MatMul, Linear_Layer, PointToPointOperation, Batch_Norm, Relu

class Predicted_Conv(Estimator):
    def __init__(self, width, length, in_channels, out_channels, kernel_size, bias=False):
        computation_single_pixel = MatMul(1, kernel_size ** 2 * in_channels, out_channels).computation()
        computation_single_pixel = computation_single_pixel if not bias else computation_single_pixel + out_channels
        computation_whole_image = computation_single_pixel * width * length

        super(Predicted_Conv, self).__init__(0, computation_whole_image)

class Learned_Conv(Estimator):
    def __init__(self, width, length, in_channels, out_channels, kernel_size, bias=True):
        layer = Linear_Layer(kernel_size ** 2 * in_channels, out_channels, bias=bias)

        memory = layer.memory()
        computation = width * length * layer.computation()

        super(Learned_Conv, self).__init__(memory, computation)


class Predicted_Residual_Block(Estimator):
    def __init__(self, width, length, in_channels, out_channels, kernel_size, downsizing, bias=False):
        actual_width = width // 2 if downsizing else width
        actual_length = length // 2 if downsizing else length

        reslayer = Learned_Conv(actual_width, actual_length, in_channels, out_channels, 1)
        conv1 = Predicted_Conv(actual_width, actual_length, in_channels, out_channels, kernel_size, bias)
        conv2 = Predicted_Conv(actual_width, actual_length, out_channels, out_channels, kernel_size, bias)
        bn = Batch_Norm(actual_width * actual_length * out_channels)
        relu = Relu(actual_width * actual_length * out_channels)
        residual_sum = PointToPointOperation(actual_width * actual_length * out_channels)

        memory = conv1.memory() + conv2.memory() + 2 * relu.memory()\
                 + 2 * bn.memory() + residual_sum.memory()
        memory = memory + reslayer.memory() if downsizing else memory

        computation = conv1.computation() + conv2.computation() + 2 * relu.computation()\
                      + 2 * bn.computation() + residual_sum.computation()
        computation = computation + reslayer.computation() if downsizing else computation

        super(Predicted_Residual_Block, self).__init__(memory, computation)



