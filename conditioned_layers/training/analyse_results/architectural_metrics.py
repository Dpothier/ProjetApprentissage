

class PolicyCNN():
    def __init__(self, in_channels, out_classes, width, length, kernel_size):
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.width = width
        self.length = length
        self.kernel_size = kernel_size

    def count_parameters(self, lr, state_size, seed_size, t):
        return self.in_channels * state_size + \
               self.out_classes * state_size + \
               4 * state_size * seed_size + \
               3 * seed_size ** 2 + \
               6 * seed_size + \
               seed_size * state_size * self.kernel_size ** 2 + \
               state_size * self.kernel_size ** 2

    def count_operations(self, lr, state_size, seed_size, t):
        initial_layer = operations_CNN_layer(self.in_channels, 1, state_size, self.width, self.length)
        activation = operations_scalar_operation(self.width, self.length, state_size)
        state_calculation = operations_scalar_operation(self.width, self.length, state_size)

        seed_calculation = operations_stepwise_GRU(state_size, seed_size)
        layer_generation = operations_CNN_layer_generator(state_size, seed_size, self.kernel_size)
        conv_layer_calculation = operations_CNN_layer(state_size, self.kernel_size, state_size,
                                                      self.width, self.length,
                                                      uses_bias=False)

        classification_layer = operations_linear_layer(state_size, self.out_classes)

        return initial_layer + activation + state_calculation + \
               t * (seed_calculation + layer_generation + conv_layer_calculation + activation + state_calculation) + \
               classification_layer


class StandardCNN():
    def __init__(self, in_channels, out_classes, width, length, kernel_size):
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.width = width
        self.length = length
        self.kernel_size = kernel_size

    def count_parameters(self, lr, state_size, seed_size, t):
        return self.in_channels * state_size + \
               state_size * self.out_classes + \
               state_size + \
               self.out_classes + \
               t * (self.kernel_size ** 2 * state_size ** 2 + state_size)

    def count_operations(self, lr, state_size, seed_size, t):
        initial_layer = operations_CNN_layer(self.in_channels, 1, state_size, self.width, self.length)
        activation = operations_scalar_operation(self.width, self.length, state_size)

        conv_layer_calculation = operations_CNN_layer(state_size, self.kernel_size, state_size, self.width,
                                                      self.length)

        global_avg_pooling = operations_scalar_operation(self.width, self.length, state_size)
        classification_layer = operations_linear_layer(state_size, self.out_classes)

        return initial_layer + activation + \
               t * (conv_layer_calculation + activation) + \
               global_avg_pooling + classification_layer

class PolicyMLP():
    def __init__(self, in_size, out_classes):
        self.in_size = in_size
        self.out_classes = out_classes

    def count_parameters(self, lr , state_size, seed_size, t):
        return self.in_size * state_size + state_size + \
               4 * state_size * seed_size + \
               3 * seed_size ** 2 + \
               6 * seed_size + \
               self.out_classes * state_size

    def count_operations(self, lr, state_size, seed_size, t):
        initial_layer = operations_linear_layer(self.in_size, state_size)
        activation = operations_scalar_operation(state_size)

        seed_calculation = operations_stepwise_GRU(state_size, state_size)
        tanh_seed_activation = operations_tanh(state_size, state_size)
        layer_application = operations_linear_layer(state_size, state_size, use_bias=False)

        classification_layer = operations_linear_layer(state_size, self.out_classes)

        return initial_layer + activation + \
               t * (seed_calculation + tanh_seed_activation + layer_application + activation) + \
               classification_layer


class StandardMLP():
    def __init__(self, in_size, out_classes):
        self.in_size = in_size
        self.out_classes = out_classes

    def count_parameters(self, lr, state_size, seed_size, t):
        return self.in_size * state_size + \
               state_size + \
               t * (state_size ** 2 + state_size) + \
               self.out_classes * state_size

    def count_operations(self, lr, state_size, seed_size, t):
        initial_layer = operations_linear_layer(self.in_size, state_size)
        activation = operations_scalar_operation(state_size)

        mid_layers = operations_linear_layer(state_size, state_size)

        classification_layer = operations_linear_layer(state_size, self.out_classes)

        return initial_layer + activation + \
               t * (mid_layers + activation) + \
               classification_layer




def operations_CNN_layer(in_channels, kernel_size, out_channels, width, length, uses_bias=True):
    operation_from_weights = 2 *in_channels * kernel_size**2 * out_channels * width * length
    operations_from_bias = width * length * out_channels

    return operation_from_weights + operations_from_bias if uses_bias else operation_from_weights

# Batch does not refer to the number of instances,
# but other form of batches such as the number of state_channels in the GRU
def operations_linear_layer(in_size, out_size, batch=1, use_bias=True):
    operations_for_weight = 2 * batch * in_size * out_size
    operations_for_biases = batch * out_size

    return operations_for_weight + operations_for_biases if use_bias else operations_for_weight


def operations_stepwise_GRU(state_channels, seed_size):
    return 32 * state_channels * seed_size + \
           6 * seed_size**2 * state_channels

def operations_CNN_layer_generator(state_channels, seed_size, kernel_size):
    return operations_linear_layer(seed_size, kernel_size**2 * state_channels, state_channels) +\
           operations_tanh(kernel_size**2 * state_channels, state_channels)

# An operation applied to every element of a tensor creates a number of operation equals to the number of elements in the tensor
# E.g: a point-to-point sum or multiplication
def operations_scalar_operation(*dimensions):
    produce = 1
    for dim in dimensions:
        produce *= dim
    return produce

def operations_tanh(*dimensions):
    return 4 * operations_scalar_operation(*dimensions)

def operations_sigmoid(*dimensions):
    return 4 * operations_scalar_operation(*dimensions)


