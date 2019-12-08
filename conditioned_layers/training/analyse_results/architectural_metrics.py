

def number_of_parameters_policy_CNN(in_channels, state_channels, seed_size, out_classes, kernel_size):
    return in_channel * state_channels + \
           out_classes * state_channels + \
           4 * state_channels * seed_size + \
           3 * seed_size**2 + \
           6 * seed_size + \
           seed_size * state_channels * kernel_size**2 + \
           state_channels * kernel_size**2

def number_of_parameters_standard_CNN(in_channels, state_channels, depth, out_classes, kernel_size):
    return in_channels * state_channels + \
           state_channels * out_classes + \
           state_channels + \
           out_classes + \
           depth * (kernel_size**2 * state_channels**2 + state_channels)

def number_of_parameters_policy_MLP(in_size, state_size, seed_size, class_number):
    return in_size * state_size + state_size + \
           4 * state_size * seed_size + \
           3 * seed_size**2 + \
           6 * seed_size +\
           class_number * state_size

def number_of_parameters_standard_MLP(in_size, state_size, depth, class_number)
    return in_size * state_size + \
           state_size + \
           depth * (state_size ^ 2 + state_size) + \
           class_number * state_size

def number_of_operations_policy_CNN(in_channel, state_channels, seed_size, width, length,kernel_size, T, number_of_classes):
    initial_layer = operations_CNN_layer(in_channel, 1, state_channels, width, length)
    activation = operations_scalar_operation(width, height, state_channels)
    state_calculation = operations_scalar_operation(width, height, state_channels)

    seed_calculation = operations_stepwise_GRU(state_channels, seed_size)
    layer_generation = operations_CNN_layer_generator(state_channels,seed_size, kernel_size)
    conv_layer_calculation = operations_CNN_layer(state_channels, kernel_size,state_channels, width, length, uses_bias=False )

    classification_layer = operations_linear_layer(state_channels, number_of_classes)

    return initial_layer + activation + state_calculation + \
           T * (seed_calculation + layer_generation + conv_layer_calculation + activation + state_calculation) + \
           classification_layer

def number_of_operations_standard_CNN(in_channel, state_channels, width, length,kernel_size, T, number_of_classes):
    initial_layer = operations_CNN_layer(in_channel, 1, state_channels, width, length)
    activation = operations_scalar_operation(width, height, state_channels)

    conv_layer_calculation = operations_CNN_layer(state_channels, kernel_size,state_channels, width, length)

    global_avg_pooling = operations_scalar_operation(width, height, state_channels)
    classification_layer = operations_linear_layer(state_channels, number_of_classes)

    return initial_layer + activation + \
           T * (conv_layer_calculation + activation) + \
            global_avg_pooling + classification_layer


def number_of_operations_policy_MLP(in_size, state_size, T, number_of_classes):
    initial_layer = operations_linear_layer(in_size, state_size)
    activation = operations_scalar_operation(width, height, state_channels)

    seed_calculation = operations_stepwise_GRU(state_size, state_size)
    tanh_seed_activation = operations_tanh(state_size, state_size)
    layer_application = operations_linear_layer(state_size, state_size, use_bias=False)

    classification_layer = operations_linear_layer(state_size, number_of_classes)

    return initial_layer + activation + \
           T * (seed_calculation + tanh_seed_activation + layer_application + activation) + \
           classification_layer


def number_of_operations_standard_MLP(in_size, state_size, T, number_of_classes):
    initial_layer = operations_linear_layer(in_size, state_size)
    activation = operations_scalar_operation(width, height, state_channels)

    mid_layers = operations_linear_layer(state_size, state_size)

    classification_layer = operations_linear_layer(state_size, number_of_classes)

    return initial_layer + activation + \
           T * (mid_layers + activation) + \
           classification_layer

def operations_CNN_layer(in_channels, kernel_size, out_channels, width, length, uses_bias=True):
    operation_from_weights = 2 *in_channels * kernel_size**2 * out_channels * width * length
    operations_from_bias = width * lenght * out_channels

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
           operation_tanh(kernel_size**2 * state_channels, state_channels)

# An operation applied to every element of a tensor creates a number of operation equals to the number of elements in the tensor
# E.g: a point-to-point sum or multiplication
def operations_scalar_operation(*dimensions):
    produce = 1
    for dim in dimensions:
        produce *= dim
    return produce

def operations_tanh(*dimensions):
    return 4 * operations_scalar_operation(dimensions)

def operations_sigmoid(*dimensions):
    return 4 * operations_scalar_operation(dimensions)


