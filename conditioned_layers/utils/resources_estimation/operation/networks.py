from utils.resources_estimation.operation.estimator import Estimator
from utils.resources_estimation.operation.convolutions import Learned_Conv, Predicted_Residual_Block
from utils.resources_estimation.operation.basics import Relu, Batch_Norm, Vector_Table, Linear_Layer, GlobalAvgPool2D
from utils.resources_estimation.operation.RNN import GRU_Cell_State_Update, LSTM_Cell_State_Update

#This estimator assumes that dimension reduction occurs at regular interval
class Static_Hypernetwork(Estimator):
    def __init__(self, embedding_size, base_channel_count, initial_image_height, initial_image_width, depth, reduction_step_count, class_count):
        kernel_size = 3
        first_conv = Learned_Conv(initial_image_height, initial_image_width,3, base_channel_count, kernel_size, bias=True)
        first_conv_bn = Batch_Norm(initial_image_width * initial_image_height * base_channel_count)
        first_conv_relu = Relu(initial_image_width * initial_image_height * base_channel_count)

        policy_first_layer = Linear_Layer(embedding_size, base_channel_count * embedding_size)
        policy_second_layer = Linear_Layer(embedding_size, base_channel_count * kernel_size ** 2)

        layers_per_reduction_step = depth // reduction_step_count
        embedding_counts_total = 0
        residual_blocks = []
        policy_computation = 0
        for i in range(depth):
            reduction_step = i // layers_per_reduction_step
            downsize_layer = depth % reduction_step_count == 0 and reduction_step != 0
            if downsize_layer:
                in_channels = base_channel_count * (2 ** (reduction_step - 1))
                out_channels = base_channel_count * (2 ** reduction_step)
                image_height = initial_image_height // (2 ** (reduction_step - 1))
                image_width = initial_image_width // (2 ** (reduction_step - 1))
                embedding_count_conv1 = (2 ** reduction_step-1) * (2 ** reduction_step)
                embedding_count_conv2 = (2 ** reduction_step) ** 2
            else:
                in_channels = base_channel_count * (2 ** reduction_step)
                out_channels = base_channel_count * (2 ** reduction_step)
                image_height = initial_image_height // (2 ** reduction_step)
                image_width = initial_image_width // (2 ** reduction_step)
                embedding_count_conv1 = (2 ** reduction_step) ** 2
                embedding_count_conv2 = (2 ** reduction_step) ** 2

            embedding_counts_total += embedding_count_conv1 + embedding_count_conv2
            policy_computation += embedding_count_conv1 *\
                                  (policy_first_layer.computation() + base_channel_count * policy_second_layer.computation())
            policy_computation += embedding_count_conv2 *\
                                  (policy_first_layer.computation() + base_channel_count * policy_second_layer.computation())

            residual_blocks.append(Predicted_Residual_Block(image_width, image_height, in_channels, out_channels, kernel_size, downsize_layer, bias=False))

        embeddings = Vector_Table(embedding_size, embedding_counts_total)

        final_width = initial_image_width / 2 ** (reduction_step_count - 1)
        final_heigth = initial_image_height / 2 ** (reduction_step_count - 1)
        final_channels = base_channel_count * 2 ** (reduction_step_count - 1)
        global_average_pooling = GlobalAvgPool2D(final_width, final_heigth, final_channels)
        classification_layer = Linear_Layer(final_channels, class_count)

        memory = first_conv.memory() + first_conv_bn.memory() + first_conv_relu.memory() +\
                 embeddings.memory() + policy_first_layer.memory() + policy_second_layer.memory() + \
                 sum([layer.memory() for layer in residual_blocks]) + \
                 global_average_pooling.memory() + classification_layer.memory()

        computation = first_conv.computation() + first_conv_bn.computation() + first_conv_relu.computation() +\
                      embeddings.computation() + policy_computation +\
                      sum([layer.computation() for layer in residual_blocks]) +\
                      global_average_pooling.computation() + classification_layer.computation()

        super(Static_Hypernetwork, self).__init__({
            "first_conv": first_conv.memory() + first_conv_bn.memory() + first_conv_relu.memory(),
            "embeddings": embeddings.memory(),
            "policy": policy_first_layer.memory() + policy_second_layer.memory(),
            "layer": sum([layer.memory() for layer in residual_blocks]),
            "classification": global_average_pooling.memory() + classification_layer.memory(),
            "total": memory
        }, {
            "first_conv": first_conv.computation() + first_conv_bn.computation() + first_conv_relu.computation(),
            "embeddings": embeddings.computation(),
            "policy": policy_computation,
            "layer": sum([layer.computation() for layer in residual_blocks]),
            "classification": global_average_pooling.computation() + classification_layer.computation(),
            "total": computation
        })


class Policy_Hypernetwork(Estimator):
    def __init__(self, input_height, input_width, network_depth, reduction_step_count, embeddings_factors_size, embeddings_factors_count,
                 channels_factors_size, channels_factors_count, class_count, state_update_cell, state_update_depth):
        embeddings_size = embeddings_factors_size * embeddings_factors_count
        channels_count = channels_factors_size * channels_factors_count

        kernel_size = 3
        number_of_layers_in_residual_blocks = 2
        first_conv = Learned_Conv(input_height, input_width, 3, channels_count, kernel_size, bias=True)
        first_conv_bn = Batch_Norm(input_width * input_height * channels_count)
        first_conv_relu = Relu(input_width * input_height * channels_count)

        if state_update_cell == 'gru':
            state_update = GRU_Cell_State_Update(state_update_depth, channels_factors_size, embeddings_factors_size,
                                                 channels_factors_count, embeddings_factors_count, bias=True)
        else:
            state_update = LSTM_Cell_State_Update(state_update_depth, channels_factors_size, embeddings_factors_size,
                                                  channels_factors_count, embeddings_factors_count, bias=True)

        policy_first_layer = Linear_Layer(embeddings_size, channels_factors_size * embeddings_size)
        policy_second_layer = Linear_Layer(embeddings_size, channels_factors_size * kernel_size ** 2)

        state_update_computation = network_depth * channels_factors_count ** 2 * number_of_layers_in_residual_blocks * state_update.computation()

        policy_computation = network_depth * channels_factors_count ** 2 * number_of_layers_in_residual_blocks * (policy_first_layer.computation() +
                channels_factors_size * policy_second_layer.computation())

        layers_per_reduction_step = network_depth // reduction_step_count
        residual_blocks = []
        for i in range(network_depth):
            reduction_step = i // layers_per_reduction_step

            reduced_height = input_height // (2 ** (reduction_step))
            reduced_width = input_width // (2 ** (reduction_step))

            residual_blocks.append(Predicted_Residual_Block(reduced_width, reduced_height, channels_count, channels_count, kernel_size, downsizing=False, bias=False))

        global_average_pooling = GlobalAvgPool2D(input_width, input_height, channels_count)
        classification_layer = Linear_Layer(channels_count, class_count)

        memory = first_conv.memory() + first_conv_bn.memory() + first_conv_relu.memory() +\
                 state_update.memory() + policy_first_layer.memory() + policy_second_layer.memory() + \
                 sum([layer.memory() for layer in residual_blocks]) +\
                 global_average_pooling.memory() + classification_layer.memory()

        computation = first_conv.computation() + first_conv_bn.computation() + first_conv_relu.computation() +\
                      state_update_computation + policy_computation +\
                      sum([layer.computation() for layer in residual_blocks]) +\
                      global_average_pooling.computation() + classification_layer.computation()

        super(Policy_Hypernetwork, self).__init__({
            "first_conv": first_conv.memory() + first_conv_bn.memory() + first_conv_relu.memory(),
            "state_update": state_update.memory(),
            "policy": policy_first_layer.memory() + policy_second_layer.memory(),
            "layer": sum([layer.memory() for layer in residual_blocks]),
            "classification": global_average_pooling.memory() + classification_layer.memory(),
            "total": memory
        }, {
            "first_conv": first_conv.computation() + first_conv_bn.computation() + first_conv_relu.computation(),
            "state_update": state_update_computation,
            "policy": policy_computation,
            "layer": sum([layer.computation() for layer in residual_blocks]),
            "classification": global_average_pooling.computation() + classification_layer.computation(),
            "total": computation
        })

