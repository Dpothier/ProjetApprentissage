from utils.ressources_estimation.estimator import Estimator
from utils.ressources_estimation.convolutions import Learned_Conv, Predicted_Residual_Block
from utils.ressources_estimation.basics import Relu, Batch_Norm, Layer_Embeddings, Linear_Layer

#This estimator assumes that dimension reduction occurs at regular interval
class Static_Hypernetwork(Estimator):
    def __init__(self, embedding_size, base_channel_count, initial_image_height, initial_image_width, depth, reduction_counts):
        kernel_size = 3
        first_conv = Learned_Conv(initial_image_height, initial_image_width, kernel_size, base_channel_count, 1)
        first_conv_bn = Batch_Norm(initial_image_width * initial_image_height * base_channel_count)
        first_conv_relu = Relu(initial_image_width * initial_image_height * base_channel_count)

        policy_first_layer = Linear_Layer(embedding_size, base_channel_count * embedding_size)
        policy_second_layer = Linear_Layer(embedding_size, base_channel_count * kernel_size ** 2)

        layers_per_reduction_step = depth // reduction_counts
        embedding_counts = 0
        residual_blocks = []
        for i in range(depth):
            reduction_step = i // layers_per_reduction_step
            downsize_layer = depth % reduction_counts == 0 and reduction_step == 0
            embedding_counts += 2 ** reduction_step
            if downsize_layer:
                in_channels = base_channel_count * (2 ** (reduction_step - 1))
                out_channels = base_channel_count * (2 ** reduction_step)
                image_height = initial_image_height // (2 ** (reduction_step - 1))
                image_width = initial_image_width // (2 ** (reduction_step - 1))
            else:
                in_channels = base_channel_count * (2 ** reduction_step)
                out_channels = base_channel_count * (2 ** reduction_step)
                image_height = initial_image_height // (2 ** reduction_step )
                image_width = initial_image_width // (2 ** reduction_step )

            residual_blocks.append(Predicted_Residual_Block(image_width, image_height, in_channels, out_channels, kernel_size, downsize_layer, bias=False))

            embeddings = Layer_Embeddings(embedding_size, embedding_counts)


