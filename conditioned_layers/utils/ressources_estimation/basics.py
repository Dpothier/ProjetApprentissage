from utils.ressources_estimation.estimator import Estimator

class Sigmoid(Estimator):
    def __init__(self, size):
        memory=0
        computation= 4 * size

        super(Sigmoid, self).__init__(memory, computation)

class Tanh(Estimator):
    def __init__(self, size):
        memory=0
        computation= 4 * size

        super(Tanh, self).__init__(memory, computation)

class Relu(Estimator):
    def __init__(self, size):
        memory=0
        computation= size

        super(Relu, self).__init__(memory, computation)

class PointToPointOperation(Estimator):
    def __init__(self, size):
        memory=0
        computation= size

        super(PointToPointOperation, self).__init__(memory, computation)

class Layer_Embeddings(Estimator):
    def __init__(self, embedding_size, embedding_count):
        memory = embedding_size * embedding_count
        computation = 0

        super(Layer_Embeddings, self).__init__(memory, computation)

class MatMul(Estimator):
    def __init__(self, outer_1, inner, outer_2):
        memory = 0
        computation = 2 * inner * outer_1 * outer_2

        super(MatMul, self).__init__(memory, computation)

class Linear_Layer(Estimator):
    def __init__(self, in_size, out_size, bias=True):
        layer_memory = in_size * out_size
        layer_computation = MatMul(1, in_size, out_size).computation()
        layer_computation = layer_computation if not bias else layer_computation + out_size



        super(Linear_Layer, self).__init__(layer_memory, layer_computation)

#Like all operation, Batch_Norm simplify the batch_size term of it's computation cost
class Batch_Norm(Estimator):
    def __init__(self, x_size):
        avg = x_size
        variance = 3 * x_size
        normalize = 4 * x_size
        scale_shift = 2 * x_size
        computation = avg + variance + normalize + scale_shift

        super(Batch_Norm, self).__init__(2, computation)
