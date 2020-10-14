from utils.resources_estimation.operation.estimator import Estimator
from utils.resources_estimation.operation.basics import Linear_Layer, Sigmoid, Tanh, PointToPointOperation, Vector_Table

class LSTM_Cell(Estimator):
    def __init__(self, input_size, hidden_size, bias=True):
        sigmoid = Sigmoid(hidden_size)
        tanh = Tanh(hidden_size)
        input_gate = Linear_Layer(input_size + hidden_size, hidden_size, bias=bias)
        forget_gate = Linear_Layer(input_size + hidden_size, hidden_size, bias=bias)
        cell_gate = Linear_Layer(input_size + hidden_size, hidden_size, bias=bias)
        output_gate = Linear_Layer(input_size + hidden_size, hidden_size, bias=bias)
        point_to_point_operations = PointToPointOperation(hidden_size)

        memory = input_gate.memory() + forget_gate.memory() + cell_gate.memory()\
                 + output_gate.memory() + 4 * point_to_point_operations.memory()\
                 + 3 * sigmoid.memory() + tanh.memory()
        computation = input_gate.computation() + forget_gate.computation() + cell_gate.computation()\
                      + output_gate.computation() + 4 * point_to_point_operations.computation()\
                      + 3 * sigmoid.computation() + tanh.computation()

        super(LSTM_Cell, self).__init__(memory, computation)

class LSTM_Cell_State_Update(Estimator):
    def __init__(self, layer_count, channels_factor_size, embedding_factors_size, channels_factors_count, embedding_factors_count, bias=True):
        hidden = Vector_Table(embedding_factors_size, channels_factors_count ** 2 * embedding_factors_count)
        cs = Vector_Table(embedding_factors_size, channels_factors_count ** 2 * embedding_factors_count)
        first_cell = LSTM_Cell(channels_factor_size, embedding_factors_size, bias=bias)
        other_cells = LSTM_Cell(embedding_factors_size, embedding_factors_size, bias=bias)

        memory = hidden.memory() + cs.memory() + first_cell.memory() + max(0, layer_count-1) * other_cells.memory()
        computation = hidden.computation() + cs.computation() + first_cell.computation() + max(0, layer_count-1) * other_cells.computation()
        super(LSTM_Cell_State_Update, self).__init__(memory, computation)

class GRU_Cell(Estimator):
    def __init__(self, input_size, hidden_size, bias=True):
        sigmoid = Sigmoid(hidden_size)
        tanh = Tanh(hidden_size)
        r = Linear_Layer(input_size + hidden_size, hidden_size, bias=bias)
        z = Linear_Layer(input_size + hidden_size, hidden_size, bias=bias)
        n = Linear_Layer(input_size + hidden_size, hidden_size, bias=bias)
        point_to_point_operations = PointToPointOperation(hidden_size)

        memory = r.memory() + z.memory() + n.memory()\
                 + 5 * point_to_point_operations.memory() + 2 * sigmoid.memory() + tanh.memory()
        computation = r.computation() + z.computation() + n.computation()\
                      + 5 * point_to_point_operations.computation() + 2 * sigmoid.computation() + tanh.computation()

        super(GRU_Cell, self).__init__(memory, computation)

class GRU_Cell_State_Update(Estimator):
    def __init__(self, layer_count, channels_factor_size, embedding_factors_size, channels_factors_count, embedding_factors_count, bias=True):
        Layers_in_residual_block = 2
        hidden = Vector_Table(embedding_factors_size, Layers_in_residual_block * channels_factors_count ** 2 * embedding_factors_count)
        first_cell = GRU_Cell(channels_factor_size, embedding_factors_size, bias=bias)
        other_cells = GRU_Cell(embedding_factors_size, embedding_factors_size, bias=bias)

        memory = hidden.memory() + first_cell.memory() + max(0, layer_count-1) * other_cells.memory()
        computation = hidden.computation() + first_cell.computation() + max(0, layer_count-1) * other_cells.computation()
        super(GRU_Cell_State_Update, self).__init__(memory, computation)
