from utils.ressources_estimation.estimator import Estimator
from utils.ressources_estimation.basics import Linear_Layer, Sigmoid, Tanh, PointToPointOperation

class LSTM_Cell(Estimator):
    def __init__(self, input_size, hidden_size):
        sigmoid = Sigmoid(hidden_size)
        tanh = Tanh(hidden_size)
        input_gate = Linear_Layer(input_size + hidden_size, hidden_size)
        forget_gate = Linear_Layer(input_size + hidden_size, hidden_size)
        cell_gate = Linear_Layer(input_size + hidden_size, hidden_size)
        output_gate = Linear_Layer(input_size + hidden_size, hidden_size)
        point_to_point_operations = PointToPointOperation(hidden_size)

        memory = input_gate.memory() + forget_gate.memory() + cell_gate.memory()\
                 + output_gate.memory() + 4 * point_to_point_operations.memory()\
                 + 3 * sigmoid.memory() + tanh.memory()
        computation = input_gate.computation() + forget_gate.computation() + cell_gate.computation()\
                      + output_gate.computation() + 4 * point_to_point_operations.computation()\
                      + 3 * sigmoid.computation() + tanh.computation()

        super(LSTM_Cell, self).__init__(memory, computation)

class GRU_cell(Estimator):
    def __init__(self, input_size, hidden_size):
        sigmoid = Sigmoid(hidden_size)
        tanh = Tanh(hidden_size)
        r = Linear_Layer(input_size + hidden_size, hidden_size)
        z = Linear_Layer(input_size + hidden_size, hidden_size)
        n = Linear_Layer(input_size + hidden_size, hidden_size)
        point_to_point_operations = PointToPointOperation(hidden_size)

        memory = r.memory() + z.memory() + n.memory()\
                 + 5 * point_to_point_operations.memory() + 2 * sigmoid.memory() + tanh.memory()
        computation = r.computation() + z.computation() + n.computation()\
                      + 5 * point_to_point_operations.computation() + 2 * sigmoid.computation() + tanh.computation()

        super(GRU_cell, self).__init__(memory, computation)
