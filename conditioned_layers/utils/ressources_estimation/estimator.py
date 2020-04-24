import abc

class Estimator(abc.ABC):
    def __init__(self, memory, computation):
        self.memory = memory
        self.computation = computation

    def estimate_memory(self):
        return self.memory

    @abc.abstractmethod
    def estimate_computation(self):
        return self.computation