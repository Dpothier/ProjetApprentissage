class Estimator():
    def __init__(self, memory, computation):
        self._memory = memory
        self._computation = computation

    def memory(self):
        return self._memory

    def computation(self):
        return self._computation