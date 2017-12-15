

class ConsoleOutput:
    def __call__(self, result):
        print('Results for {}: {}'.format(result[0], result[1]))