

class ConsoleOutput:
    def __call__(self, result):
        print('Results for {}'.format(result['name']))
        print('Accuracy: {}'.format(result['accuracy']))
        print('Best hyperparameters: {}'.format(result['best_hyperparameters']))