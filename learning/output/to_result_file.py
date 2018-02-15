

class ResultFileOutput:
    def __init__(self, output_file):
        self.output_file = open("{}.txt".format(output_file), mode="w", encoding="utf8")

    def __call__(self, result):
        self.output_file.writelines('Results for {}'.format(result['name']))
        self.output_file.writelines('Accuracy: {}'.format(result['accuracy']))
        self.output_file.writelines('Best hyperparameters: {}'.format(result['best_hyperparameters']))

    def __del__(self):
        self.output_file.close()