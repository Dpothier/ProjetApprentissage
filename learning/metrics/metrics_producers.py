from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

class MeanAccuracyMetricProducer:

    def __call__(self, results):
        train_accuracies = []
        test_accuracies = []
        for result in results:
            train_accuracies.append(accuracy_score(result['y_train'], result['pred_train']))
            test_accuracies.append(accuracy_score(result['y_test'], result['pred_test']))
        return {
            'mean_train_accuracy': sum(train_accuracies) / len(train_accuracies),
            'mean_test_accuracy': sum(test_accuracies) / len(test_accuracies)
        }

class MeanTrainingTimeMetricProducer:
    def __call__(self, results):
        train_times = []
        for result in results:
            train_times.append(result['execution_time'])

        return {
            'mean_train_time': sum(train_times)/len(train_times)
        }


class MeanConfusionMatrixMetricProducer:
    def __call__(self, results):
        matrices = []
        for result in results:
            matrices.append(confusion_matrix(result['y_test'], result['pred_test']))

        return {'mean_confusion_matrix': sum(matrices)/len(matrices)}

class ModelDumper:
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, results):
        for i in range(0, len(results)):
            complete_filename = "{}_{}.pkl".format(self.filename, i)
            joblib.dump(results[i]['model'], complete_filename)
