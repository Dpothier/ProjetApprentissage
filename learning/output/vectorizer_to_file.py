from sklearn.externals import joblib

class VectorizerOutput:

    def __init__(self, filename):
        self.filename = filename

    def __call__(self, result):
        complete_filename = "{}_{}.pkl".format(self.filename, result['name'])
        joblib.dump(result['vectorizer'], complete_filename)
