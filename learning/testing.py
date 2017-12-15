from sklearn.datasets import load_iris
from algorithms.SVM import SVM
from algorithms.MLP import MLP
from algorithms.ADABoost import ADABoost
from getData import get_bdrv_data
from sklearn.feature_extraction.text import CountVectorizer
import experiment.experiment_setup as setup

data, targets = get_bdrv_data()

iris = load_iris()


setup.With_kfold(10)\
    .use_ADABoost()\
    .use_validation_set(0.2)\
    .use_raw_data()\
    .output_to_console()\
    .go(iris.data, iris.target)