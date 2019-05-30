import sys

sys.path.append('../common')
sys.path.append('/mnt/storage/dpothier/tmp/pytoune')


from conditioned_layers.datasets.Cola.ColaDataset import ColaDataset
from conditioned_layers.training.metrics_util import *
from conditioned_layers.training.embeddings import load


TEST_MODE = False


def main():
    """
    Trains the LSTM-based integrated pattern-based and distributional method for hypernymy detection
    :return:
    """

    dataset_prefix = sys.argv[1]
    embeddings_file = sys.argv[2]


    np.random.seed(133)

    word_vectors, vocab_table = load(embeddings_file)

    t = ColaDataset("{}train.tsv".format(dataset_prefix), vocab_table, TEST_MODE=TEST_MODE)
    d = ColaDataset("{}dev.tsv".format(dataset_prefix), vocab_table, TEST_MODE=TEST_MODE)
    train = DataLoader(t, batch_size=32, collate_fn=t.get_collate_fn())
    dev = DataLoader(d, batch_size=32, collate_fn=d.get_collate_fn())

    train_classes_ratio = calculate_weight(train)
    print("Ratio of classes in train set: {}".format(train_classes_ratio))
    dev_classes_ratio = calculate_weight(dev)
    print("Ratio of classes in dev set: {}".format(dev_classes_ratio))



if __name__ == '__main__':
    main()