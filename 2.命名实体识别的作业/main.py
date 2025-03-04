# %%
from data_utils import sent2features, sent2labels, sent2tokens, read_examples_from_file
import pycrfsuite
from nervaluate import Evaluator

mode = "bio"

train_file = f"./data/processed/train1_{mode}.txt"
test_file = f"./data/processed/testright_{mode}.txt"
model = f"./msra_{mode}.crfsuite"
# %%
if __name__ == "__main__":
    # read data
    trainset = read_examples_from_file(train_file, "train")
    testset = read_examples_from_file(test_file, "test")

    X_train = [sent2features(s) for s in trainset]
    y_train = [sent2labels(s) for s in trainset]

    X_test = [sent2features(s) for s in testset]
    y_test = [sent2labels(s) for s in testset]
    # training
    #############
    # TODO
    # buld trainer with pycrfsuite and set related hyper-parameters
    trainer = None  # comment this after defining your own trainer
    ##################

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.train(model)
    # evaluation
    tagger = pycrfsuite.Tagger()
    tagger.open(model)

    labels = [sent2labels(s) for s in testset]
    pred = [tagger.tag(sent2features(s)) for s in testset]

    tags = []
    for label in labels:
        tags += [l.split("-")[-1] for l in label]
    tags = list(set(tags))
    tags.remove("O")
    evaluator = Evaluator(labels, pred, tags=tags, loader="list")

    results, results_by_tag = evaluator.evaluate()
    print(results_by_tag)
    print(results)
