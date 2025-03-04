# %%
import argparse
from ast import arg
from data_utils import sent2features, sent2labels, sent2tokens, read_examples_from_file
import pycrfsuite
from nervaluate import Evaluator

argparser = argparse.ArgumentParser()
argparser.add_argument("--mode", default="bio", type=str)
argparser.add_argument("--use_simple_feature_only", action="store_true")
args = argparser.parse_args()


mode = args.mode
simple = args.use_simple_feature_only


train_file = f"./data/processed/train1_{mode}.txt"
test_file = f"./data/processed/testright_{mode}.txt"
model = f"./msra_{mode}_{simple}.crfsuite"

# %%
if __name__ == "__main__":
    # read data
    trainset = read_examples_from_file(train_file, "train")
    testset = read_examples_from_file(test_file, "test")

    X_train = [sent2features(s, simple) for s in trainset]
    y_train = [sent2labels(s) for s in trainset]

    X_test = [sent2features(s, simple) for s in testset]
    y_test = [sent2labels(s) for s in testset]
    # training
    #############
    # buld trainer with pycrfsuite and set related hyper-parameters
    import pycrfsuite

    trainer = pycrfsuite.Trainer(verbose=True)
    trainer.set_params(
        {
            "c1": 1.0,  # coefficient for L1 penalty
            "c2": 1e-3,  # coefficient for L2 penalty
            "max_iterations": 50,  # stop earlier
            # include transitions that are possible, but not observed
            "feature.possible_transitions": True,
        }
    )
    ##################

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.train(model)
    # evaluation
    tagger = pycrfsuite.Tagger()
    tagger.open(model)

    labels = [sent2labels(s) for s in testset]
    pred = [tagger.tag(sent2features(s, simple)) for s in testset]

    tags = []
    for label in labels:
        tags += [l.split("-")[-1] for l in label]
    tags = list(set(tags))
    tags.remove("O")
    evaluator = Evaluator(labels, pred, tags=tags, loader="list")

    # results, results_per_tag = evaluator.evaluate()
    results, results_per_tag, result_indices, result_indices_by_tag = evaluator.evaluate()
    print(results_per_tag)
    print(results)
    print(result_indices)
    print(result_indices_by_tag)
