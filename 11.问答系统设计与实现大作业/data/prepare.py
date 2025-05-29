# coding: utf-8
# 从训练集中选取数据使用BERT进行相关度二分类训练

import json
import random
import os
from tqdm import tqdm
import sys

sys.path.append("/userhome/KE_final")
from utils.normalize import filter_tags

datasets = []
pid_list = []


def prepare_dataset(filename1, filename2):
    """Prepare the sentence pair task training dataset for bert"""
    with (
        open(filename1, "r", encoding="utf-8") as f,
        open(filename2, "r", encoding="utf-8") as g,
    ):
        documents = g.readlines()

        for lidx, line in enumerate(tqdm(f)):
            #             print(line)
            #             break
            sample = json.loads(line.strip())
            qid = sample["pid"]
            print(qid)
            pid_list.append(qid)
            question = sample["question"]
            related_para = sample["answer_sentence"][0].replace("\n", "")
            datasets.append([1, question, related_para])
            for line in documents:
                sample_ = json.loads(line.strip())
                if qid == sample_["pid"]:
                    print("find!")
                    doc = sample_["document"]
                    for i in range(len(doc)):
                        current_doc = doc[i].replace("\n", "")
                        if current_doc != related_para:
                            irrelated_para = current_doc
                            datasets.append([0, question, irrelated_para])
                            # print(datasets)
    return datasets


def write_tsv(output_path, datasets):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, data in enumerate(datasets):
            write_line = "\t".join(
                [
                    str(data[0]),
                    str(i),
                    str(i),
                    filter_tags(data[1]),
                    filter_tags(data[2]),
                ]
            )
            f.write(write_line + "\n")


def main():
    if not os.path.exists("/userhome/KE_final"):
        os.mkdir("/userhome/KE_final")
    print("Start loading data file.")
    datasets = prepare_dataset(
        "/userhome/KE_final/data/train.json",
        "/userhome/KE_final/data/passages_multi_sentences.json",
    )
    # random.shuffle(datasets)
    train_datasets = datasets[:100000]
    test_datasets = datasets[100000:]
    write_tsv("/userhome/KE_final/data/train.tsv", train_datasets)
    print("Done with preparing training dataset.")
    write_tsv("/userhome/KE_final/data/test.tsv", test_datasets)
    print("Done with preparing testing dataset.")


#     print(len(datasets))
#     print(len(pid_list))


if __name__ == "__main__":
    main()
