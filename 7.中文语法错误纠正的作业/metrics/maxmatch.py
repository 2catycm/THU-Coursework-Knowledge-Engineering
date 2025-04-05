from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
import sys
sys.path.append((this_directory/"m2scorer").as_posix())

import levenshtein as levenshtein
from util import paragraphs
from util import smart_open
from typing import Any

def load_annotation(gold_file):
    source_sentences = []
    gold_edits = []
    fgold = smart_open(gold_file, "r")
    puffer = fgold.read()
    fgold.close()
    # puffer = puffer.decode('utf8')
    for item in paragraphs(puffer.splitlines(True)):
        item = item.splitlines(False)
        sentence = [line[2:].strip() for line in item if line.startswith("S ")]
        assert sentence != []
        annotations = {}
        for line in item[1:]:
            if line.startswith("I ") or line.startswith("S "):
                continue
            assert line.startswith("A ")
            line = line[2:]
            fields = line.split("|||")
            start_offset = int(fields[0].split()[0])
            end_offset = int(fields[0].split()[1])
            etype = fields[1]
            if etype == "noop":
                start_offset = -1
                end_offset = -1
            corrections = [
                c.strip() if c != "-NONE-" else "" for c in fields[2].split("||")
            ]
            # NOTE: start and end are *token* offsets
            original = " ".join(" ".join(sentence).split()[start_offset:end_offset])
            annotator = int(fields[5])
            if annotator not in list(annotations.keys()):
                annotations[annotator] = []
            annotations[annotator].append(
                (start_offset, end_offset, original, corrections)
            )
        tok_offset = 0
        for this_sentence in sentence:
            tok_offset += len(this_sentence.split())
            source_sentences.append(this_sentence)
            this_edits = {}
            for annotator, annotation in annotations.items():
                this_edits[annotator] = [
                    edit
                    for edit in annotation
                    if edit[0] <= tok_offset
                    and edit[1] <= tok_offset
                    and edit[0] >= 0
                    and edit[1] >= 0
                ]
            if len(this_edits) == 0:
                this_edits[0] = []
            gold_edits.append(this_edits)
    return (source_sentences, gold_edits)


def maxmatch_metric(prediction_file: str # a file containing predicted output
                    , label_file: str # a file containig groundtruth output
                    , verbose:bool = True
                    ) -> Any:
    """
    calculate maxmatch metrics

    File content example
    # prediction file
    ```
    冬 阴功 是 泰国 最 著名 的 菜 之一 ， 它 虽然 不 是 很 豪华 ， 但 它 的 味 确实 让 人 上瘾 ， 做法 也 不 难 、 不 复杂 。
    首先 ， 我们 得 准备 : 大 虾六 到 九 只 、 盐 一 茶匙 、 已 搾 好 的 柠檬汁 三 汤匙 、 泰国 柠檬 叶三叶 、 柠檬 香草 一 根 、 鱼酱 两 汤匙 、 辣椒 6 粒 ， 纯净 水 4量杯 、 香菜 半量杯 和 草菇 10 个 。
    ```
    # label_file
    ```
    S 冬 阴功 是 泰国 最 著名 的 菜 之一 ， 它 虽然 不 是 很 豪华 ， 但 它 的 味 确实 让 人 上瘾 ， 做法 也 不 难 、 不 复杂 。
    A 9 11|||W|||虽然 它|||REQUIRED|||-NONE-|||0

    S 首先 ， 我们 得 准备 : 大 虾六 到 九 只 、 盐 一 茶匙 、 已 搾 好 的 柠檬汁 三 汤匙 、 泰国 柠檬 叶三叶 、 柠檬 香草 一 根 、 鱼酱 两 汤匙 、 辣椒 6 粒 ， 纯净 水 4量杯 、 香菜 半量杯 和 草菇 10 个 。
    A 17 18|||S|||榨|||REQUIRED|||-NONE-|||0
    A 38 39|||S|||六|||REQUIRED|||-NONE-|||0
    A 43 44|||S|||四 量杯|||REQUIRED|||-NONE-|||0
    A 49 50|||S|||十|||REQUIRED|||-NONE-|||0
    ```
    """
    max_unchanged_words = 2
    beta = 0.5
    ignore_whitespace_casing = False
    very_verbose = False

    # load source sentences and gold edits
    source_sentences, gold_edits = load_annotation(label_file)

    # load system hypotheses
    fin = smart_open(prediction_file, "r")
    system_sentences = [line.strip() for line in fin.readlines()]
    fin.close()

    p, r, f1 = levenshtein.batch_multi_pre_rec_f1(
        system_sentences,
        source_sentences,
        gold_edits,
        max_unchanged_words,
        beta,
        ignore_whitespace_casing,
        verbose,
        very_verbose,
    )

    metrics = {
        "Precision": p,
        "Recall": r,
        "F_{}".format(beta): f1
    }

    return metrics

# 如果需要测试函数，可以调用它并打印结果
if __name__ == "__main__":
    prediction_file = (this_directory/"../data/test_prediction_file_correct.txt").as_posix()
    label_file = (this_directory/"../data/test_label_file.txt").as_posix()
    metrics = maxmatch_metric(prediction_file, label_file)
    print("Metrics:", metrics)
