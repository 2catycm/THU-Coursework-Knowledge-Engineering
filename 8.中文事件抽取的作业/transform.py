import json
from tqdm import tqdm

"""
convert predictions for event trigger identification to IOB2 format for stage2 training (argument identification)
"""


def convert(preds):
    """
    Args:
        preds: List[List[List[int]]], [[[trigger1_start_index, trigger2_start_index,...], [trigger1_end_index, trigger2_end_index, ...]], ...]
    """
    with open(f"./data/raw/dev.json", encoding="utf-8") as f:
        lines = f.readlines()
    outlines = []
    bar = tqdm(enumerate(lines), desc="Processing")
    for i, line in bar:
        trigger_start = preds[i][0]  # list[int] of start token index
        trigger_end = preds[i][1]  # list[int] of end token index
        argument_dict = dict()
        line = line.strip()
        d = json.loads(line)
        id_ = d["id"]
        text = d["text"]
        for label in d["labels"]:
            for k in label:
                if k == "trigger":
                    continue
                if label[k]:
                    start = label[k][1]
                    end = start + len(label[k][0])
                    tmp_dict = dict(
                        zip(
                            range(start, end),
                            [f"B-{k}"] + [f"I-{k}"] * (end - start - 1),
                        )
                    )
                    argument_dict.update(tmp_dict)
        for i in range(len(text)):
            if i in trigger_start:
                outlines.append("<event> O")
            if i in trigger_end:
                outlines.append("<event/> O")
            outlines.append(" ".join([text[i], argument_dict.get(i, "O")]))
        outlines.append("")
    with open("./data/processed/argument/test.txt", "w", encoding="utf-8") as f:
        f.writelines("\n".join(outlines))


def read_prediction():
    with open("./checkpoint/trigger/checkpoint-best/eval_predictions.txt") as f:
        lines = f.readlines()
    trigger_preds = []  # [[[start1, start2,...], [end1, end2, ...]]]
    pred = [[], []]  # [[start], [end]]
    i = 0
    idx = 0
    while idx < len(lines):
        # print(idx)
        line = lines[idx].strip()
        if line:
            line_list = line.split(" ")
            if line_list[-1] == "B-EVENT":
                pred[0].append(i)
                idx += 1
                i += 1
                line = lines[idx].strip()
                line_list = line.split(" ")
                while line and line_list[-1] == "I-EVENT":
                    idx += 1
                    i += 1
                    line = lines[idx].strip()
                    line_list = line.split(" ")
                pred[1].append(i)
            else:
                i += 1
                idx += 1
                continue
        else:
            i = 0  # reset token index
            trigger_preds.append(pred)  # save one sent
            pred = [[], []]
            idx += 1
    trigger_preds.append(pred)
    return trigger_preds


if __name__ == "__main__":
    preds = read_prediction()
    convert(preds)
