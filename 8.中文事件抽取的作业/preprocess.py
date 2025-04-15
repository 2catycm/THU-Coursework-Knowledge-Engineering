# %%
import json
import os

from tqdm import tqdm


# trigger data
def process_trigger_data(file):
    """
    convert raw data into sequence-labeling format data for trigger identification, and save to `./data/processed/trigger`
    each line in the converted contains `token[space]label`
    you can use IOB2 format tagging https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging), or other tagging schema you think fit
    use empty line to indicate end of one sentence
    if using IOB2 format, the example output would be like
    ```
    本 O
    平 O
    台 O
    S O
    S O
    R O
    N O
    发 B-EVENT
    表 I-EVENT
    了 O
    题 O
    为 O
    《 O
    夏 O
    季 O
    冠 O
    状 O
    病 O
    毒 O
    流 O
    行 O
    会 O
    减 O
    少 O
    吗 O

    对 O
    于 O
    该 O
    举 O
    动 O
    ```
    """

    with open(f"./data/raw/{file}.json", encoding="utf-8") as f:
        lines = f.readlines()
    outlines = []
    ##################
    bar = tqdm(lines, desc="Processing trigger data")
    for line in bar:
        sample_object = json.loads(line.strip())
        text = sample_object["text"]
        tokens = list(text) # 单字分词
        labels = sample_object["labels"]
        labels_seq = ["O"] * len(tokens) # 默认情况
        # 依据 labels 进行标注
        for label in labels:
            if label["trigger"]:
                start = label["trigger"][1] # 起始位置
                end = start + len(label["trigger"][0]) # 结束位置
                # 触发词标注
                labels_seq[start] = "B-EVENT"
                for i in range(start + 1, end):
                    labels_seq[i] = "I-EVENT"
        # 生成输出
        for i in range(len(tokens)):
            outlines.append(f"{tokens[i]} {labels_seq[i]}")
        outlines.append("") # 句子结束标志，用空行分割

    ##################

    if not os.path.exists("./data/processed/trigger"):
        os.makedirs("./data/processed/trigger", exist_ok=True)
    with open(f"./data/processed/trigger/{file}.txt", "w") as f:
        f.writelines("\n".join(outlines))


# %%
# argument data
def process_argument_data(file):
    """
    convert raw data into sequence-labeling format data for argument identification, and save to `./data/processed/argument`
    event triggers are surrounded by `<event>` `<event/>` markers
    each line in the converted contains `token[space]label`
    you can use BIO format tagging https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging), or other tagging schema you think fit
    use empty line to indicate end of one sentence
    if using IOB2 format, the example output would be like
    ```
    3 B-object
    0 I-object
    年 I-object
    代 I-object
    <event> O
    参 O
    加 O
    <event/> O
    中 B-subject
    共 I-subject
    中 I-subject
    央 I-subject
    的 I-subject
    特 I-subject
    种 I-subject
    领 I-subject
    导 I-subject
    工 I-subject
    作 I-subject

    他 O
    告 O
    诉 O
    新 O
    京 O
    报 O
    记 O
    者 O
    ```
    """
    with open(f"./data/raw/{file}.json", encoding="utf-8") as f:
        lines = f.readlines()
    outlines = []
    #############
    bar = tqdm(lines, desc="Processing argument data")
    for line in bar:
        sample_object = json.loads(line.strip())
        text = sample_object["text"]
        tokens = list(text)
        labels = sample_object["labels"]
        labels_seq = ["O"] * len(tokens)
        # 依据 labels 进行标注
        for label in labels:
            # 先处理触发词， 如果强行插入新词的话，不太好，我决定后面输出的时候特别操作。
            if label["trigger"]:
                start = label["trigger"][1] # 起始位置
                end = start + len(label["trigger"][0]) # 结束位置
                # 触发词标注
                labels_seq[start] = "B-EVENT"
                for i in range(start + 1, end-1):
                    labels_seq[i] = "O"
                labels_seq[end-1] = "E-EVENT" # 触发词最后一个字标注为 E-EVENT

            # 处理 object subject
            if label["object"]:
                start = label["object"][1]
                end = start + len(label["object"][0])
                labels_seq[start] = "B-object"
                for i in range(start + 1, end):
                    labels_seq[i] = "I-object"

            if label["subject"]:
                start = label["subject"][1]
                end = start + len(label["subject"][0])
                labels_seq[start] = "B-subject"
                for i in range(start + 1, end):
                    labels_seq[i] = "I-subject"

            if label['time']:
                start = label['time'][1]
                end = start + len(label['time'][0])
                labels_seq[start] = "B-time"
                for i in range(start + 1, end):
                    labels_seq[i] = "I-time"

            if label['location']:
                start = label['location'][1]
                end = start + len(label['location'][0])
                labels_seq[start] = "B-location"
                for i in range(start + 1, end):
                    labels_seq[i] = "I-location"

        # 生成输出
        for i in range(len(tokens)):
            if labels_seq[i] == "B-EVENT":
                outlines.append("<event> O")
                outlines.append(f"{tokens[i]} O")
            elif labels_seq[i] == "E-EVENT":
                outlines.append(f"{tokens[i]} O")
                outlines.append("<event/> O")
            else:
                # 普通情况
                outlines.append(f"{tokens[i]} {labels_seq[i]}")
        outlines.append("") # 句子结束标志，用空行分割

    #############

    if not os.path.exists("./data/argument"):
        os.makedirs("./data/processed/argument", exist_ok=True)
    with open(f"./data/processed/argument/{file}.txt", "w") as f:
        f.writelines("\n".join(outlines))


def gen_labels(mode):
    path = f"./data/processed/{mode}/train.txt"
    labels = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        labels.append(line.strip().split(" ")[-1])
    labels = list(set(labels))
    if "" in labels:
        labels.remove("")
    with open(f"./data/processed/{mode}/labels.txt", "w") as f:
        f.writelines("\n".join(labels))


# %%
if __name__ == "__main__":
    files = ["train", "dev"]
    for f in files:
        process_argument_data(f)
        process_trigger_data(f)
    gen_labels("trigger")
    gen_labels("argument")

# %%
