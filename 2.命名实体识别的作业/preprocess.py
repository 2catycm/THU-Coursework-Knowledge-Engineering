# %%
from typing import List


# | export
from typing import List


def bio(
    chars: str, tag: str
) -> List[List[str]]:  # Output: each sub list is a pair of character and tag
    """
    BIO tagging format
    Examples:
        Input: "北京大学", "nt"
        Output: [["北", "B-NT"], ["京", "I-NT"], ["大", "I-NT"], ["学", "I-NT"]]
    """
    tag = tag.upper()  # 根据助教的实例，输出的格式要求大写
    if tag == "O":
        return [[char, tag] for char in chars]  # 不是实体，返回一个O
    else:
        return [[chars[0], f"B-{tag}"]] + [
            [char, f"I-{tag}"]
            for char in chars[1:]  # 如果是单个字的情况下，那就是 B-tag
        ]


def bioes(
    chars: str, tag: str
) -> List[List[str]]:  # Output: each sub list is a pair of character and tag
    """
    BIOES tagging format
    Examples:
        Input: "北京大学", "nt"
        Output: [["北", "B-NT"], ["京", "I-NT"], ["大", "I-NT"], ["学", "E-NT"]]
    """
    tag = tag.upper()
    if tag == "O":
        return [[char, tag] for char in chars]
    elif len(chars) == 1:
        return [[chars[0], f"S-{tag}"]]  # 单个字的实体
    else:
        return (
            [[chars[0], f"B-{tag}"]]
            + [
                [char, f"I-{tag}"]
                for char in chars[1:-1]  # 如果是两个字的情况下, chars[1:-1]为空
            ]
            + [[chars[-1], f"E-{tag}"]]
        )


# %%
def process(file, target, mode="bio"):
    if mode == "bio":
        labeling = bio
    elif mode == "bioes":
        labeling = bioes
    else:
        raise NotImplementedError
    with open(file, encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        new_line = []
        line = line.strip()
        words = line.split(" ")
        for word in words:
            chars, tag = word.split("/")
            new_line += labeling(chars, tag)
        new_lines.append(new_line)
    with open(target, "w", encoding="utf-8") as fw:
        fw.writelines(
            "\n\n".join(
                ["\n".join([" ".join(pair) for pair in line]) for line in new_lines]
            )
        )


# %%
if __name__ == "__main__":
    process("./data/raw/train1.txt", "./data/processed/train1_bio.txt", "bio")
    process("./data/raw/train1.txt", "./data/processed/train1_bioes.txt", "bioes")
    process("./data/raw/testright1.txt", "./data/processed/testright_bio.txt", "bio")
    process(
        "./data/raw/testright1.txt", "./data/processed/testright_bioes.txt", "bioes"
    )

# %%
