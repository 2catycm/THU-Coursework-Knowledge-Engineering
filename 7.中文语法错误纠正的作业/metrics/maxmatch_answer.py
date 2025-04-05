from typing import *
import subprocess


def maxmatch_metric(prediction_file: str, label_file: str) -> Any:
    """
    calculate maxmatch metrics
    Args:
        prediction_file: a file containing predicted output
        label_file: a file containig groundtruth output
    Returns:
        Any

    File content example
    # prediction file
    ```
    冬 阴功 是 泰国 最 著名 的 菜 之一 ， 它 虽然 不 是 很 豪华 ， 但 它 的 味 确实 让 人 上瘾 ， 做法 也 不 难 、 不 复杂 。
    首先 ， 我们 得 准备 : 大 虾六 到 九 只 、 盐 一 茶匙 、 已 搾 好 的 柠檬汁 三 汤匙 、 泰国 柠檬 叶三叶 、 柠檬 香草 一 根 、 鱼酱 两 汤匙 、 辣椒 6 粒 ， 纯净 水 4量杯 、 香菜 半量杯 和 草菇 10 个 。
    ```
    # label_file
    ```
    S 冬阴功 是 泰国 最 著名 的 菜 之一 ， 它 虽然 不 是 很 豪华 ， 但 它 的 味 确实 让 人 上瘾 ， 做法 也 不 难 、 不 复杂 。
    A 9 11|||W|||虽然 它|||REQUIRED|||-NONE-|||0

    S 首先 ， 我们 得 准备 : 大 虾六 到 九 只 、 盐 一 茶匙 、 已 搾 好 的 柠檬汁 三 汤匙 、 泰国 柠檬 叶三叶 、 柠檬 香草 一 根 、 鱼酱 两 汤匙 、 辣椒 6 粒 ， 纯净 水 4量杯 、 香菜 半量杯 和 草菇 10 个 。
    A 17 18|||S|||榨|||REQUIRED|||-NONE-|||0
    A 38 39|||S|||六|||REQUIRED|||-NONE-|||0
    A 43 44|||S|||四 量杯|||REQUIRED|||-NONE-|||0
    A 49 50|||S|||十|||REQUIRED|||-NONE-|||0
    ```
    """
    subprocess.check_call(
        ["python", "metrics/m2scorer/m2scorer.py", prediction_file, label_file]
    )
    # raise NotImplementedError
