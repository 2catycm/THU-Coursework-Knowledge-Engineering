import re
from typing import *
import subprocess


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
    # subprocess.check_call(
    #     ["python", "metrics/m2scorer/m2scorer.py", prediction_file, label_file]
    # )
    # 执行命令并捕获输出
    result = subprocess.check_output(
        ["python", "metrics/m2scorer/m2scorer.py", prediction_file, label_file],
        text=True  # 直接获取文本输出，无需解码
    )

    # 打印原始输出
    if verbose:
        print("m2scorer评测中:")
        print(result)

    # 使用正则表达式提取指标
    metrics = {}
    metrics_pattern = re.compile(r'(Precision|Recall|F0\.5)\s*:\s*([\d\.]+)')
    matches = metrics_pattern.findall(result)
    
    # 将匹配的结果转换为字典
    for key, value in matches:
        metrics[key] = float(value)

    return metrics
