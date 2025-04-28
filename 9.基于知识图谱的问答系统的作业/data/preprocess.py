import json
import os
from tqdm import tqdm

def preprocess():
    """预处理知识图谱数据，构建实体映射和三元组索引"""
    # 初始化数据结构
    kg = {
        "head2id": {},     # 头实体到ID的映射
        "tail2id": {},     # 尾实体到ID的映射  
        "relation2id": {}, # 关系到ID的映射
        "relation_triplets": [],  # 存储(hid, rid, tid)形式的三元组
        "head2relations": {},  # 头实体到它拥有的关系的映射
        "head_relations2answers": {},  # 头实体到关系和答案的映射
    }

    try:
        # 确保输出目录存在
        os.makedirs("./processed", exist_ok=True)
        
        # 读取原始JSONL文件（注意：使用..表示上级目录）
        with open("/home/ycm/repos/coursework/THU-Coursework-Knowledge-Engineering/9.基于知识图谱的问答系统的作业/zhishime.json", "r", encoding="utf-8") as f:
            # 逐行解析JSON对象（适用于JSON Lines格式）
            # raw_relation_data = [json.loads(line.strip()) for line in f.readlines()]
            
            # 等价写法（更易理解）：
            raw_relation_data = []
            for line in f:
                data = json.loads(line)
                raw_relation_data.append(data)

        # 遍历每个三元组进行索引构建
        bar = tqdm(raw_relation_data)
        for item in bar:
            head = item["head"]
            relation = item["relation"]
            
            # 处理包含换行符的尾实体（示例数据中的<br/>分隔符）
            tail = item["tail"].replace("\u003cbr/\u003e", ", ")  # 将HTML换行符转换为逗号分隔
            
            # 构建实体ID映射（自动递增分配ID）
            # head2id.setdefault等效写法，但更推荐当前写法
            if head not in kg["head2id"]:
                kg["head2id"][head] = len(kg["head2id"])
                
            if tail not in kg["tail2id"]:
                kg["tail2id"][tail] = len(kg["tail2id"])
                
            if relation not in kg["relation2id"]:
                kg["relation2id"][relation] = len(kg["relation2id"])
            
            # 构建三元组索引
            hid = kg["head2id"][head]
            rid = kg["relation2id"][relation]
            tid = kg["tail2id"][tail]
            kg["relation_triplets"].append((hid, rid, tid))

            # 构建头实体到关系的映射
            if head not in kg["head2relations"]:
                kg["head2relations"][head] = set()
            kg["head2relations"][head].add(relation)

            # 构建头实体到关系和答案的映射
            if (head, relation) not in kg["head_relations2answers"]:
                kg["head_relations2answers"][(head, relation)] = ""
            kg["head_relations2answers"][(head, relation)] += f"{tail}, "

        # 打印统计信息
        print(f"[统计] 头实体数: {len(kg['head2id'])} | 尾实体数: {len(kg['tail2id'])} | 关系类型数: {len(kg['relation2id'])}")
        print(f"[统计] 总三元组数: {len(kg['relation_triplets'])}")

        # 保存处理结果
        with open("./processed/kg.json", "w", encoding="utf-8") as json_file:
            json.dump(kg, json_file, 
                     ensure_ascii=False,  # 保留非ASCII字符原文
                     indent=4)           # 美化格式便于查看
            
    except FileNotFoundError:
        print("错误：未找到原始数据文件，请检查路径是否正确")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误：第{e.lineno}行数据格式异常，错误详情：{e.msg}")

if __name__ == "__main__":
    preprocess()
