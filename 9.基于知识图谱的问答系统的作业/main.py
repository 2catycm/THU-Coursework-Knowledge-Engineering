import json
import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel


sorted_heads = None

def search_head_entity(kg: dict, question: str) -> str:
    """基于正向最大匹配的头实体识别
    Args:
        kg: 知识图谱字典，包含head2id等字段
        question: 待查询的问题文本
    Returns:
        匹配成功的头实体字符串，未找到返回None
    """
    # 获取所有可能的头实体
    global sorted_heads
    if sorted_heads is None:
        all_heads = list(kg['head2id'].keys())
        
        # 对头实体按长度排序，优先匹配长的实体
        sorted_heads = sorted(all_heads, key=len, reverse=True)
        print("头实体数量：", len(sorted_heads))
        print(sorted_heads[:5])
    # 遍历所有头实体，检查是否在问题中出现
    for head in sorted_heads:
        if head in question:
            return head
    
    # 如果没有找到匹配的头实体
    return ""


if __name__ == "__main__":
    # 读取问题，跳过第一行，问题和答案分开存储
    with open("data/raw/问题.txt", "r", encoding="utf-8") as f:
        questions = []
        answers = []
        for line in f.readlines()[1:]:
            question, answer = line.split("A：")
            questions.append(question.replace("Q：", ""))
            answers.append(answer.strip())
    # 读取知识图谱
    with open("data/processed/kg.json", "r", encoding="utf-8") as f:
        kg = json.load(f)
    # 加载模型和分词器
    # model = BertModel.from_pretrained("bert-base-chinese")
    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    
    # 遍历每个问题
    for idx, question in enumerate(questions):
        # 在知识图谱中找到问题中的实体
        head = search_head_entity(kg, question)
        # if head is None:
        if head=="":
            print("问题：", question)
            print("答案：", "无法找到实体")
            print("原答案：", answers[idx])
            print("---------------------------")
            continue
        # 找到实体后，找到实体对应的关系
        # relations = list(kg[head].keys())
        relations = list(kg["head2relations"][head])
        # 计算问题和关系的相似度，找到最相似的关系
        max_sim = 0
        max_relation = ""
        for relation in relations:
            fake_question = head + "的" + relation + "？"
            input_text = [question, fake_question]
            inputs = tokenizer(input_text, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            # 求解向量的余弦相似度
            question_vec = outputs.pooler_output[0].detach().numpy()
            fake_question_vec = outputs.pooler_output[1].detach().numpy()
            cos_sim = np.dot(question_vec, fake_question_vec) / (
                np.linalg.norm(question_vec) * np.linalg.norm(fake_question_vec)
            )
            if cos_sim > max_sim:
                max_sim = cos_sim
                max_relation = relation
        # 找到最相似的关系后，找到关系对应的答案
        # answer = kg[head][max_relation]
        answer = kg["head_relations2answers"][f"({head}, {max_relation})"]
        input_text = [answers[idx], answer]
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        # 求解向量的余弦相似度
        answer_vec = outputs.pooler_output[0].detach().numpy()
        fake_answer_vec = outputs.pooler_output[1].detach().numpy()
        cos_sim = np.dot(answer_vec, fake_answer_vec) / (
            np.linalg.norm(answer_vec) * np.linalg.norm(fake_answer_vec)
        )
        print("问题：", question)
        print("实体：", head)
        print("关系：", max_relation)
        print("预测答案：", answer)
        print("原答案：", answers[idx])
        print("答案和正确答案的余弦相似度 %.2f" % cos_sim)
        print("---------------------------")
