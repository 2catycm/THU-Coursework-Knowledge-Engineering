import json

def preprocess():
    kg = {}
    with open('../zhishime.json', 'r', encoding='utf-8') as f:
        #TODO
        raw_relation_data = list(map(json.loads, f))

    head2id = dict()
    tail2id = dict()
    relation2id = dict()
    relation_triplets = []

    for item in raw_relation_data:
        head = item['head']
        tail = item['tail']
        relation = item['relation']


        if head not in head2id:
            head2id[head] = len(head2id)
        if tail not in tail2id:
            tail2id[tail] = len(tail2id)
        if relation not in relation2id:
            relation2id[relation] = len(relation2id)

        relation_triplets.append((
            head2id[head], relation2id[relation], tail2id[tail]
        ))

    print(f'{len(head2id)} head entities, {len(tail2id)} tail entities, {len(relation2id)} relations.')
    print(f'{len(relation_triplets)} relation triplets')

    kg = {
        'head2id': head2id,
        'tail2id': tail2id,
        'relation2id': relation2id,
        'relation_triplets': relation_triplets
    }



    with open('./processed/kg.json', 'w', encoding='utf-8') as json_file:
        json.dump(kg, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    preprocess()