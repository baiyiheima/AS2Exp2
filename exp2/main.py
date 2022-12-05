# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def read_concept_embedding(embedding_path):
    fin = open(embedding_path, encoding='utf-8')
    info = [line.strip() for line in fin]
    dim = len(info[0].split(' ')[1:])
    n_concept = len(info)
    embedding_mat = []
    id2concept, concept2id = [], {}
    # add padding concept into vocab
    id2concept.append('<pad_concept>')
    concept2id['<pad_concept>'] = 0
    embedding_mat.append([0.0 for _ in range(dim)])
    for line in info:
        concept_name = line.split(' ')[0]
        embedding = [float(value_str) for value_str in line.split(' ')[1:]]
        assert len(embedding) == dim and not np.any(np.isnan(embedding))
        embedding_mat.append(embedding)
        concept2id[concept_name] = len(id2concept)
        id2concept.append(concept_name)
    embedding_mat = np.array(embedding_mat, dtype=np.float32)
    return id2concept, concept2id, embedding_mat
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    '''
      id2concept, concept2id, concept_embedding_mat = read_concept_embedding(
        "D:/2022/ExpTwo/AS2Exp2/exp2/retrieve_concepts/KB_embeddings/wn_concept2vec.txt")
    
    import paddle.fluid as fluid

    class_num = 7
    x = fluid.layers.data(name='x', shape=[3, 10], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    predict = fluid.layers.fc(input=x, size=class_num, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    
    import torch

    import torch.nn.functional as F

    x = torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

    y1 = F.softmax(x, dim=0)  # 对每一列进行softmax
    print(y1)

    y2 = F.softmax(x, dim=1)  # 对每一行进行softmax
    print(y2)

    x1 = torch.Tensor([1, 2, 3, 3])
    print(x1)

    y3 = F.softmax(x1, dim=0)  # 一维时使用dim=0，使用dim=1报错
    print(y3)
    '''
    steps = None
    map = 1
    mrr = 1
    with open('output/result.txt', mode='a+', encoding='utf-8') as file_obj:
        if steps != None :
            file_obj.write("Steps:{} Eval performance:\n* MAP: {}\n* MRR: {}\n".format(steps, map, mrr))
        else :
            file_obj.write("Final Eval performance:\n* MAP: {}\n* MRR: {}\n".format(map, mrr))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
