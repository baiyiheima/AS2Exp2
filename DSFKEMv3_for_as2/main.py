
from utils.text_to_uri import standardized_uri
import numpy as np
from tqdm import tqdm
def read_concept_embedding(embedding_path):
    fin = open(embedding_path, encoding='utf-8')
    info = [line.strip() for line in fin]
    dim = len(info[1].split(' ')[1:])
    print(len(info))
    print(dim)
    n_concept = len(info)
    embedding_mat = []
    id2concept, concept2id = [], {}
    # add padding concept into vocab
    id2concept.append('<pad_concept>')
    concept2id['<pad_concept>'] = 0
    embedding_mat.append([0.0 for _ in range(dim)])
    count = 0
    for line in tqdm(info):
        if count == 0:
            count += 1
            continue
        concept_name = line.split(' ')[0]
        embedding = [float(value_str) for value_str in line.split(' ')[1:]]
        assert len(embedding) == dim and not np.any(np.isnan(embedding))
        embedding_mat.append(embedding)
        concept2id[concept_name] = len(id2concept)
        id2concept.append(concept_name)
    embedding_mat = np.array(embedding_mat, dtype=np.float32)
    return id2concept, concept2id, embedding_mat
if __name__ == '__main__':
    a,b,c = read_concept_embedding('../data/KB_embeddings/numberbatch-en.txt')
    print(len(a))