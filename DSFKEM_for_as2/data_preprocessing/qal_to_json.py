'''
question answer label
'''

'''
status:train, dev, test

'''

import json


def read_data(filename, status):
    with open(filename, 'r', encoding="utf8") as datafile:
        res = []
        count = 0
        for line in datafile:
            count = count + 1
            if count == 1:
                continue
            line = line.strip().split('\t')
            # print(line)
            dic = {}
            if status == 'train':
                dic["id"] = str(10000000 + count - 1)
            elif status == 'dev':
                dic["id"] = str(20000000 + count - 1)
            elif status == 'test':
                dic["id"] = str(30000000 + count - 1)

            dic["question"] = line[0].lstrip()
            dic["answer"] = line[1].lstrip()
            dic["label"] = float(line[2])
            res.append(dic)
    return res


if __name__ == '__main__':
    train_filename = "../data/SelQA/raw/selqa_train.tsv"
    dev_filename = "../data/SelQA/raw/selqa_valid.tsv"
    test_filename = "../data/SelQA/raw/selqa_test.tsv"

    train_object = read_data(train_filename, 'train')
    result_train = {}
    result_train["data"] = train_object
    train_output_path = '../data/SelQA/json/selqa_train.json'
    json.dump(result_train, open(train_output_path, 'w', encoding='utf-8'))

    dev_object = read_data(dev_filename, 'dev')
    result_dev = {}
    result_dev["data"] = dev_object
    dev_output_path = '../data/SelQA/json/selqa_dev.json'
    json.dump(result_dev, open(dev_output_path, 'w', encoding='utf-8'))

    test_object = read_data(test_filename, 'test')
    result_test = {}
    result_test["data"] = test_object
    test_output_path = '../data/SelQA/json/selqa_test.json'
    json.dump(result_test, open(test_output_path, 'w', encoding='utf-8'))
