import torch
from utils.data import read_data_for_predict
from utils.imap_qa import calc_map1,calc_mrr1
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

def eval_model(model, predict_file_path, batch_predict_inputs):

    model.eval()
    all_preds = []
    with torch.no_grad():  # 插在此处
        for i in tqdm(range(len(batch_predict_inputs))):
            inputs = batch_predict_inputs[i]
            '''
            batch_tokenized = tokenizer.batch_encode_plus(inputs, add_special_tokens=True,
                                                          max_length=args.max_seq_len, padding='max_length',
                                                          truncation=True)  # tokenize、add special token、pad
            '''
            input_ids = torch.tensor(inputs['input_ids'])
            token_type_ids = torch.tensor(inputs['token_type_ids'])
            attention_mask = torch.tensor(inputs['attention_mask'])

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()

            outputs = model(False, input_ids, token_type_ids, attention_mask)
            all_preds.extend(outputs[:, 1])

    t_f = read_data_for_predict(predict_file_path)
    score = calc_map1(t_f, all_preds)
    score2 = calc_mrr1(t_f, all_preds)
    return score,score2
    #print(predict_file_path + " MAP: " + str(score * 1))
    #print(predict_file_path + " MRR: " + str(score2 * 1))