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
            outputs = model(inputs)
            all_preds.extend(outputs)

    t_f = read_data_for_predict(predict_file_path)
    score = calc_map1(t_f, all_preds)
    score2 = calc_mrr1(t_f, all_preds)
    return score,score2
    #print(predict_file_path + " MAP: " + str(score * 1))
    #print(predict_file_path + " MRR: " + str(score2 * 1))