from argparse import ArgumentParser
import os
import torch
import pdb
import json

from transformers import AutoTokenizer

from main_loop import main_loop, prediction, main_parser
from models import QAReasoner
from hotpotqa.hotpot_evaluate_utils import update_answer

def logits2span(start_logits, end_logits, top_k=5):
    top_start_logits, top_start_indices = torch.topk(start_logits.squeeze_(0), k=top_k)
    top_end_logits, top_end_indices = torch.topk(end_logits.squeeze_(0), k=top_k)
    ret = []
    for start_pos in top_start_indices:
        for end_pos in top_end_indices:
            if end_pos - start_pos < 0:
                adds = -100000
            elif end_pos - start_pos > 80:
                adds = -20
            else:
                adds = 0
            ret.append((adds + start_logits[start_pos] + end_logits[end_pos], start_pos, end_pos))
    ret.sort(reverse=True)
    return ret[0][1], ret[0][2] + 1

def eval_newsqa(pred, root_dir):
    with open(os.path.join(root_dir, 'data', 'newsqa_test.json'), 'r') as fin:
        gt = json.load(fin)
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    for k, v in gt.items():
        if k not in pred:
            continue
        pred_v = pred[k]
        if v == 'None':
            v = 'noanswer'
        if pred_v == 'None':
            pred_v = 'noanswer'
        update_answer(metrics, pred_v, v)
    N = len(gt)
    for k in metrics.keys():
        metrics[k] /= N
    return metrics

if __name__ == "__main__":
    print('Please confirm the newsqa data are ready by ./newsqa/process_newsqa.py!')
    print('=====================================')
    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = ArgumentParser(add_help=False)
    # ------------ add dataset-specific argument ----------
    parser.add_argument('--only_predict', action='store_true')
    # ---------------------------------------------
    parser = main_parser(parser)
    parser.set_defaults(
        train_source = os.path.join(root_dir, 'data', 'newsqa_train_roberta-large.pkl'),
        test_source = os.path.join(root_dir, 'data', 'newsqa_test_roberta-large.pkl')
    )
    config = parser.parse_args()
    config.reasoner_cls_name = 'QAReasoner'
    if not config.only_predict: # train
        main_loop(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    ans = {}
    for qbuf, dbuf, buf, relevance_score, ids, output in prediction(config):
        _id = qbuf[0]._id
        start, end = logits2span(*output)
        ans_ids = ids[start: end]
        ans[_id] = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ans_ids)).replace('</s>', '').replace('<pad>', '').strip()
        # supporting facts  
        # sp[_id] = extract_supporing_facts(config, buf, relevance_score, start, end)
    with open(os.path.join(config.tmp_dir, 'pred.json'), 'w') as fout:
        json.dump(ans, fout)
        print(eval_newsqa(ans, root_dir))

