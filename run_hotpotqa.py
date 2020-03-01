from argparse import ArgumentParser
import os
import torch

from main_loop import main_loop, prediction, main_parser
from retriever import Retriever
from working_memory import WorkingMemory
from models import QAReasoner
from hotpotqa.hotpot_evaluate_utils import eval_func

def logits2span(start_logits, end_logits, top_k=3):
    top_start_logits, top_start_indices = torch.topk(start_logits.squeeze(0), k=top_k)
    top_end_logits, top_end_indices = torch.topk(end_logits.squeeze(0), k=top_k)
    ret = []
    for start_pos in top_start_indices:
        for end_pos in top_end_indices:
            if end_pos - start_pos < 0:
                adds = -10000
            elif end_pos - start_pos > 8:
                adds = -10
            else:
                adds = 0
            ret.append((adds + start_logits[start_pos] + end_logits[end_pos], start_pos, end_pos))
    ret.sort(reverse=True)
    return ret[0][1], ret[0][2] + 1

def extract_supporing_facts(config, buf, score, start, end):
    ret = []
    # the result sentence
    for i, sen_end in enumerate(buf.block_ends()):
        if end >= sen_end:
            if buf[i].blk_type > 0:
                ret.append(list(buf[i].origin))
            break
    # best 2 entity sentence
    for idx in score.argsort(descending=True):
        entity = buf[idx].origin[0]
        if buf[idx].blk_type > 0 and all([entity != fact[0] for fact in ret]):
            ret.append(list(buf[i].origin))
        if len(ret) >= 2:
            break
    # auxiliary sp
    gold_entities = [t[0] for t in ret]
    for i, blk in enumerate(buf):
        entity, sen_idx = blk.origin
        if sen_idx == 0 and entity in gold_entities and score[i] > config.sp_threshold:
            ret.append([entity, sen_idx])
    return ret

if __name__ == "__main__":
    print('Please confirm the hotpotqa data are ready by ./hotpotqa/process_hotpotqa.py!')
    print('=====================================')
    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = ArgumentParser(add_help=False)
    # ------------ add hotpotqa argument ----------
    parser.add_argument('--sp_threshold', type=int, default=0.8)
    parser.add_argument('--only_predict', action='store_true')
    # ---------------------------------------------
    parser = main_parser(parser)
    parser.set_defaults(
        train_source = os.path.join(root_dir, 'data', 'hotpotqa_train_roberta-base.pkl'),
        test_source = os.path.join(root_dir, 'data', 'hotpotqa_test_roberta-base.pkl'),
        introspect = True
    )
    config = parser.parse_args()
    if not config.only_predict: # train
        reasoner = QAReasoner.from_pretrained(config.model_name)
        main_loop(config, reasoner)

    sp, ans = {}, {}
    for qbuf, dbuf, buf, relevance_score, ids, output in prediction(config):
        _id = qbuf[0]._id
        start, end = logits2span(output)
        ans_ids = ids[start: end]
        ans[_id] = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(ans_ids))
        # supporting facts
        sp[_id] = extract_supporing_facts(config, buf, relevance_score, start, end)
    with open(os.path.join(config.tmp_dir, 'pred.json'), 'w') as fout:
        pred = {'answer': ans, 'sp': sp}
        json.dump(pred, fout)
        print(eval_func(pred, os.path.join(root_dir, 'data', 'hotpot_dev_distractor_v1.json')))

