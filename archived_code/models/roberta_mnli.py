"""
This code contains: 

Zero-shot: 
RoBERTa already finetuned on MNLI 
    Following: https://github.com/pytorch/fairseq/tree/master/examples/roberta#use-roberta-for-sentence-pair-classification-tasks 
"""
#import ipdb 
from datetime import date
import logging
logger = logging.getLogger(__name__) 

import json, argparse, os  
import torch
import numpy as np 
from fairseq.data.data_utils import collate_tokens

label2declarative = {
"ANY_ACTION": "Police did something.", 
"ARREST": "Police arrested someone.",
"FORCE": "Police used violence.", 
"KILL": "Police killed someone.", 
"FAIL": "Police failed to intervene." 
}

label2question = {
"ANY_ACTION": "Did police do anything?", 
"ARREST": "Did police arrest someone?",
"FORCE": "Did police use force or violence?", 
"KILL": "Did police kill someone?", 
"FAIL": "Did police fail to intervene?"
}

def load_roberta_mnli_model(): 
    """
    Load the (already fine-tuned) RoBERTa + MNLI model 
    """
    # Download RoBERTa already finetuned for MNLI
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
    return roberta 

def pred_roberta_mnli_batch(roberta, batch_of_pairs): 
    """
    Batched predictions with the RoBERTa MNLI model

    Inputs: 
    - roberta : torch pre-trained RoBERTa model 

    - batch_of_pairs : list of list, each entry is (sent + context, question)
        example 
        batch_of_pairs = [
            ['Police were there. Police killed civilians.', 'Police killed someone'],
            ['People died by police firing.', 'Police killed someone.']
            ]

    Output: 
    - prob_pos : probability the model assigns to "entailment"
    - pred_pos : 0 or 1, whether the model predicts positive, "entailment" (is argmax across the three classes)
    """
    label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'} #from RoBERTa people 

    roberta.eval()  # disable dropout for evaluation
    batch = collate_tokens(
        [roberta.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
    )
    logprobs = roberta.predict('mnli', batch)
    prob_pos = np.exp(logprobs.detach().numpy()[:, 2]) #probability of the "entailment"
    pred_pos = (logprobs.argmax(dim=1).detach().numpy() == 2).astype(int)

    assert len(batch_of_pairs) == len(prob_pos) == len(pred_pos)
    return prob_pos, pred_pos

def create_batches(query_form, text_level):
    """
    Output: 
    batch_list 
        List with list entries as dict 
            { batch_of_pairs : [
                ['Police were there. Police killed civilians.', 'Police killed someone'],
                ['People died by police firing.', 'Police killed someone.']
                ], 
            sent_ids : [1, 2, 3], #None if text_level = doc 
            window_ids: [1, 2, 3]
            doc_ids : [1000, 2134,...],
            query: }
    """
    batch_list = [] 
    if text_level == "sent": 
        infile = '../data/test/test_sentlevel.jsonl' 
    elif text_level == "window": 
        infile = '../data/test/test_sentwindow.jsonl'  

    singe_batch_count = 0 
    obj = {"batch_of_pairs": [], "sent_ids": [],
                    "window_ids": [], "doc_ids": []}

    for line in open(infile, 'r'):
        #go through each example 
        dd = json.loads(line)

        if text_level == "sent": 
            obj["batch_of_pairs"].append([dd["sent_text"], query_form])
            obj["sent_ids"].append(dd["sent_id"])
            obj["doc_ids"].append(dd["doc_id"])

        elif text_level == "window": 
            obj["batch_of_pairs"].append([dd["window_text"], query_form])
            obj["window_ids"].append(dd["window_id"])
            obj["doc_ids"].append(dd["doc_id"]) 

        singe_batch_count += 1
        #check if new batch 
        if singe_batch_count % BATCH_SIZE == 0: #new batch 
            singe_batch_count = 0
            #write
            batch_list.append(obj) 

            #reset  
            obj = {"batch_of_pairs": [], "sent_ids": [],
                    "window_ids": [], "doc_ids": []}

    #make the last batch smaller if necessary 
    if singe_batch_count != 0: 
        batch_list.append(obj) 

    logger.info('made {0} batches of size {1} '.format(len(batch_list), BATCH_SIZE))
    return batch_list 

def save_preds(batch, prob_pos, pred_pos, outfile_name, text_level, query_form, query):
    assert len(batch['batch_of_pairs']) == len(prob_pos) == len(pred_pos) 

    if text_level == "sent": 
        for sent_id, doc_id, prob, pred in zip(batch['sent_ids'], batch['doc_ids'], prob_pos, pred_pos): 
            obj = {'doc_id': doc_id, 'sent_id': sent_id, 'model': "roberta-mnli",
                    'text_level': text_level, 'prob_pos': float(prob), 'pred': int(pred), 
                    'query_form': query_form, 'query': query}
            json.dump(obj, open(outfile_name, 'a+'))
            with open(outfile_name, 'a+') as w: 
                w.write('\n')

    elif text_level == "window":  
        for window_id, doc_id, prob, pred in zip(batch['window_ids'], batch['doc_ids'], prob_pos, pred_pos): 
            obj = {'doc_id': doc_id, 'window_id': window_id, 'model': "roberta-mnli",
                    'text_level': text_level, 'prob_pos': float(prob), 'pred': int(pred), 
                    'query_form': query_form, 'query': query}
            json.dump(obj, open(outfile_name, 'a+'))
            with open(outfile_name, 'a+') as w: 
                w.write('\n')

def go_one_question(query, text_level):
    if args.swarm2: 
        outfile_name = "/mnt/nfs/work1/brenocon/kkeith/event_retrieval/predict/pred_roberta_mnli_{0}_{1}.jsonl".format(text_level, query)
    else: 
        outfile_name = "../predict/pred_roberta_mnli_{0}_{1}.jsonl".format(text_level, query)
    if os.path.exists(outfile_name): os.remove(outfile_name)

    print("outfile=", outfile_name)
    query_form = label2declarative[query]
    batch_list = create_batches(query_form, text_level)

    roberta = load_roberta_mnli_model()

    for i, batch in enumerate(batch_list): 
        prob_pos, pred_pos = pred_roberta_mnli_batch(roberta, batch['batch_of_pairs'])
        save_preds(batch, prob_pos, pred_pos, outfile_name, text_level, query_form, query)
        logger.info('preds done batch '+str(i))

    logger.info('wrote to-> '+outfile_name)

def logging_set_up(): 
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s — %(funcName)s:%(lineno)d — %(message)s")
    today = date.today()

    if args.swarm2: 
        log_dir = "/mnt/nfs/work1/brenocon/kkeith/event_retrieval/predict/logs/"
    else: 
        log_dir = "logs/"

    if not os.path.exists(log_dir): os.mkdir(log_dir)
    #made this log specific to the input 
    log_file = log_dir + str(today)+ "-"+ str(os.path.basename(__file__)) + '-' + args.query_text + ".log"
    file_handler = logging.FileHandler(log_file) 
    logger.addHandler(file_handler)
    print('saving log file to: ', log_file)

if __name__ == '__main__':
    BATCH_SIZE = 50 #not sure what optimal is for making fast and tradeoffs with memory   

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", help="ARREST, KILL, FAIL etc. with a dot", type=str)
    parser.add_argument("--swarm2", help="use if swarm2 so saves to correct partition", action='store_true')
    #parser.add_argument("query", help="ARREST, KILL, FAIL etc.", type=str)
    #parser.add_argument("text_level", help="choose from [window, sent]", type=str)
    args = parser.parse_args()
    ss = args.query_text.strip().split('.')
    query = ss[0]
    text_level = ss[1]
    assert query in label2declarative.keys()
    assert text_level in ['window', 'sent'] #no doc-level for this model 
    print("query=", query, "; text_level=", text_level)

    logging_set_up()
    go_one_question(query, text_level)



