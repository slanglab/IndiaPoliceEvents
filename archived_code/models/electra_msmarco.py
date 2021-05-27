"""
BERT-based model trained on MS MARCO with a cross-encoder architecture 

Code from: https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/information-retrieval#pre-trained-cross-encoders-re-ranker
"""
from datetime import date
import logging
logger = logging.getLogger(__name__) 

import ipdb 
import json, argparse, os, sys 
import numpy as np 
from sentence_transformers import CrossEncoder

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

def load_msmarco_model(): 
    model = CrossEncoder('cross-encoder/ms-marco-electra-base', max_length=512)
    return model 

def format_data_one_doc(one_doc_info, query_form, current_doc):
    """
    Output: 
    {"encoded_doc": [('Did police do something?', 'Here is a bad example.'), ('Did police do something?', 'Police officers arrested many people.'), ('Did police do something?', 'There were many complaints about the police station.'), ('Did police do something?', 'blah gross.')],
            "doc_id": 456, 
            "query_form": query_form, 
            "window_ids": [] 
            }


    """
    encoded_doc = []
    window_ids = []
    for dd in one_doc_info: 
        assert dd['doc_id'] == current_doc
        encoded_doc.append((query_form, dd['window_text']))
        window_ids.append(dd['window_id'])

    obj = {"encoded_doc": encoded_doc, "doc_id": current_doc, "query_form": query_form, "window_ids": window_ids}
    return obj 

def create_doc_list(query_form):
    """
    Output: 
        list, entries are dict (each entry is one doc)
            {"encoded_doc": [('Did police do something?', 'Here is a bad example.'), ('Did police do something?', 'Police officers arrested many people.'), ('Did police do something?', 'There were many complaints about the police station.'), ('Did police do something?', 'blah gross.')],
            "doc_id": 456, 
            "query_form": query_form 
            }
    """
    docs_formatted = []
    infile = '../data/test/test_sentwindow.jsonl'

    current_doc = None 
    one_doc_info = []

    for line in open(infile, 'r'):
        dd = json.loads(line)

        #reset every new doc  
        if dd['window_id'] == 0: 
            #clear the old doc 
            if current_doc != None: 
                obj = format_data_one_doc(one_doc_info, query_form, current_doc)
                docs_formatted.append(obj)

            #reset 
            current_doc = dd['doc_id']
            one_doc_info = []

        one_doc_info.append(dd)

    #clear again at the very end 
    obj = format_data_one_doc(one_doc_info, query_form, current_doc)
    docs_formatted.append(obj)
    logger.info('loaded {0} docs and formatted for ms marco'.format(len(docs_formatted)))
    return docs_formatted

def model_predict(model, encoded_doc):
    """
    This scores will output a score between 0.0 and 1.0
    for each passage of whether or not it is "relevant" to the query 
    """
    scores = model.predict(encoded_doc)
    return scores 

def save_preds(doc_obj, prob_pos, outfile_name, query, query_form):
    obj = {'doc_id':doc_obj['doc_id'], 
            'model': 'ms-marco-electra-base',
            'text_level': 'window', 
            'prob_pos': [float(x) for x in prob_pos], 
            'query': query, 
            'query_form': query_form}
    json.dump(obj, open(outfile_name, 'a+'))
    with open(outfile_name, 'a+') as w: 
        w.write('\n') 

def go_one_query(query): 
    if args.swarm2: 
        outfile_name = "/mnt/nfs/work1/brenocon/kkeith/event_retrieval/predict/pred_electra_msmarco_{0}.jsonl".format(query)
    else: 
        outfile_name = "../pred_electra_msmarco_{0}.jsonl".format(query)
    print("outfile=", outfile_name)
    if os.path.exists(outfile_name): os.remove(outfile_name)

    query_form = label2declarative[query]
    docs_formatted = create_doc_list(query_form)

    model = load_msmarco_model()

    for i, doc_obj in enumerate(docs_formatted):
        prob_pos =  model_predict(model, doc_obj['encoded_doc'])
        save_preds(doc_obj, prob_pos, outfile_name, query, query_form)
        logger.info('preds done doc '+str(i))

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
    log_file = log_dir + str(today)+ "-"+ str(os.path.basename(__file__)) + '-' + args.query + ".log"
    file_handler = logging.FileHandler(log_file) 
    logger.addHandler(file_handler)
    print('saving log file to: ', log_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="ARREST, KILL, FAIL", type=str)
    parser.add_argument("--swarm2", help="use if swarm2 so saves to correct partition", action='store_true')
    args = parser.parse_args()

    query = args.query 
    assert query in label2declarative.keys()
    print("query=", query)

    logging_set_up()
    go_one_query(query)