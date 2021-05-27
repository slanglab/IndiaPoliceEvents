"""
Keyword matching 

Sent level: 
    - keywords occur in the same sentence 

Document level: 
    - keywords have to occur in the same document 

"""
from datetime import date
import logging
logger = logging.getLogger(__name__) 

import json, argparse, os, re 

def return_regex(text, keyword_list): 
    regex =  "".join([p+"|" for p in keyword_list]).strip("|")
    if re.search(regex, text, re.IGNORECASE) == None:
        is_match = False
    else: 
        is_match = True 
    return is_match

def kill_classify(sent, keyword_dict):
    """
    If both (a) police keyword, and (b) kill keyword occur 
    in the same piece of text, classify it as a positive 
    """
    is_match = return_regex(sent, keyword_dict['police']+keyword_dict['police_acronyms'])
    if is_match == False: 
        return is_match
    is_match = return_regex(sent, keyword_dict['kill'])
    return is_match 
    
def arrest_classify(sent, keyword_dict):
    is_match = return_regex(sent, keyword_dict['police']+keyword_dict['police_acronyms'])
    if is_match == False: 
        return is_match
    is_match = return_regex(sent, keyword_dict['arrest'])
    return is_match 

def intervene_classify(sent, keyword_dict):
    """
    TODO: do we want to make this better? should I add a "NOT:  
    
    just the intervene words will be potentially high-recall, lower precision
    """
    is_match = return_regex(sent, keyword_dict['police']+keyword_dict['police_acronyms'])
    if is_match == False: 
        return is_match
    is_match = return_regex(sent, keyword_dict['intervene'])
    return is_match 

def other_force_classify(sent, keyword_dict): 
    """
    Other force needs to match the NLI query "Police used violence."
    """
    is_match = return_regex(sent, keyword_dict['police']+keyword_dict['police_acronyms'])
    if is_match == False: 
        return is_match
    is_match = return_regex(sent, keyword_dict['other_force']+keyword_dict['kill'])
    return is_match

def all_classify(sent, keyword_dict): 
    """
    ANY_ACTION= "Police did something"

    Baseline is just if there are any police keywords at all 
    """
    is_match = return_regex(sent, keyword_dict['police']+keyword_dict['police_acronyms'])
    return is_match 

def test_kill_classify(): 
    sent = "This is a negative sent.".lower()
    assert kill_classify(sent, keyword_dict) == False 
    
    sent = "only police in this sentence".lower()
    assert kill_classify(sent, keyword_dict) == False 
    
    sent = "Police killed civilians.".lower()
    assert kill_classify(sent, keyword_dict) == True 
    
    sent = "The people were murdered by constables.".lower()
    assert kill_classify(sent, keyword_dict) == True
    
    sent = "The GRP are responsible for killing people.".lower() #test acronyms 
    assert kill_classify(sent, keyword_dict) == True
    
def test_arrest_classify(): 
    sent = "This is a negative sent.".lower()
    assert arrest_classify(sent, keyword_dict) == False 
    
    sent = "only police in this sentence".lower()
    assert arrest_classify(sent, keyword_dict) == False 
    
    sent = "The people were arrested by many policemen.".lower()
    assert arrest_classify(sent, keyword_dict) == True 
    
    sent = "Many arrests occurred by the STF .".lower()
    assert arrest_classify(sent, keyword_dict) == True 
    
def test_other_force_classify(): 
    sent = "This is a negative sent.".lower()
    assert other_force_classify(sent, keyword_dict) == False 
    
    sent = "only police in this sentence".lower()
    assert other_force_classify(sent, keyword_dict) == False 
    
    sent = "Police brutally beat many people.".lower()
    assert other_force_classify(sent, keyword_dict) == True 
    
    sent = "The men were choked by one police officer who is now under custody.".lower()
    assert other_force_classify(sent, keyword_dict) == True
    
    sent = "No one was beat today." #no police but has the other keyword
    assert other_force_classify(sent, keyword_dict) == False
    
def test_intervene_classify(): 
    sent = "This is a negative sent.".lower()
    assert intervene_classify(sent, keyword_dict) == False 
    
    sent = "only police in this sentence".lower()
    assert intervene_classify(sent, keyword_dict) == False 
    
    sent = "Police stood by and watched as everyone was murdered.".lower()
    assert intervene_classify(sent, keyword_dict) == True 
    
    sent = "The constables did not intervene.".lower() 
    assert intervene_classify(sent, keyword_dict) == True  

def test_all_classify(): 
    sent = "This is a negative sent.".lower()
    assert all_classify(sent, keyword_dict) == False
    
    sent = "only police in this sentence".lower()
    assert all_classify(sent, keyword_dict) == True
    
    sent = "Cops killed and beat people.".lower()
    assert all_classify(sent, keyword_dict) == True

def match_one_text(query, text, keyword_dict): 
    if query == "KILL":
        is_match = kill_classify(text, keyword_dict)
    elif query == "ARREST": 
        is_match = arrest_classify(text, keyword_dict)
    elif query == "FAIL": 
        is_match = intervene_classify(text, keyword_dict)
    elif query == "FORCE": 
        is_match = other_force_classify(text, keyword_dict)
    elif query == "ANY_ACTION": 
        is_match = all_classify(text, keyword_dict)
    return is_match

def save_preds(outfile_name, is_match, query, dd): 
    #save predictions 
    if text_level == "sent": 
        obj = {'doc_id': dd['doc_id'], 'sent_id': dd['sent_id'], 'model': "keyword_sent",
                'text_level': text_level, 'pred': int(is_match), 'query': query}
    elif text_level == "doc": 
        obj = {'doc_id': dd['doc_id'], 'model': "keyword_doc",
                'text_level': text_level, 'pred': int(is_match), 'query': query}
    with open(outfile_name, 'a+') as w:
        json.dump(obj, w)
        w.write('\n')

def which_infile(text_level): 
    if text_level == "sent": 
        infile = '../data/test/test_sentlevel.jsonl' 
    elif text_level == "doc": 
        infile = '../data/test/test_doclevel.jsonl' 
    logger.info('loading-> '+infile)
    return infile  

def go_matches(query, text_level): 
    if args.swarm2: 
        outfile_name = "/mnt/nfs/work1/brenocon/kkeith/event_retrieval/predict/pred_keywords_{0}_{1}.jsonl".format(text_level, query)
    else: 
        outfile_name = "../predict/pred_keywords_{0}_{1}.jsonl".format(text_level, query) 
    print("outfile=", outfile_name)
    if os.path.exists(outfile_name): os.remove(outfile_name)

    keyword_dict = json.load(open('manual_keyword_list.json', 'r'))
    logger.info('loaded keywords')

    #load infile
    infile = which_infile(text_level)

    for line in open(infile, 'r'): 
        dd = json.loads(line)
        if text_level == "sent":
            text = dd['sent_text']
        elif text_level == "doc":
            text = dd['doc_text']

        text = text.lower()
        is_match = match_one_text(query, text, keyword_dict)
        save_preds(outfile_name, is_match, query, dd)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", help="e.g. ARREST.sent or KILL.window", type=str)
    parser.add_argument("--swarm2", help="use if swarm2 so saves to correct partition", action='store_true')
    args = parser.parse_args()
    ss = args.query_text.strip().split('.')
    query = ss[0]
    text_level = ss[1]
    assert text_level in ['sent', 'doc'] #no doc-level for this model 
    assert query in ["KILL", "ARREST", "FAIL", "FORCE", "ANY_ACTION"] 

    logging_set_up()
    logger.info("query="+ query+ "; text_level="+text_level)
    go_matches(query, text_level)
