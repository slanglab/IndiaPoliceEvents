import json, argparse
import numpy as np  
from collections import defaultdict
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, log_loss 

"""
This script used a version of the raw_annotations.jsonl file and the original annotations labels (ann_kill, ann_arrest) to evaluate. 

TODO: this needs to be cleaned up to use the final label version of the data. 
"""


def false_pos_rank_by_cross_entropy(false_pos_ids2pred, id2text, id2labels): 
    """
    y_true = 0 
    y_pred_pos >0.5 or higher 
    """
    print()
    print('num false pos=', len(false_pos_ids2pred))
    TOPK = 10
    top = sorted(false_pos_ids2pred.items(), key=lambda kv: -kv[1])[0:TOPK]
    for ids, probs in top: 
        print({'pred_probs': np.round(probs,3), 'true': 0, 'text': id2text[ids],'ids': ids, 'all_doc_labels': id2labels[ids]})

def false_negs_rank_by_cross_entropy(false_neg_ids2pred, id2text, id2labels):
    """
    Want top false negatives where 
    y_true = 1 
    y_pred_pos <0.5 or lower 
    """
    print()
    print('num false neg=', len(false_neg_ids2pred))
    TOPK = 10  
    top = sorted(false_neg_ids2pred.items(), key=lambda kv: kv[1])[0:TOPK]
    for ids, probs in top: 
        print({'pred_probs': np.round(probs,3), 'true': 1, 'text': id2text[ids],'ids': ids, 'all_doc_labels': id2labels[ids]})  

def locate_truthfile(eval_level):
    #TODO: should this be text_level or eval_level??

    if eval_level == "sent": 
        truthfile = '../data/test/test_sentlevel.jsonl'
    elif eval_level == "window": 
        truthfile = '../data/test/test_sentwindow.jsonl'
    elif eval_level == "doc": 
        truthfile = '../data/test/test_doclevel.jsonl'
    elif eval_level == "task2": #task2 also uses doc-level eval
        truthfile = '../data/test/test_doclevel.jsonl'

    print('truthfile = ', truthfile)
    return truthfile

def locate_predfile(query, model, text_level):
    if model == "mnli":
        predfile = '../predict/pred_roberta_mnli_{0}_{1}.jsonl'.format(text_level, query) 
    elif model == "keyword": 
        predfile = '../predict/pred_keywords_{0}_{1}.jsonl'.format(text_level, query)
    elif model == "msmarco": 
        predfile = '../predict/pred_electra_msmarco_{0}.jsonl'.format(query)
    
    #bm25 models 
    elif model == "bm25rm3_question": 
        # predfile = "../predict/result_bm25_rm3.json"
        predfile = "../predict/bm25rm3_question.json"
    elif model == "bm25rm3_keywords": 
        predfile = "../predict/bm25rm3_keywords.json"
    elif model == "bm25_question":
        predfile = "../predict/bm25_question.json"
    elif model == "bm25_keywords":
        predfile = "../predict/bm25_keywords.json"

    print('pred file = ', predfile)
    return predfile

def is_true_positive(query, dd, eval_level):
    if eval_level == "sent": 
        key = "sent_labels"
    elif eval_level in ["doc", "task2"]: 
        key = "doc_labels"

    #weird query is "ALL" which means "there was some police action"
    #this should be all the labels except for ann_fail
    #['ARREST', 'NA', 'OTHER_RESPONSE', 'KILL']
    if query == "ANY_ACTION":
        labels_of_interest = set(["ann_kill", "ann_arrest", "ann_other_response", "ann_na"])
        if len(set(dd[key]) & labels_of_interest) > 0: 
            return 1
        else: 
            return 0

    #For human annotations on sentences, it seems like both KILL=“Did police kill someone?” and OTHER_RESPONSE=“Did police use other force or violence?” 
    #correspond to the query I’m giving to MNLI “Police used violence.” 
    #so I'm evaluating on the union of these 
    elif query == "FORCE": 
        labels_of_interest = set(["ann_kill", "ann_other_response"])
        if len(set(dd[key]) & labels_of_interest) > 0: 
            return 1
        else: 
            return 0

    #normal queries, should see ['ARREST'] in the sent_labels if 'ARREST' is true 
    else: 
        if query in dd[key]: #the labels are there 
            return 1 
        else: 
            return 0

def load_truth_sent(truthfile):
    id2true = {}
    id2text = {} 
    id2labels = {}

    for line in open(truthfile, 'r'): 
        dd = json.loads(line)
        ids = (dd['doc_id'], dd['sent_id'])
        id2text[ids] = dd['sent_text']
        id2true[ids] = is_true_positive(query, dd, eval_level)
        id2labels[ids] = dd['sent_labels']
    return id2true, id2text, id2labels

def load_truth_window(truthfile): 
    pass

def load_truth_doc(truthfile, query, eval_level):
    id2true = {}
    id2text = {} 
    id2labels = {}

    for line in open(truthfile, 'r'): 
        dd = json.loads(line)
        docid = dd['doc_id']
        id2true[docid] = is_true_positive(query, dd, eval_level)
        id2text[docid] = dd['doc_text']
        id2labels[docid] = dd['doc_labels']
    return id2true, id2text, id2labels

def load_truth(truthfile, query, eval_level): 
    print('truth file = ', truthfile)

    if eval_level == "sent": 
        return load_truth_sent(truthfile)
    elif eval_level == "window": 
        return load_truth_window(truthfile)
    elif eval_level == "doc":
        return load_truth_doc(truthfile, query, eval_level)
    elif eval_level == "task2": #task2 is at the doc level as well 
        return load_truth_doc(truthfile, query, eval_level)

def load_pred_sent_evalSent(predfile):
    id2pred = {} #0/1 prediction
    id2predprob = {} #probability

    for i, line in enumerate(open(predfile, 'r')): 
        if len(line) == 0: continue 
        dd = json.loads(line)
        if i == 0 and 'query_form' in dd.keys():
            print('QUERY FORM=', dd['query_form'])
        ids = (dd['doc_id'], dd['sent_id'])
        id2pred[ids] = int(dd['pred'])
        if 'prob_pos' in dd.keys():
            id2predprob[ids] = dd['prob_pos']
        else: #keyword where we don't have probabilistic predictions
            id2predprob[ids] = int(dd['pred'])

    return id2pred, id2predprob 

def load_pred_window(predfile):
    pass

def load_pred_doc_evalDoc(predfile): 
    id2pred, id2predprob = {}, {}
    for line in open(predfile, 'r'): 
        dd = json.loads(line)
        doc_id = dd['doc_id']
        id2pred[doc_id] = dd['pred']
        if 'pred_pos' in dd.keys():
            id2predprob[doc_id] = dd['pred_pos']
        else: 
            id2predprob[doc_id] = dd['pred']
    return id2pred, id2predprob

def load_pred_sent_evalDoc(predfile):
    """
    Takes the max over sentence-level predictions
    """
    docid2pred_list = defaultdict(list) #0/1 prediction
    docid2predprob_list = defaultdict(list)  #probability

    for i, line in enumerate(open(predfile, 'r')): 
        dd = json.loads(line)
        if i == 0 and 'query_form' in dd.keys():
            print('QUERY FORM=', dd['query_form'])
        docid = dd['doc_id']

        if 'prob_pos' in dd.keys():
            pred_pos = dd['prob_pos']
        else: #keyword where we don't have probabilistic predictions
            pred_pos = int(dd['pred'])

        docid2pred_list[docid].append(dd['pred'])
        docid2predprob_list[docid].append(pred_pos)

    #take the max 
    id2pred, id2predprob = {}, {}
    for docid in docid2pred_list.keys(): 
        id2pred[docid] = max(docid2pred_list[docid])
        id2predprob[docid] = max(docid2predprob_list[docid])
    return id2pred, id2predprob 

def load_predictions(predfile, query, model, text_level, eval_level): 
    """
    Returns: 
        id2pred: keys: docid, values: hard classification predictions
        id2predprob: keys: docid, values: soft probabilistic predictions
    """

    if text_level == "sent" and eval_level == "sent": 
        return load_pred_sent_evalSent(predfile)

    elif text_level == "sent" and eval_level == "doc": 
        return load_pred_sent_evalDoc(predfile) 

    elif text_level == "doc" and eval_level == "doc": 
        if model in ["bm25rm3_question", "bm25rm3_keywords", "bm25_question", "bm25_keywords"]: 
            #need to return "None" b/c bm25 doesn't have hard classification rankings
            return None, load_bm25_preds(predfile, query)
        else:
            return load_pred_doc_evalDoc(predfile)

    # elif text_level == "window": 
    #     return load_pred_window(truthfile)
    # elif text_level == "doc":
    #     return load_pred_doc(truthfile) 

def eval_classification(predfile, truthfile, query, model, text_level, eval_level, id2pred, id2predprob, id2true, id2text, id2labels): 
    if eval_level == "sent": 
        TOKEN_CUTOFF = 5 #evaluation ignores 
        print("**evaluation ignores sentences <= {0} tokens (by whitespace)".format(TOKEN_CUTOFF))
        print()

    assert len(id2pred) == len(id2true)
    id_list = []
    y_true = []
    y_pred = []
    y_pred_probs = []
    false_pos_ids2pred = {}
    false_neg_ids2pred = {}

    for ids in id2pred.keys():
        if eval_level == "sent": 
            #ignores short sentences
            toks = id2text[ids].split(' ')
            if len(toks) <= TOKEN_CUTOFF: continue 

        id_list.append(ids)
        true = id2true[ids]
        y_true.append(true)
        y_pred.append(id2pred[ids])
        pred_pos = id2predprob[ids]
        y_pred_probs.append(pred_pos)

        #false pos
        if pred_pos > 0.5 and true == 0: 
            false_pos_ids2pred[ids] = pred_pos

        #false negs
        if pred_pos < 0.5 and true == 1: 
            false_neg_ids2pred[ids] = pred_pos

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print('MODEL=', model)
    print('EVAL LEVEL=', eval_level)
    print('QUERY =', query)
    if query=='ALL': print('ALL EVAL =there is police activity, all labels except for FAIL')

    #number true
    print('y_true mean={0:.3f}, support={1}'.format(np.mean(y_true), len(y_true)))
    print('y_pred mean={0:.3f}, support={1}'.format(np.mean(y_pred), len(y_pred)))
    print('true num pos =', list(y_true).count(1))
    print('pred num pos =', list(y_pred).count(1))

    #precision
    pr = precision_score(y_true, y_pred)
    print('precision =', pr)

    #recall 
    recall = recall_score(y_true, y_pred)
    print('recall =', recall)

    #F1
    f1 = f1_score(y_true, y_pred) 
    print('f1=', f1)

    #acc 
    acc = accuracy_score(y_true, y_pred)
    print('accuracy=', acc) 

    #cross entropy 
    ll = log_loss(y_true, y_pred_probs)
    print('cross entropy=', ll)
    print("=="*20)

    #confusion matrix 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    false_pos_rank_by_cross_entropy(false_pos_ids2pred, id2text, id2labels)
    false_negs_rank_by_cross_entropy(false_neg_ids2pred, id2text, id2labels)
    
    #save metrics 
    obj = {"text_level": text_level, "eval_level": eval_level, "query": str(query), "model": str(model), 
        "predfile": predfile, "truthfile": truthfile,  
            "precision": float(pr), "recall": float(recall), "f1": float(f1), 
            "acc": float(acc), "cross-entropy": float(ll), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)} 
    print(type(obj))

    outfile_name = "saved_metrics/{0}_{1}_{2}_eval={3}.json".format(model, query, text_level, eval_level)
    json.dump(obj, open(outfile_name, 'w'))
    print('saved to ->', outfile_name)

def create_ranked_list(y_true, y_score): 
    """
    Ranks based on the scores 
    """
    assert len(y_true) == len(y_score)
    true_with_score = [(x, y) for x, y in zip(y_true, y_score)]
    ranked_list = sorted(true_with_score, key=lambda kv: -kv[1])
    return ranked_list

def test_create_ranked_list(): 
    y_true = [1, 1, 0, 0]
    y_score = [0.7, 0.2, 0.5, 0.4]
    assert create_ranked_list(y_true, y_score) == [(1, 0.7), (0, 0.5), (0, 0.4), (1, 0.2)]
    
    y_true = [1, 1, 1, 0, 0]
    y_score = [0.2, 0.9, 0.5, 0.3, 0.4]
    assert create_ranked_list(y_true, y_score) == [(1, 0.9), (1, 0.5), (0, 0.4), (0, 0.3), (1, 0.2)]

def propRead_at_recall(y_true, y_score, recall_target=0.95): 
    """
    Proportion of the corpus read to get that percentage recall 
    
    e.g. PropRead@Recall95 means proportion of the corpus read 
    to achieve 95% recall 
    """
    assert len(y_true) == len(y_score) 
    
    #brute force way, go threshold by threshold 
    ranked_list = create_ranked_list(y_true, y_score)
    y_true = [x[0] for x in ranked_list] 
    for docsRead in range(len(y_true)): 
        y_pred = [1 for x in range(docsRead)]+[0 for x in range(len(y_true)-docsRead)]
        recall = recall_score(y_true, y_pred)
        if recall >= recall_target: 
            return docsRead/len(y_true)
    
    #end condition, have to read all the docs 
    return 1.0

def test_propRead_at_recall():
    y_true = [1, 1, 1, 0, 0]
    y_score = [0.2, 0.9, 0.5, 0.3, 0.4]
    #[(1, 0.9), (1, 0.5), (0, 0.4), (0, 0.3), (1, 0.2)]
    assert propRead_at_recall(y_true, y_score, recall_target=0.95) == 1.0
    
    y_true = [1, 1, 1, 0, 0, 0]
    y_score = [0.9, 0.9, 0.9, 0.6, 0.6, 0.6]
    assert propRead_at_recall(y_true, y_score, recall_target=0.95) == 3/6
    
    y_true = [1, 1, 1, 0, 0, 0]
    y_score = [0.9, 0.9, 0.5, 0.6, 0.4, 0.4]
    assert propRead_at_recall(y_true, y_score, recall_target=0.95) == 4/6
    
    y_true = [0, 1]
    y_score = [0.6, 0.4]
    assert propRead_at_recall(y_true, y_score, recall_target=0.95) == 2/2
    
    y_true = [1, 0]
    y_score = [0.6, 0.4]
    assert propRead_at_recall(y_true, y_score, recall_target=0.95) == 1/2

def max_f1_score(y_true, y_score):
    """
    Given a ranked list, calculates F1 at every threshold and returns the max 
    """
    maxF1 = -1 
    ranked_list = create_ranked_list(y_true, y_score)
    y_true = [x[0] for x in ranked_list] 
    for thresh in range(len(y_true)): 
        y_pred = [1 for x in range(thresh)]+[0 for x in range(len(y_true)-thresh)]
        f1 = f1_score(y_true, y_pred)
        if f1 > maxF1: 
            maxF1 = f1
    return maxF1

def test_max_f1_score():
    y_true = [1, 1, 1, 0, 0]
    y_score = [0.2, 0.9, 0.5, 0.3, 0.4]
    #ranked_list = [(1, 0.9), (1, 0.5), (0, 0.4), (0, 0.3), (1, 0.2)]
    assert max_f1(y_true, y_score) ==0.8 #threshold 2 should give max F1 score 

def eval_task2(predfile, truthfile, query, model, text_level, eval_level, id2scores, id2true, id2text, id2labels):
    print('MODEL=', model)
    print('EVAL LEVEL=', eval_level)
    print('QUERY =', query)

    # print('len(id2scores)', len(id2scores))
    print('len(id2text)', len(id2text)) 
    print()

    #load stuff 
    y_true = []
    y_score = []
    for docid in id2true.keys():
        y_true.append(id2true[docid])
        
        #all docids must be ints 
        assert type(docid) == int
        x = list(id2scores.keys())[0]
        assert type(x) == int

        #some models (like bm25) give a score of 0 if there is no match 
        if id2scores.get(docid) == None: 
            y_score.append(0.0)
        else:
            y_score.append(id2scores[docid])
    #     ipdb.set_trace()

    # ipdb.set_trace()

    #average precision
    ap = average_precision_score(y_true, y_score)
    print('average precision=', ap)

    #PropRead@Recall95
    propread_metric = propRead_at_recall(y_true, y_score, recall_target=0.95)
    print('PropRead@95= ', propread_metric)

    #maxF1
    maxF1 = max_f1_score(y_true, y_score)
    print('maxF1=', maxF1)

    #save metrics 
    obj = {"text_level": text_level, "eval_level": eval_level, "query": str(query), "model": str(model), 
        "predfile": predfile, "truthfile": truthfile,  
            "ave_precision": float(ap), "PropRead@95": float(propread_metric), "maxF1": float(maxF1)} 
    print(type(obj))

    outfile_name = "saved_metrics/{0}_{1}_{2}_eval={3}.json".format(model, query, text_level, eval_level)
    json.dump(obj, open(outfile_name, 'w'))
    print('saved to ->', outfile_name)

    print("=="*20)
    print() 

def load_msmarco_preds(predfile, query):
    id2scores = {} #keys: docids, values: max passage score 
    for line in open(predfile, 'r'): 
        dd = json.loads(line)
        doc_id = int(dd['doc_id'])
        max_passage_prob = max(dd['prob_pos'])
        id2scores[doc_id] = max_passage_prob
    return id2scores 

# def load_bm25_preds_OLD(predfile, query): 
#     id2scores = {} #keys: docids, values: max passage score 
#     doc_preds = json.load(open(predfile, 'r'))[query]
#     for doc, score in doc_preds:
#         id2scores[doc] = score 
#     print('len(id2scores)=', len(id2scores))
#     return id2scores

def load_bm25_preds(predfile, query): 
    id2scores = {} #keys: docids, values: max passage score 
    doc_preds = json.load(open(predfile, 'r'))[query]
    for doc, score, text in doc_preds:
        id2scores[int(doc)] = score 
    print('len(id2scores)=', len(id2scores))
    return id2scores

def load_mnli_preds_task2(predfile, query): 
    """
    The score (the probability) for the document is the 
    max of all the sentence-level probabilities  
    """
    id2scores = {}
    docid2predprob_list = defaultdict(list)  #probability

    for i, line in enumerate(open(predfile, 'r')): 
        dd = json.loads(line)
        docid = dd['doc_id']
        docid2predprob_list[docid].append(dd['prob_pos'])

    #take the max 
    for docid in  docid2predprob_list.keys(): 
        id2scores[docid] = max(docid2predprob_list[docid])
    return id2scores  

def load_predictions_rank(predfile, query, model):
    if model == "msmarco": 
        return load_msmarco_preds(predfile, query)
    elif model == "mnli": 
        return load_mnli_preds_task2(predfile, query)

    elif model in ["bm25rm3_question", "bm25rm3_keywords", "bm25_question", "bm25_keywords"]: 
        return load_bm25_preds(predfile, query)

def go_one_eval(query, model, text_level, eval_level): 
    truthfile = locate_truthfile(eval_level)
    predfile = locate_predfile(query, model, text_level)  

    if eval_level in ["sent", "doc"]:
        id2pred, id2predprob, = load_predictions(predfile, query, model, text_level, eval_level)
        id2true, id2text, id2labels= load_truth(truthfile, query, eval_level)
        eval_classification(predfile, truthfile, query, model, text_level, eval_level, id2pred, id2predprob, id2true, id2text, id2labels)
    elif eval_level in ["task2"]: 
        id2scores = load_predictions_rank(predfile, query, model) 
        truthfile = locate_truthfile(eval_level) 
        id2true, id2text, id2labels = load_truth(truthfile, eval_level)
        eval_task2(predfile, truthfile, query, model, text_level, eval_level, id2scores, id2true, id2text, id2labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="ARREST, KILL, etc.", type=str)
    parser.add_argument("model", help="[keyword, mnli, bm25, msmarco]", type=str)
    parser.add_argument("text_level", help="[sent, window, doc]", type=str)
    parser.add_argument("eval_level", help="[sent, doc, task2]", type=str)
    args = parser.parse_args()

    query = args.query
    model = args.model 
    text_level = args.text_level 
    eval_level = args.eval_level 

    go_one_eval(query, model, text_level, eval_level)