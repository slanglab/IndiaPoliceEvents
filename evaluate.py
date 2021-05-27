"""
Evaluation script to accompany the paper 

Halterman, Keith, Sarwar, and O'Connor. "Corpus-Level Evaluation for Event QA:
The IndiaPoliceEvents Corpus Covering the 2002 Gujarat Violence." Findings of ACL, 2021. 

If you use this script, please cite the paper 

@inproceedings{halterman2021corpus,
author = {Halterman, Andrew and Keith, Katherine A. and Sarwar, Sheikh Muhammad, and O'Connor, Brendan}, 
title = {Corpus-Level Evaluation for Event {QA}:
The {IndiaPoliceEvents} Corpus Covering the 2002 {G}ujarat Violence},
booktitle = {{Findings of ACL}},
year = 2021}
"""
import argparse
import pandas as pd 
import numpy as np 
from collections import Counter 
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, log_loss  
from scipy.stats import spearmanr

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
    Returns: 
        Proportion of the corpus read to get that percentage recall 
        
        e.g. PropRead@Recall95 means proportion of the corpus read 
        to achieve 95% recall 

    Inputs: 
        y_true : array, length=number of documents 
            with integers 0, 1 indicating true binary classification 

        y_score : array, length=number of documents 
            with floats between 0.0 and 1.0 indicating the relevance scores 

        recall_target : recall achievement target 
            (e.g. recall_target=0.95 will return means proportion of the corpus read 
            to achieve 95% recall)
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

def task1_sentence_classification(label2pred, label2gold):
    """
    Prints F1 score, precision and recall for each label 
    """ 
    print('TASK 1 EVAL\n')
    for label in LABEL_LIST:
        print('CLASS=', label) 
        pred_array = label2pred[label]
        gold_array = label2gold[label] 
        print('sentence level F1=', f1_score(gold_array, pred_array))
        print('\t precision=', precision_score(gold_array, pred_array))
        print('\t recall=', recall_score(gold_array, pred_array))
        print()

def task2_document_ranking(label2pred, label2gold): 
    """
    Prints average precision and PropRead@RecallX for each label 
    """
    print('TASK 2 EVAL\n')
    for label in LABEL_LIST:
        print('CLASS=', label) 
        pred_score_array = label2pred[label]
        gold_array = label2gold[label] 
        print('average precision=', average_precision_score(gold_array, pred_score_array))
        print('PropRead@Recall95=', propRead_at_recall(gold_array, pred_score_array, recall_target=0.95))
        print()

def load_docid2date(): 
    metadata = pd.read_csv('data/final/metadata.csv')
    docid2date = {doc_id: date for doc_id, date in zip(metadata['doc_id'], metadata['date'])}
    return docid2date

def aggregate_by_date(label2pred, label2gold, docid_order, docid2date): 
    gold_label2date2count = {label: Counter() for label in LABEL_LIST}
    pred_label2date2count = {label: Counter() for label in LABEL_LIST}

    for label in LABEL_LIST:
        for docid, pred, gold in zip(docid_order, label2pred[label], label2gold[label]): 
            docid_date = docid2date[docid]
            gold_label2date2count[label][docid_date] += gold 
            pred_label2date2count[label][docid_date] += pred 

    return gold_label2date2count, pred_label2date2count

def test_aggregate_by_date(): 
    label2pred = {'KILL': [0, 1, 1]}
    label2gold = {'KILL': [1, 1, 1]}
    docid_order = [1, 2, 3]
    docid2date = {1: '2020-01', 2: '2020-01', 3: '2020-02'}
    true_gold_label2date2count = {'KILL': {'2020-01': 2, '2020-02': 1}}
    true_pred_label2date2count = {'KILL': {'2020-01': 1, '2020-02': 1}} 
    assert true_gold_label2date2count, true_pred_label2date2count == aggregate_by_date(label2pred, label2gold, docid_order, docid2date) 
 
def metric_aggregate_spearman_rho(gold_date2count, pred_date2count):
    assert len(gold_date2count) == len(pred_date2count)

    #put data in same date order 
    gold_arr = []
    pred_arr = []
    for date in gold_date2count.keys(): 
        gold_arr.append(gold_date2count[date])
        pred_arr.append(pred_date2count[date])

    #calculate spearmanr
    rho, pval = spearmanr(gold_arr, pred_arr)
    return rho

def test_metric_aggregate_spearman_rho(): 
    gold_date2count = {'2020-01': 2, '2020-02': 1, '2020-03': 5}
    pred_date2count = {'2020-01': 1, '2020-02': 1, '2020-03': 6}
    assert np.abs(metric_aggregate_spearman_rho(gold_date2count, pred_date2count) - 0.8660254037844387) <= 1e-5

def task3_temporal_aggregates(label2pred, label2gold, docid_order):
    """
    Prints Spearman's rho between gold-standard and predicted document predictions aggregated by publication
    date of the article 
    """ 
    print('TASK 3 EVAL\n')
    docid2date = load_docid2date()
    gold_label2date2count, pred_label2date2count = aggregate_by_date(label2pred, label2gold, docid_order, docid2date)
    for label in LABEL_LIST: 
        print('CLASS=', label) 
        print('Spearmans rho=', metric_aggregate_spearman_rho(gold_label2date2count[label], pred_label2date2count[label]))
        print()

def load_gold_file(task_eval_choice): 
    if task_eval_choice == 1: #task 1: sentence classification 
        gold_file = 'data/final/sents.csv'

    elif task_eval_choice == 2: #task 2: document ranking 
        gold_file = 'data/final/docs.csv'

    elif task_eval_choice == 3: # task 3: substantive temporal aggregates (at the doc level)
        gold_file = 'data/final/docs.csv'
    return gold_file


def check_prediction_file_in_correct_format(pred_data, gold_data, task_eval_choice): 
    #check that they are the same number of lines 
    if len(pred_data) != len(gold_data):   
        raise Exception("number of lines in the prediction file does not equal number of lines in the gold file")
    
    if task_eval_choice == 1: #sentence level, unique identifier (docid, sentid)
        if not np.array_equal(pred_data['doc_id'], gold_data['doc_id']) or not np.array_equal(pred_data['sent_id'], gold_data['sent_id']):
           raise Exception('prediction_file (docid, sentid) order does not match gold_file')
    
    elif task_eval_choice in [2, 3]: #doc level, unique indentifier (docid)
        if not np.array_equal(pred_data['doc_id'], gold_data['doc_id']): 
            raise Exception('prediction_file (docid, sentid) order does not match gold_file')   

    print('confirmed prediction_file in correct format')

def format_predictions_gold(prediction_file, gold_file, task_eval_choice): 
    pred_df =  pd.read_csv(prediction_file)
    gold_df = pd.read_csv(gold_file)
    check_prediction_file_in_correct_format(pred_df, gold_df, task_eval_choice)

    label2pred = {}
    label2gold = {}
    for label in LABEL_LIST: 
        #check to make sure Task 1 and Task 3 have binary labels (0, 1) and Task 2 has scores (e.g real-values between 0 and 1)
        pred_array = pred_df[label].to_numpy()
        if task_eval_choice in [1, 3] and not np.issubdtype(pred_array.dtype, np.integer): 
            raise Exception ('predictions for Task 1 and 3 must be dtype=integer')
        elif task_eval_choice == 2 and not np.issubdtype(pred_array.dtype, np.floating): 
            raise Exception ('predictions for Task 2 must be dtype=floating, real-valued document-level relevance scores between 0.0 and 1.0')

        label2pred[label] = pred_df[label].to_numpy()
        label2gold[label] = gold_df[label].to_numpy() 

    docid_order = gold_df['doc_id'].to_numpy()

    return label2pred, label2gold, docid_order

def go_evaluate(label2pred, label2gold, docid_order, task_eval_choice): 
    if task_eval_choice == 1:
        task1_sentence_classification(label2pred, label2gold)

    elif task_eval_choice == 2: 
        task2_document_ranking(label2pred, label2gold)

    elif task_eval_choice == 3: 
        task3_temporal_aggregates(label2pred, label2gold, docid_order)


if __name__ == '__main__':
    """
    This script assumes the gold-standard files are in a data/final/ directory. 

    Inputs: 
        args.prediction_file

            The prediction file used for this evaluation script must be a .csv file (similar to the gold file) 

            Note: the ordering of docids, sentids is assumed to be the same in the prediction 
            and gold standard file (the script will throw an exception if it is not.)

            Task 1: The prediction_file should contain a sentence-level binary label (0 or 1) 
                for the ['KILL', 'ARREST', 'ANY_ACTION', 'FAIL', 'FORCE'] columns 

            Task 2: The prediction_file should contain a document-level relevance score between 0.0 and 1.0 
                for the ['KILL', 'ARREST', 'ANY_ACTION', 'FAIL', 'FORCE'] columns

            Task 3: The prediction_file should contain a document-level binary label (0 or 1)
                for the ['KILL', 'ARREST', 'ANY_ACTION', 'FAIL', 'FORCE'] columns


        args.task_eval_choice

            The task one wishes to evaluate the predictions on 

            1 = Task1--Sentence Classification 
            2 = Task2--Document Ranking
            3 = Task3--Substantive temporal aggregates

    Example usage: 
        python evaluate.py sent_predictions.csv 1

        python evaluate.py document_ranking_predictions.csv 2

        python evaluate.py document_classification_predictions.csv 3

        (where sent_predictions.csv, document_ranking_predictions.csv, document_classification_predictions.csv 
        are your own prediction files)
    
    """
    LABEL_LIST = ['KILL', 'ARREST', 'ANY_ACTION', 'FAIL', 'FORCE']

    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_file", help=".jsonl file", type=str)
    parser.add_argument("task_eval_choice", help="Choose from 1=task1, 2=task2, 3=task3", type=int) #optional argument that if flag is made, stores as true
    args = parser.parse_args()     
    task_eval_choice = args.task_eval_choice

    if task_eval_choice not in [1, 2, 3]: 
        raise Exception('task_eval_choice must be in {1,2,3} corresponding to task1, task2, or task3')

    gold_file = load_gold_file(task_eval_choice)

    label2pred, label2gold, docid_order = format_predictions_gold(args.prediction_file, gold_file, task_eval_choice)

    go_evaluate(label2pred, label2gold, docid_order, task_eval_choice) 




