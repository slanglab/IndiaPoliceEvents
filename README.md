# IndiaPoliceEvents

Data and code to accompany the paper: Halterman, Keith, Sarwar, and O'Connor. "Corpus-Level Evaluation for Event QA:
The IndiaPoliceEvents Corpus Covering the 2002 Gujarat Violence." Findings of ACL, 2021. 

If you use this data or code, please cite the paper 

```
@inproceedings{halterman2021corpus,
author = {Halterman, Andrew and Keith, Katherine A. and Sarwar, Sheikh Muhammad, and O'Connor, Brendan}, 
title = {Corpus-Level Evaluation for Event {QA}:
The {IndiaPoliceEvents} Corpus Covering the 2002 {G}ujarat Violence},
booktitle = {{Findings of ACL}},
year = 2021}
```

## Directory structure 

- `evaluate.py` Evaluation script for the three tasks presented in the paper
- `explore_data.ipynb` Jupyter notebook that walks the user through using the data and using basline models (e.g. RoBERTa+MNLI) for inference on the data. 
- `data/`
    - `final/` Final data with adjudicated labels used for evaluation in the paper 
        - `sents.jsonl` (docid, sentid, text, labels)
        - `sents.csv`
        - `docs.jsonl` (docid, text, labels)
        - `docs.csv`
        - `metadata.jsonl` (docid, url, date)
        - `metadata.csv`
    - `retrieval/` In TREC format for information retrieval tasks
    - `raw/` Raw data with the complete annotation information 
    
- `archived_code/`
    - `models/` Baseline zero-shot models described in the paper 
        - `roberta_mnli.py`
        - `electra_msmarco.py`
        - `keyword_matching.py`
        - `bm25_variants.py`
    - `eval/` Evaluation scripts 
        - `master_eval.py`

## evaluate.py
Script with the evaluation metrics for Task 1-3 in the paper. 

Example usage: 
```
python evaluate.py sent_predictions.csv 1

python evaluate.py document_ranking_predictions.csv 2

python evaluate.py document_classification_predictions.csv 3
```
Where `sent_predictions.csv`, `document_ranking_predictions.csv`, and `document_classification_predictions.csv` are your own prediction files whose document and sentence order match the corresponding gold-standard .csv files: `data/final/*.csv`.

## explore_data.ipynb

We provide a Jupyter Notebook script `explore_data.ipynb` with examples for users on how to explore the data and use RoBERTa trained on MNLI data for inference. 

## IndiaPoliceEvents Corpus 

The final, adjudicated *IndiaPoliceEvents* corpus is located in the `data/final/` folder. The `.jsonl` and `.csv` files hold the same data in different formats for user convenience. Sentence-level labels can be found in the `sents.jsonl` and `sents.csv`, document-level labels can be found in the `docs.jsonl` and `docs.csv`, and metadata about the documents (document url and document dates) can be found in `metadata.jsonl` and `metadata.csv`. 

The sentence and document-level labels correspond to a positive answer to the boolean questions in the paper: 

- `"KILL"`: The text item is indicative of "Yes" to the question "Did police kill someone?"
- `"ARREST"`: The text item is indicative of "Yes" to the question "Did police arrest someone?"
- `"FAIL"`: The text item is indicative of "Yes" to the question "Did police fail to intervene"
- `"FORCE"`: The text item is indicative of "Yes" to the question "Did police use force or violence?"
- `"ANY_ACTION"`: The text item is indicative of "Yes" to the question "Did police do anything?"

## Raw Data 

The raw annotated data in `data/raw/raw_annotations.jsonl` consists of information about the original text along with our collected annotations. The data is presented as a newline-delimited JSON file, each row of which is a dictionary with information on one article from the news source.

#### Document information:

- `full_text`: The full text of the news story  
- `sents`: The sentence splits used in the annotation process. Note that many of the stories are all lower case meaning that the sentence boundary detection is often imperfect.  
- `doc_id`: A document ID for internal use   
- `date`: The publication date of the story  
- `url`: The URL of the story 

#### Raw annotations:

Each document has annotations for each sentence and for the document as a whole:

- `sent_labels`: An array with the length of `sents`, each of which contains the final, adjudicated labels for each sentence. Note that one sentence can have multiple labels.  
- `doc_labels`: The set of labels that were applied to at least one sentence in the document. 

The labels are as follows:

- `"ann_kill"`: "Did police kill someone?"
- `"ann_arrest"`: "Did police arrest someone?"
- `"ann_fail"`: "Did police fail to act?"
- `"ann_other_response"`: "Did police use other force or violence?"
- `"ann_na"`: "None of the above/police did something else."
- `"ann_multi_sent"`: Records whether annotators reported using information from elsewhere in the document to annotate the sentence.
- `"ann_help"`: During the adjudication round, adjudicators could flag difficult examples as unresolved, and the item was sent to a domain expert and an NLP expert to make the final judgment.  

Note that the final form of the data used in the paper collapses these categories:

- `"KILL"`: `{"ann_kill"}`
- `"ARREST"`: `{"ann_arrest"}`
- `"FAIL"`: `{"ann_fail"}`
- `"FORCE"`: `{"ann_kill", "ann_other_response"}`
- `"ANY_ACTION"`: `{"ann_kill", "ann_arrest", "ann_other_response", "ann_na"}`

The documents also have metadata about the annotation process and detailed information on each annotator's labels.

- `assigned_annotators`: Which annotators were assigned to the story? In most cases, this is two annotators. Annotators are identified with an ID number only to preserve their privacy.  
- `adjudication`: a boolean reporting whether the sentence was adjudicated. Sentences on which the original two annotators disagreed were referred to a third "adjudicator" to resolve the disagreement.
- `raw_annotations`: list of length of `sents`, each element of which is a list of dictionary items of each annotator and their annotation. For example, the annotations on one sentence could look like `[{'coder': 9019, 'annotation': ['ann_na']}, {'coder': 9020, 'annotation': []}, {'coder': '9017_adjudication', 'annotation': ['ann_na']}]`. If the sentence went to adjudication, the adjudicator's annotation is indicated with `{id_number)_adjudication`.  
- `annotation_explanations`: Annotators had the option to describe why they chose a particular label or to point out a difficult or borderline annotation. This is an array with the length of `sents`. 


## archived_code

In the `archived_code/` directory we provide code that was used to run the zero-shot baseline model experiments in the paper. We provide this code for documentation and replication purposes, but most of it is not directly runable. 



