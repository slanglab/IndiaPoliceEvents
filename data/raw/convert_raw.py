import json 
import ipdb 

switch = {
    'KILL': 'ann_kill', 
    'ARREST': 'ann_arrest', 
    'FAIL': 'ann_fail', 
    'OTHER_RESPONSE': 'ann_other_response', 
    'NA': 'ann_na',
    'MULTI_SENT': 'ann_multi_sent',
    'HELP': 'ann_help',
}

fout = 'raw_annotations.jsonl'
ww = open(fout, 'w')

fin = 'raw_annotations--raw.jsonl'
for line in open(fin, 'r'): 
    dd = json.loads(line)
    dd_copy = dd.copy()
    # if len(dd['doc_labels']) > 0: 
    #     print(dd.keys())
    #     ipdb.set_trace()

    del dd_copy['sent_labels']

    #change sent labels 
    sent_labels = []
    for sent in dd['sent_labels']: 
        new_sent = []
        if len(sent) > 0: 
            for label in sent: 
                new_sent.append(switch[label])
        sent_labels.append(new_sent)
    dd_copy['sent_labels'] = sent_labels 
    assert len(dd_copy['sent_labels']) == len(dd_copy['sents'])

    #change raw annotations 
    del dd_copy["raw_annotations"] 
    raw_anns = []
    for sent in dd['raw_annotations']: 
        new_sent = []
        for code_item in sent:
            new_code_item = code_item.copy() 
            new_ann = []
            ann = code_item['annotation'] 
            if len(ann) > 0: 
                for x in ann: 
                    new_ann.append(switch[x])
            new_code_item['annotation'] = new_ann
            new_sent.append(new_code_item)
        assert len(new_sent) == len(sent)
        raw_anns.append(new_sent)
    dd_copy['raw_annotations'] = raw_anns 
    assert len(dd_copy['raw_annotations']) == len(dd_copy['sents'])
    assert len(dd_copy['raw_annotations'][0]) == len(dd['raw_annotations'][0])

    #change doc labels 
    del dd_copy['doc_labels']
    doc_labels = []
    if len(dd['doc_labels']) > 0: 
        for x in dd['doc_labels']: 
            doc_labels.append(switch[x])
    dd_copy['doc_labels'] = doc_labels
    assert len(dd_copy['doc_labels']) == len(dd['doc_labels'])

    # if dd['doc_id'] == 27: 
    #     ipdb.set_trace()

    json.dump(dd_copy, ww)
    ww.write('\n')

ww.close()
print('wrote to ->', fout)

