#!/venv/bin/env python3
# -*- coding: utf-8 -*-
# Created By  : tosin_dairo
# Created Date: 28/01/2022
# version ='1.0

import pandas as pd
import numpy as np
import tqdm as tqdm
from nltk import ngrams
from sklearn.model_selection import train_test_split

def read_input(datapath):
    data = pd.read_csv(datapath)
    doc_id_list, transcript_list, label_dict_list = data['doc_id'].values, data['full_transcribed'].values, data['label'].values
    return doc_id_list, transcript_list, label_dict_list

def annotate(doc_id, transcript_list, label_dict):
    key = [k for k,v in label_dict.items()]
    # print(key)
    transcript_ = transcript_list[0].split()
    exc_f_transcript = []
    for i in key:
        sentence = label_dict[i]
        n = len(sentence.split())
        n_gram = ngrams(sentence.split(), n)
        n_gram = list(*n_gram)
        for x in transcript_: 
            if x in n_gram:
                exc_f_transcript.append(x)
    # print(exc_f_transcript)
    f_transcript = [x for x in transcript_ if x not in exc_f_transcript]
    # print(f_transcript)
    df_1 = pd.DataFrame(f_transcript, columns = ['words'])
    df_1['labels'] = 'O'

    label_dict_ = {}
    for k,v in label_dict.items():
        if len(label_dict[k].split()) < 2:
            label_dict_[k] = v
        elif len(label_dict[k].split()) > 2:
            sentece_list = label_dict[k].split()
            label_dict_['B-' + k] = sentece_list[0]
            label_dict_['L-' + k] = sentece_list[-1]
            label_dict_['I-' + k] = sentece_list[1:-1]
        else:
            sentece_list = label_dict[k].split()
            label_dict_['B-' + k] = sentece_list[0]
            label_dict_['L-' + k] = sentece_list[-1]

    df_2 = pd.DataFrame(label_dict_.items(), columns = ['labels', 'words'])
    df_2 = df_2.explode('words')

    df = pd.concat([df_1, df_2]).sample(frac=1).reset_index(drop=True)
    df['sentence_id'] = doc_id
    df = df[['sentence_id', 'words', 'labels']]
    print(len(df.index))
    # print(df.head())
    return df


def data_split(data):
    return train_test_split(data, test_size=0.3, random_state=123, shuffle=True)
