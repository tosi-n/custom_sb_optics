#!/venv/bin/env python3
# -*- coding: utf-8 -*-
# Created By  : tosin_dairo
# Created Date: 28/01/2022
# version ='1.0

import os
import logging
import pandas as pd
from ast import literal_eval
from src.custom_sb_optics.pre_process import transcribe_input, annotate, data_split
from src.custom_sb_optics.train_runner import NERModel
from src.custom_sb_optics.config.global_args import global_args


if __name__ == "__main__":
    # doc_id = 0
    # data_label = {
    #     'County' :'Greater Manchester Combined Authority',
    #     'Election_day' : 'Thursday 6 May 2021',
    #     'Name' : 'Oluwatosin Dairo',
    #     'Address' : '707 Ambassador Apartments Waterman Walk Salford M503AW',
    #     'Reg_number' : 'QU3-115/1',
    #     }

    # f_transcript = ['Election of Mayor for Greater Manchester Combined Authority, Date of election Thursday 6 May 2021 Your details: 157699 / QU3 Oluwatosin Dairo 707 Ambassador Apartments Waterman Walk Salford M503AW Number on register: QU3-115/1 You do not need to take this card with you in order to vote. Helpline number: 0161 793 2500 Email: elections@salford.gov.uk www.salford.gov.uk www.gmelects.org.uk, Election of Mayor for Greater Manchester Combined Authority, Date of election Thursday 6 May 2021 Your details: 157700 / QU3 Terri Richardson 707 Ambassador Apartments Waterman Walk Salford M503AW Number on register: QU3-116 You do not need to take this card with you in order to vote. Helpline number: 0161 793 2500 Email: elections@salford.gov.uk www.salford.gov.uk www.gmelects.org.uk']



    df_list = []

    doc_id_list, transcript_list, label_dict_list = transcribe_input('/home/ubuntu/tosi-n/custom_sb_optics/data/demo_data_custom_optics.csv')

    for (a, b, c) in zip(doc_id_list, transcript_list, label_dict_list):
        df_list.append(annotate(a, b, literal_eval(c)))

    data = pd.concat(df_list).sample(frac=1).reset_index(drop=True)
    print(len(data.index))

    train_data, eval_data = data_split(data)
    label = list(set(data.labels.tolist()))
    print(label)

    model = NERModel( "roberta", "roberta-base", labels=label, args=global_args, use_cuda=True)#, args=global_args

    # Train the model
    model.train_model(train_data, eval_data=eval_data)

    # Evaluate the model
    result, model_outputs, preds_list = model.eval_model(eval_data)
    print(result)