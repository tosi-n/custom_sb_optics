#!/venv/bin/env python3
# -*- coding: utf-8 -*-
# Created By  : tosin_dairo
# Created Date: 28/01/2022
# version ='1.0

import os
import logging
import pandas as pd
from ast import literal_eval
from src.custom_sb_optics.pre_process import read_input, annotate, data_split #,transcribe_input
from src.custom_sb_optics.train_runner import NERModel

BUSINESS = 'swapbase'
CATEGORY = 'voter_card'
THRESHOLD = 0.75

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
    doc_id_list, transcript_list, label_dict_list, model_dir = read_input('/home/ubuntu/tosi-n/custom_sb_optics/data/demo_data_custom_optics.csv', BUSINESS, CATEGORY)
    
    # doc_id_list, transcript_list, label_dict_list = transcribe_input('/home/ubuntu/tosi-n/custom_sb_optics/data/demo_data_custom_optics.csv')

    for (a, b, c) in zip(doc_id_list, transcript_list, label_dict_list):
        df_list.append(annotate(a, b, literal_eval(c)))

    data = pd.concat(df_list).sample(frac=1).reset_index(drop=True)
    print(len(data.index))

    train_data, eval_data = data_split(data)
    label = list(set(data.labels.tolist()))
    print(label)

    best_model_dir_ = os.path.join(model_dir, 'best_model')

    model = NERModel( "roberta", "roberta-base", labels=label, args={'output_dir': model_dir, 'best_model_dir': best_model_dir_}, use_cuda=True)#, args=global_args

    # Train the model
    model.train_model(train_data, eval_data=eval_data)

    # Evaluate the model
    result, model_outputs, preds_list = model.eval_model(eval_data)

    # Model training decision...'
    if result['f1_score'] >= THRESHOLD:
        deploy_decision = 1
        artefact_dir = [os.path.join(model_dir, 'baseline', i) for i in os.listdir('/home/ubuntu/tosi-n/custom_sb_optics/models/baseline') if i !=  'checkpoint-0-epoch-6']
        for i in artefact_dir:
            cmd = 'rm -r ' + i #'find . -type d -name a -exec rmdir {} \;'
            os.system(cmd)
        print('Deploy decision indicates model training data was optimal at threshold level, so {}"s model can be deployed...'.format(CATEGORY))
        print('Model accuracy =>> {}'.format(str(result['accuracy']*100)))
    else:
        deploy_decision = 0
        artefact_dir = [os.path.join(model_dir, 'baseline', i) for i in os.listdir('/home/ubuntu/tosi-n/custom_sb_optics/models/baseline') if i not in  ['best_model', 'checkpoint-0-epoch-1', 'checkpoint-0-epoch-2', 'checkpoint-0-epoch-3', 'checkpoint-0-epoch-4', 'checkpoint-0-epoch-5', 'checkpoint-0-epoch-6', 'training_progress_scores.csv']]
        for i in artefact_dir:
            cmd = 'rm -r ' + i #'find . -type d -name a -exec rmdir {} \;'
            os.system(cmd)
        print('Deploy decision does not indicates model training data was optimal at threshold level, so {}"s model needs more data for incremental training...'.format(CATEGORY))
        print('Model accuracy =>> {}'.format(str(result['accuracy']*100)))
    
