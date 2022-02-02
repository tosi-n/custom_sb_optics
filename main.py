import os
import logging
from src.custom_sb_optics.pre_process import annotate, data_split
from src.custom_sb_optics.train_runner import NERModel
from src.custom_sb_optics.config.global_args import global_args

if __name__ == "__main__":
    data_label = {
        'County' :'Greater Manchester Combined Authority',
        'Election_day' : 'Thursday 6 May 2021',
        'Name' : 'Oluwatosin Dairo',
        'Address' : '707 Ambassador Apartments Waterman Walk Salford M503AW',
        'Reg_number' : 'QU3-115/1',
        }

    f_transcript = ['Election of Mayor for Greater Manchester Combined Authority, Date of election Thursday 6 May 2021 Your details: 157699 / QU3 Oluwatosin Dairo 707 Ambassador Apartments Waterman Walk Salford M503AW Number on register: QU3-115/1 You do not need to take this card with you in order to vote. Helpline number: 0161 793 2500 Email: elections@salford.gov.uk www.salford.gov.uk www.gmelects.org.uk, Election of Mayor for Greater Manchester Combined Authority, Date of election Thursday 6 May 2021 Your details: 157700 / QU3 Terri Richardson 707 Ambassador Apartments Waterman Walk Salford M503AW Number on register: QU3-116 You do not need to take this card with you in order to vote. Helpline number: 0161 793 2500 Email: elections@salford.gov.uk www.salford.gov.uk www.gmelects.org.uk']

    
    data = annotate(f_transcript, data_label)

    train_data, eval_data = data_split(data)
    label = list(set(data.labels.tolist()))
    print(label)

    model = NERModel( "roberta", "roberta-base", labels=label, args=global_args, use_cuda=True)#, args=global_args

    # Train the model
    model.train_model(train_data, eval_data=eval_data)

    # Evaluate the model
    result, model_outputs, preds_list = model.eval_model(eval_data)
    print(result)