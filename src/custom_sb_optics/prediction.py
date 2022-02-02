#!/venv/bin/env python3
# -*- coding: utf-8 -*-
# Created By  : tosin_dairo
# Created Date: 28/01/2022
# version ='1.0

from train_runner import NERModel
from scipy.special import softmax
import numpy as np

base_model = './models/baseline'

model = NERModel( "roberta", base_model, args={'reprocess_input_data': True}, use_cuda=True)

# Model fn_type inference and confidence score
def predict(doc_content):
	doc_content = [doc_content]
	preds_l, outputs = model.predict(doc_content)
	
	# confidence = []
	# probabilities = np.array([softmax(element) for element in outputs])
	# for i in probabilities:
	# 	long_probs = np.amax(i)
	# 	confidence.append(long_probs)
	
	# print(preds_l)
	# print(outputs)
	return preds_l#, outputs
	# for k, v in label_dict.items():
	# 	if preds_l[0] == k:
	# 		i = v
	# 		# print(v)
	# 			# return i

	# return i, str(confidence[0])


sample = predict('Election of Mayor for Greater Manchester Combined Authority, Date of election Thursday 6 May 2021 Your details: 157699 / QU3 Oluwatosin Dairo 707 Ambassador Apartments Waterman Walk Salford M503AW Number on register: QU3-115/1 You do not need to take this card with you in order to vote. Helpline number: 0161 793 2500 Email: elections@salford.gov.uk www.salford.gov.uk www.gmelects.org.uk, Election of Mayor for Greater Manchester Combined Authority, Date of election Thursday 6 May 2021 Your details: 157700 / QU3 Terri Richardson 707 Ambassador Apartments Waterman Walk Salford M503AW Number on register: QU3-116 You do not need to take this card with you in order to vote. Helpline number: 0161 793 2500 Email: elections@salford.gov.uk www.salford.gov.uk www.gmelects.org.uk')

print(sample)
