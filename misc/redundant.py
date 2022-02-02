# # %%
# import pandas as pd
# import numpy as np
# import tqdm as tqdm
# from nltk import ngrams
# from sklearn.model_selection import train_test_split

# # %%
# f_transcript_i = ['Election of Mayor for Greater Manchester Combined Authority, Date of election Thursday 6 May 2021 Your details: 157699 / QU3 Oluwatosin Dairo 707 Ambassador Apartments Waterman Walk Salford M503AW Number on register: QU3-115/1 You do not need to take this card with you in order to vote. Helpline number: 0161 793 2500 Email: elections@salford.gov.uk www.salford.gov.uk www.gmelects.org.uk']
# f_transcript_i = f_transcript_i[0].split()
# f_transcript_ii = ['Election of Mayor for Greater Manchester Combined Authority, Date of election Thursday 6 May 2021 Your details: 157700 / QU3 Terri Richardson 707 Ambassador Apartments Waterman Walk Salford M503AW Number on register: QU3-116 You do not need to take this card with you in order to vote. Helpline number: 0161 793 2500 Email: elections@salford.gov.uk www.salford.gov.uk www.gmelects.org.uk']
# f_transcript_ii = f_transcript_ii[0].split()

# data_label_i = {
#     'County' :'Greater Manchester Combined Authority',
#     'Election_day' : 'Thursday 6 May 2021',
#     'Name' : 'Oluwatosin Dairo',
#     'Address' : '707 Ambassador Apartments Waterman Walk Salford M503AW',
#     'Reg_number' : 'QU3-115/1',
#     }


# data_label_ii = {
#     'County' :'Greater Manchester Combined Authority',
#     'Election_day' : 'Thursday 6 May 2021',
#     'Name' : 'Terri Richardson',
#     'Address' : '707 Ambassador Apartments Waterman Walk Salford M503AW',
#     'Reg_number' : 'QU3-116',
#     }


# # %%
# sentence = data_label_i['Reg_number']

# n = len(sentence.split())
# n_gram = ngrams(sentence.split(), n)
# n_gram = list(*n_gram)
# print(n_gram)
# # [" ".join(grams) for grams in n_gram]
# # for grams in n_gram:
# #     print(grams)
# # %%
# f_transcript_i = [x for x in f_transcript_i if x not in n_gram]
# print(f_transcript_i)
# # %%
# sentence = data_label_ii['Reg_number']

# n = len(sentence.split())
# n_gram = ngrams(sentence.split(), n)
# n_gram = list(*n_gram)
# f_transcript_ii = [x for x in f_transcript_ii if x not in n_gram]
# print(f_transcript_ii)
# # %%
# f_transcript = [*f_transcript_i, *f_transcript_ii]
# df_1 = pd.DataFrame(f_transcript, columns = ['words'])
# df_1['labels'] = 'O'
# df_1.head()

# # %%
# data_label_i_ = {}
# for k,v in data_label_i.items():
#     if len(data_label_i[k].split()) < 2:
#         data_label_i_[k] = v
#         # print(data_label_i_[k])
# # # B-Address: adfg
#     elif len(data_label_i[k].split()) > 2:
#         sentece_list = data_label_i[k].split()
#         data_label_i_['B-' + k] = sentece_list[0]
#         data_label_i_['L-' + k] = sentece_list[-1]
#         data_label_i_['I-' + k] = sentece_list[1:-1]

#     else:
#         sentece_list = data_label_i[k].split()
#         data_label_i_['B-' + k] = sentece_list[0]
#         data_label_i_['L-' + k] = sentece_list[-1]

# print(data_label_i_)

#         # [v for if != v.split()[0]]
#         # 
# # %%
# data_label_ii_ = {}
# for k,v in data_label_ii.items():
#     if len(data_label_ii[k].split()) < 2:
#         data_label_ii_[k] = v
#         # print(data_label_i_[k])
# # # B-Address: adfg
#     elif len(data_label_ii[k].split()) > 2:
#         sentece_list = data_label_ii[k].split()
#         data_label_ii_['B-' + k] = sentece_list[0]
#         data_label_ii_['L-' + k] = sentece_list[-1]
#         data_label_ii_['I-' + k] = sentece_list[1:-1]

#     else:
#         sentece_list = data_label_ii[k].split()
#         data_label_ii_['B-' + k] = sentece_list[0]
#         data_label_ii_['L-' + k] = sentece_list[-1]

# print(data_label_ii_)

# # %%
# # sentence = data_label_i['Address']

# data_label = [data_label_i_, data_label_ii_]
# df_2_list = []

# for d in data_label:
#     df_2_list.append(pd.DataFrame(d.items(), columns = ['labels', 'words']))

# df_2 = pd.concat(df_2_list)
# df_2.head()

# # %%
# # s = df_2.apply(lambda x: pd.Series(x['words']), axis=1).stack().reset_index(level=1, drop=True)
# # s.name = 'words'
# # df_2 = df_2.drop('words', axis=1).join(s)
# # df_2['words'] = pd.Series(df_2['words'], dtype=object)
# # df_2.head()
# # %%
# # df_2.words.apply(pd.Series)
# df_2 = df_2.explode('words')
# df_2.head()
# # %%

# # print({**data_label_i, **data_label_ii})
# ds = [data_label_i, data_label_ii]
# d = {}
# for k in data_label_i.keys():
#   d[k] = tuple(d[k] for d in ds)

# print(d)
# # %%

# df = pd.concat([df_1, df_2]).sample(frac=1).reset_index(drop=True)
# print(len(df.index))
# print(df.tail(10))
# print(df.head(10))
# %%
# train_data, eval_data = train_test_split(df, test_size=0.3, random_state=123, shuffle=True)












###########################################################################
###########################################################################
###########################################################################

# data_label_i = {
#     'County' :'Greater Manchester Combined Authority',
#     'Election_day' : 'Thursday 6 May 2021',
#     'Name' : 'Oluwatosin Dairo',
#     'Address' : '707 Ambassador Apartments Waterman Walk Salford M503AW',
#     'Reg_number' : 'QU3-115/1',
#     }

# f_transcript_ = ['Election of Mayor for Greater Manchester Combined Authority, Date of election Thursday 6 May 2021 Your details: 157699 / QU3 Oluwatosin Dairo 707 Ambassador Apartments Waterman Walk Salford M503AW Number on register: QU3-115/1 You do not need to take this card with you in order to vote. Helpline number: 0161 793 2500 Email: elections@salford.gov.uk www.salford.gov.uk www.gmelects.org.uk, Election of Mayor for Greater Manchester Combined Authority, Date of election Thursday 6 May 2021 Your details: 157700 / QU3 Terri Richardson 707 Ambassador Apartments Waterman Walk Salford M503AW Number on register: QU3-116 You do not need to take this card with you in order to vote. Helpline number: 0161 793 2500 Email: elections@salford.gov.uk www.salford.gov.uk www.gmelects.org.uk']


# data = annotate(f_transcript_, data_label_i)

# train_data, eval_data = train_test_split(data, test_size=0.3, random_state=123, shuffle=True)
# print(train_data['labels'].value_counts())
# print(train_data.head())
# print(train_data.tail())