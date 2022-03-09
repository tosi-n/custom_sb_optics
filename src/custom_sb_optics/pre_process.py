#!/venv/bin/env python3
# -*- coding: utf-8 -*-
# Created By  : tosin_dairo
# Created Date: 28/01/2022
# version ='1.0

import os
import cv2
import json
import base64
import numpy as np
import pytesseract
import pandas as pd
import tqdm as tqdm
from PIL import Image
from nltk import ngrams
from gfpgan import GFPGANer
from io import StringIO, BytesIO
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from sklearn.model_selection import train_test_split

upsampler_model_path = '/home/ubuntu/tosi-n/custom_sb_optics/models/RealESRGAN_x4plus.pth'
ink_model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4
upsampler = RealESRGANer(scale=netscale, model_path=upsampler_model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=True)
ink_enhancer = GFPGANer(ink_model_path, upscale=3.5, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)

# Function to adjust the brightness through gamma parameter
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Image processing fuction return gray scale image
def image_processing_passport_front(img):
    """
    Image processing fuction return gray scale image
    """
    kernel = np.ones((3,3), np.uint8)
    dilated_img = cv2.dilate(img, kernel, iterations=1)
    bg_img = cv2.medianBlur(dilated_img, 21)
    bg_img = cv2.morphologyEx(bg_img, cv2.MORPH_OPEN, kernel)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = diff_img.copy() # Needed for 3.x compatibility
    norm_img = cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # _, thr_img = cv2.threshold(norm_img, 150, 150, cv2.THRESH_TRUNC) # for passport
    _, thr_img = cv2.threshold(norm_img, 300, 50, cv2.THRESH_TRUNC) # for NINs
    norm_img = cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    adjusted = adjust_gamma(norm_img, gamma=0.5)
    gray=cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    return gray

# Read base64 image and convert into ndarray processing in OpenCV
def readb64(bs64_):
    """
    Process base64 images and convert into ndarrays to save memory usage and data input
    """
    byte2str = bs64_.encode('ascii')
    buf = BytesIO(base64.b64decode(byte2str))
    # im = Image.open(buf)
    img = Image.open(buf)
    img = cv2.resize(np.asarray(img), (600, 400))
    img = image_processing_passport_front(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Transcribe ndarray image data and output string
def transcribeText(image):
    """
    Define config parameters.
    '-l eng'  for using the English language
    '--oem 1' for using LSTM OCR Engine
    """
    # image = cv2.imread(imagepath, cv2.IMREAD_COLOR)
    config = ('--oem 1 --psm 3')
    text = pytesseract.image_to_string(image, config=config)
    return text

# Resolution optimization
def up_resolution(img):
    """
    Upgrade Image Resolustion Using Real-ESRGAN.
    """
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None
    try:
        _, _, output = ink_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    return output

# Read multiple base64 images and labels for transcription
def transcribe_input(datapath):
    """
    Process csv of multiple base64 images and labels to transcribe into precessable contextual strings data annotaions and training
    """
    data = pd.read_csv(datapath)
    doc_id_list, full_bs64_img_list, label_n_bs64_img_dict_list = data['doc_id'].values, data['full_transcribed'].values, data['label'].values
    full_img_list = [readb64(bs_64) for bs_64 in full_bs64_img_list]
    full_transcript_list = [transcribeText(img) for img in full_img_list]
    label_n_img_dict_list = { k:readb64(v) for k,v in label_n_bs64_img_dict_list.items()}
    label_transcript_dict_list = {k:transcribeText(up_resolution(v)) for k,v in label_n_img_dict_list.items()}
    return doc_id_list, full_transcript_list, label_transcript_dict_list

# Read multiple base64 images and labels for transcription
def read_input(datapath):
    """
    Process csv of labels and transcript data for data annotaions and training
    """
    data = pd.read_csv(datapath)
    doc_id_list, full_transcript_list, label_transcript_dict_list = data['doc_id'].values, data['full_transcribed'].values, data['label'].values
    return doc_id_list, full_transcript_list, label_transcript_dict_list


def annotate(doc_id, full_transcript_list, label_dict):
    """
    Annotate full transcript and labelled dictionary using BILOU multi-token chunking techniques 
    """
    key = [k for k,v in label_dict.items()]
    # print(key)
    transcript_ = full_transcript_list[0].split()
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
    """
    Random data split for model training
    """
    return train_test_split(data, test_size=0.3, random_state=123, shuffle=True)
