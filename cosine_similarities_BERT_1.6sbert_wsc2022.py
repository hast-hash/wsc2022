# -*- coding: utf-8 -*-
#load necessary applications
#!pip install transformers
#!pip install sentence_transformers
import re
import numpy as np
import string
import unicodedata
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from transformers import BertModel, BertTokenizer

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


#variables to be used
file_directory='C:\\Docs\\The Borderers\\'
filename1 = 'Borderers 1797-99 Cornell.txt'
filename2 = 'Osorio 1797 Princeton.txt'
#number_of_lines = 10
#number_of_lines = 25
number_of_lines = 8
temp=[]
targetdataencoding="utf_8"
text1 = []
text2 = []
alltexts = []
nostr = ""
text1_label="The Borderers"
text2_label="Osorio"

#preprocessing
def text_preprocessing(text):
    pattern_change = [[' tis ',' it is '],
                      [' twas ', ' it was '],
                      [' twill ', ' it will '],
                      [' \'tis ',' it is '],
                      [' \'twas ', ' it was '],
                      [' \'twill ', ' it will '],
                      [' was ', ' is '],
                      [' were ', ' is '],
                      [' tho\' ', ' though '],
                      [' thro\' ', ' through '],
                      [' i\'th\'', ' in the'],
                      ['says\'t', 'say\'st'],
                      ['twere', 'it were'],
                      ['twon\'t', 'it will not'],
                      ['twould', 'it would'],
                      ['can\'t', 'can not'],
                      ['don\'t', 'do not'],
                      [' it\'s ', ' it is '],
                      [' that\'s ', ' that is '],
                      [' there\'s ', ' there is '],
                      [' there\'re ', ' there are '],
                      [' \'til ', ' till '],
                      [' i\'d ', ' I would '],
                      [' you\'d ', ' you would '],
                      [' shou\'d ', ' should '],
                      [' try\'d ', ' tried '],
                      [' o\'er', ' over'],
                      ['e\'er ', 'ever '],
                      [' ne\'er', ' never '],
                      [' heav\'n', ' heaven'],
                      ['\'ring ', 'ering '],
                      ['\'ning ', 'ening '],
                      ['\'ve ', ' have '],
                      [' murd\'rer', ' murderer'],
                      ['\'re ', ' are '],
                      [' \'he ', ' the '],
                      ['\'ee ', ' thee '],
                      [' \'mid ', ' amid '],
                      [' \'twixt ', ' betwixt '],
                      [' \'tendence ', ' atendence '],
                      ['\'t ', ' it '],
                      ['\'d ', 'ed ']]
    pattern_cut = ['ll', 'st', 'er', 'has', 'did', 'hath', 
                   'art', 'hast', 'dost', 've']
#   pattern_keep = his, this, thy    
    text = text.lower()
    #dashs are cut, i.e. double ephiset would be one single word 
    text = text.replace('-','')
#    text = re.sub(r'¥W',' ', text)
    text = re.sub(r'[\[\]\,\.\:\;\?\!\“\”\(\)]',' ', text)
    for i in pattern_change:
        text = re.sub(i[0], i[1], text)
#    text = text.replace('\n','')
    text = re.sub(r'¥s+',' ', text)
    text = re.sub(r'\'',' ', text)
    for i in pattern_cut:
        text = re.sub(" "+i+" ", ' ', text)
    return text

#cleaning Denis Rothman p.107
def clean_lines(lines):
    cleaned = list()
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        line = text_preprocessing(line)
        line = unicodedata.normalize('NFD', line).encode('ascii', 'ignore')
        line = line.decode('UTF-8')
        line = line.split()
        line = [word.lower() for word in line]
        line = [word.translate(table) for word in line]
        line = [re_print.sub('', w) for w in line]
        line = [word for word in line if word.isalpha()]
        cleaned.append(' '.join(line))
    return cleaned

#splitting a text into texts of specific number of lines
def divide_text_into_blocks(text, number_of_lines=number_of_lines):
    text_blocks = []
    k = text.split('\n')
    temp = []
    n = 0
    for i in k:
        temp.append(i)
        n += 1
        if n == number_of_lines:
            text_blocks.append(' '.join(temp))
            n = 0
            temp = []
    if n != 0:
        text_blocks.append(' '.join(temp))
    return text_blocks

#reading a text from a file  
def read_text(filename):
    with open(filename, "r", encoding=targetdataencoding) as f:
        text = f.read()
        text_blocks = divide_text_into_blocks(text, number_of_lines)
        text_blocks_cleaned = clean_lines(text_blocks)
    return text, text_blocks, text_blocks_cleaned

#sbert.net
#model = SentenceTransformer('bert-base-nli-mean-tokens')
model = SentenceTransformer('all-mpnet-base-v2')

#getting text1(a set of 50 lines of a poem) and text2
#Osorio
text1_all, text1_blocks, sentences1 = read_text(file_directory+filename1)
#Borderers
text2_all, text2_blocks, sentences2 = read_text(file_directory+filename2)

embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

sim = np.zeros((len(sentences1),len(sentences2)))
#for i in range(len(sentences1)):
#    for j in range(len(sentences2)):
#        sim[i,j] = cos_sim(embeddings1[i], embeddings2[j])
sim = cos_sim(embeddings1, embeddings2)

#heatmap
#plt.figure()
#plt.figure(figsize=(12, 8))
#sns.heatmap(sim, cmap='Blues', linewidths=0.5)

plt.figure()
#lines = 25
plt.figure(figsize=(48, 32))
sim_h=sns.heatmap(sim, cmap='Blues', vmin=0.5)
sim_h.set(xlabel="$"+text2_label+"$", ylabel="$"+text1_label+"$")
#line=10
#plt.figure(figsize=(60, 40))
#sns.heatmap(sim, cmap='Blues', vmin=0.65, vmax=0.7)

plt.savefig('image.png')
#plt.show()
plt.close('all')


#saving list 
def list_save(filename, list, columns=nostr, index=nostr):
    if columns == "" and index == "":
        df = pd.DataFrame(list)
    elif columns == "" and index != "":
        df = pd.DataFrame(list, index=index)
    else:
        df = pd.DataFrame(list, columns=columns, index=index)
    df.to_csv(filename, encoding=targetdataencoding)

list_ext = ".csv"
list_save(file_directory+"bert_text1_blocks"+list_ext, text1_blocks)
list_save(file_directory+"bert_text2_blocks"+list_ext, text2_blocks)
list_save(file_directory+"bert_sentences1"+list_ext, sentences1)
list_save(file_directory+"bert_sentences2"+list_ext, sentences2)
list_save(file_directory+"bert_sim"+list_ext, sim)

#Blocks starting Acts and Scenes
#for 50
#Table_Osorio = [[1,'1.1.'], [8, '2.1.'], [11, '2.2.'], [14, '3.1.'], [19, '3.2.'], [20, '4.1.'], \
#          [23, '4.2.'], [27, '4.3.'], [29, '5.1.'], [31, '5.2.']]
#Table_The_Borderers = [[1, '1.1.'], [5, '1.2.'], [7, '1.3.'], [10, '2.1.'], [13, '2.2.'], \
#             [14, '2.3.'], [23, '3.2.'], [25, '3.3.'], [28, '3.4.'], [29, '3.5.'], [33, '4.1.'], \
#             [34, '4.2.'], [37, '4.3.'], [40, '5.1.'], [41, '5.2.'], [42, '5.3.']]
#for 25
#Table_text1 = [[1, '1.1.'], [10, '1.2.'], [14, '1.3.'], [20, '2.1.'], [26, '2.2.'], \
#             [28, '2.3.'], [46, '3.2.'], [50, '3.3.'], [56, '3.4.'], [58, '3.5.'], [66, '4.1.'], \
#             [68, '4.2.'], [74, '4.3.'], [80, '5.1.'], [82, '5.2.'], [84, '5.3.']]
#Table_text2 = [[1,'1.1.'], [16, '2.1.'], [22, '2.2.'], [28, '3.1.'], [38, '3.2.'], [40, '4.1.'], \
#          [46, '4.2.'], [54, '4.3.'], [58, '5.1.'], [62, '5.2.']]
#for 8
Table_text1 = [[0, '1.1.'], [33, '1.2.'], [42, '1.3.'], [65, '2.1.'], [81, '2.2.'], \
             [88, '2.3.'], [142, '3.2.'], [157, '3.3.'], [176, '3.4.'], [182, '3.5.'], [205, '4.1.'], \
             [208, '4.2.'], [236, '4.3.'], [249, '5.1.'], [253, '5.2.'], [264, '5.3.']]
Table_text2 = [[0,'1.1.'], [46, '2.1.'], [67, '2.2.'], [87, '3.1.'], [121, '3.2.'], [126, '4.1.'], \
          [145, '4.2.'], [167, '4.3.'], [179, '5.1.'], [193, '5.2.']]

def act_check(list, value):
    act = ''
    for i in list:
        if value >= i[0]:
            act = i[1]
    return act
    
#Output the pairs with their score
def print_results(threshold):
    k = 0
    for i in range(len(sentences1)):
        for j in range(len(sentences2)):
            if sim[i][j] > threshold:
                a1 = act_check(Table_text1, i)
                a2 = act_check(Table_text2, j)
                print(" {}-{} Score: {:.4f} {}-{}\n {}: {} {}\n {} \n {}: {} {}\n {} \n".format(\
                        i, j, sim[i][j], a1, a2, text1_label, i, a1, sentences1[i], text2_label, j, a2, sentences2[j]))
                k = k + 1
    print("total: {}  {:.1f}%".format(k, k/(len(sentences1)*len(sentences2))*100))

threshold = 0.7
print_results(threshold)
