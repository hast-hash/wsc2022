# -*- coding: utf-8 -*-
#load necessary applications
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd


#variables to be used
file_directory='C:\\Docs\\The Borderers\\'
filename1 = 'Borderers 1797-99 Cornell.txt'
filename2 = 'Osorio 1797 Princeton.txt'
number_of_lines = 50
#number_of_lines = 20
temp=[]
targetdataencoding="utf_8"
text1 = []
text2 = []
alltexts = []
porter = PorterStemmer()
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

def text_preprocessing_stemming(text):
    tokenized_text = word_tokenize(text)
    porter_text = [porter.stem(word) for word in tokenized_text]
    text = ' '.join(porter_text)
    return text

#splitting a text into texts of specific number of lines
def divide_text_into_blocks(text, number_of_lines=number_of_lines):
    doc = []
    k = text.split('\n')
    temp = []
    n = 0
    for i in k:
        temp.append(i)
        n += 1
        if n == number_of_lines:
            doc.append(' '.join(temp))
            n = 0
            temp = []
    if n != 0:
        doc.append(' '.join(temp))
    return doc

#reading a text from a file  
def read_text(filename):
    with open(filename, "r", encoding=targetdataencoding) as f:
        text = f.read()
        text_orig = divide_text_into_blocks(text, number_of_lines)
        text_preprocessed = []
        for i in text_orig:
            i = text_preprocessing(i)
            i = text_preprocessing_stemming(i)
            text_preprocessed.append(i)
    return text, text_orig, text_preprocessed

def get_tfidf(alltexts=alltexts):
    nptexts = np.array(alltexts)
#    vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
#    vectorizer = TfidfVectorizer(smooth_idf=False)
#    vectorizer = TfidfVectorizer(smooth_idf=False, ngram_range=(1,2))
    vectorizer = TfidfVectorizer(smooth_idf=False, stop_words='english')
#    vectorizer = TfidfVectorizer(smooth_idf=False, stop_words='english',
#                                 ngram_range=(1,2))
    vecs = vectorizer.fit_transform(nptexts)
    #tfidf
    tfidfs = vecs.toarray()
    terms = vectorizer.get_feature_names()
    return tfidfs, terms

def print_tfidf(alltexts_tfidfs):
    for n, tfidf in enumerate(alltexts_tfidfs):
        print("No.", n, "\t", end='')
        for i in tfidf:
            print("%8.4f" % i, end='')
        print()

    
    
#getting text1(a set of 50 lines of a poem) and text2 and concatenate them
#Osorio
text1_all, text1_orig, text1 = read_text(file_directory+filename1)
#Borderers
text2_all, text2_orig, text2 = read_text(file_directory+filename2)
alltexts = text1 + text2
blocks1 = len(text1)
blocks2 = len(text2)

#getting tfidfs
alltexts_tfidfs, alltexts_terms = get_tfidf(alltexts)

#getting cosine similarity
#
#This is the same as below.
#c1, c2 = np.vsplit(alltexts_tfidfs, [blocks1])
#similarity2 = cosine_similarity(c1, c2)
#
similarity = cosine_similarity(alltexts_tfidfs)

#getting only necessary values i.e. b2
a1, a2 = np.vsplit(similarity, [blocks1])
b1, b2 = np.hsplit(a1, [blocks1])
similarity_alltexts = b2

#heatmap
plt.figure()
plt.figure(figsize=(12, 8))
#plt.figure(figsize=(24, 16))
#number_of_lines = 50
#sns.heatmap(similarity_alltexts, cmap='Blues', vmin=0.35, vmax=0.4, linewidths=0.5)
#50, ngram=(1,1)
sim_h = sns.heatmap(similarity_alltexts, cmap='Blues', linewidths=0.5)
#sim_h = sns.heatmap(similarity_alltexts, cmap='Blues', vmin=0.1, vmax=0.225, linewidths=0.5)
sim_h.set(xlabel="$"+text2_label+"$", ylabel="$"+text1_label+"$")
#50, ngram=(1,2)
#sns.heatmap(similarity_alltexts, cmap='Blues', vmin=0.05, vmax=0.08, linewidths=0.5)
#number_of_lines = 20
#sns.heatmap(similarity_alltexts, cmap='Blues', vmin=0.25, vmax=0.3, linewidths=0.5)
#sns.heatmap(similarity_alltexts, cmap='hsv_r')
#plt.close('all')

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
list_save(file_directory+"list_text1"+list_ext, text1)
list_save(file_directory+"list_text2"+list_ext, text2)
list_save(file_directory+"list_alltexts_tfidfs"+list_ext, alltexts_tfidfs)
list_save(file_directory+"list_alltexts_terms"+list_ext, alltexts_terms)
list_save(file_directory+"list_similarity"+list_ext, similarity)
list_save(file_directory+"list_b2"+list_ext, b2)

c1, c2 = np.vsplit(alltexts_tfidfs, [blocks1])
similarity2 = cosine_similarity(c1, c2)
list_save(file_directory+"list_similarity2"+list_ext, similarity2)

def index_range(blocks, string):
    index = []
    i = range(blocks)
    for j in i:
        k = [string + str(j)]
        index = index + k
    return index
index = index_range(blocks1,"c") + index_range(blocks2,"w")
list_save(file_directory+"list_alltexts_tfidfs2"+list_ext, 
          alltexts_tfidfs, alltexts_terms, index)

#show same words
#scanlist = [[27,18], [34,43], [32,46], [16,43], [9,18]]
scanlist = [[18,27], [43,34], [46,32], [43,16], [18,9]]
scanterms = []
scanlist_index = []
def scanterm(scanlist=scanlist, alltexts_terms=alltexts_terms, alltexts_tfidfs=alltexts_tfidfs, blocks1=blocks1, scanterms = scanterms, scanlist_index = scanlist_index):
    for i in scanlist:
        k = 0
        temp = []
        list1 = alltexts_tfidfs[i[0]]
        list2 = alltexts_tfidfs[blocks1+i[1]]
        for j in alltexts_terms:
            check1 = list1[k]
            check2 = list2[k]
            if check1 * check2 > 0:
                temp = temp + [j]
            k += 1
        scanterms.append(temp)
        scanlist_index = scanlist_index + [str(i[0])+"-"+str(i[1])]
    return scanterms, scanlist_index

scanterms, scanlist_index = scanterm(scanlist, alltexts_terms, alltexts_tfidfs, blocks1)
list_save(file_directory+"list_scanterms"+list_ext, 
          scanterms, nostr, scanlist_index)

def elicit_print(text, keywordlist):
    color = ['\033[31m', '\033[32m', '\033[36m', 
             '\033[41m', '\033[42m', '\033[46m']
    j = len(color)
    for i in keywordlist:
        if j == len(color):
            j = 0
        text = re.sub('^'+i+'| '+i+'|^\''+i+'| \''+i, " "+color[j]+i+'\033[0m', text)
        j += 1
    print(text)
    
    
#for 50
Table_text1 = [[0, '1.1.'], [5, '1.2.'], [6, '1.3.'], [10, '2.1.'], [13, '2.2.'], \
             [14, '2.3.'], [22, '3.2.'], [25, '3.3.'], [28, '3.4.'], [29, '3.5.'], [32, '4.1.'], \
             [33, '4.2.'], [37, '4.3.'], [39, '5.1.'], [40, '5.2.'], [42, '5.3.']]
Table_text2 = [[0,'1.1.'], [7, '2.1.'], [10, '2.2.'], [14, '3.1.'], [19, '3.2.'], [20, '4.1.'], \
          [23, '4.2.'], [26, '4.3.'], [28, '5.1.'], [30, '5.2.']]
threshold = 0
sentences1 = []
sentences2 = []
sim = []
doc = [] 
    
def act_check(list, value):
    act = ''
    for i in list:
        if value >= i[0]:
            act = i[1]
    return act
  
#Output the pairs with their score
def print_results(threshold=threshold, sentences1 = sentences1, sentences2 = sentences2, sim=sim):
    print(len(sentences2))
    print(len(sentences1))
    k = 0
    for i in range(len(sentences1)):
        for j in range(len(sentences2)):
            if sim[i][j] > threshold:
                a1 = act_check(Table_text1, i)
                a2 = act_check(Table_text2, j)
                print(" {}-{} Score: {:.4f} {}-{}\n {}: {} {}\n {} \n {}: {} {}\n {} \n".format(\
                        i, j, sim[i][j], a1, a2, text1_label, i, a1, sentences1[i], text2_label, j, a2, sentences2[j]))
                k = k + 1
                doc.append([i, sentences1[i], j, sentences2[j]])
    print("total: {}  {:.1f}%".format(k, k/(len(sentences1)*len(sentences2))*100))
    return doc

def output(text1_orig=text1_orig, text2_orig=text2_orig, similarity_alltexts=similarity_alltexts): 
    threshold = 0.2
    sentences1 = text1_orig
    sentences2 = text2_orig
    sim = similarity_alltexts
    doc = print_results(threshold, sentences1, sentences2, sim)
    
    scanlist=[]
    for k in doc:
        i = k[0]
        j = k[2]
        scanlist.append([i,j])
    scanterms, scanlist_index = scanterm(scanlist)    
    print("\nCommon terms:")
    for i in range(len(scanlist_index)):
        print(scanlist_index[i])
        print(scanterms[i])
        
    return

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
def get_doc2vec(alltexts=alltexts):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(alltexts)]
    model = Doc2Vec(documents, vector_size=2, window=5, min_count=1, workers=4)
    return model

def get_doc2vecheatmap(alltexts=alltexts, text1=text1,text2=text2):
    model = get_doc2vec(alltexts)
    sim_doc2vec = []
    for j in text2:
        doc_words2 = j
        doc2vec_hline = []
        for i in text1:
            doc_words1 = i
            sim_value = model.similarity_unseen_docs(model, doc_words1, doc_words2)
            doc2vec_hline = doc2vec_hline + sim_value
        sim_doc2vec.append(doc2vec_hline)
    return sim_doc2vec            

