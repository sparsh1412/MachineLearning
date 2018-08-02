from nltk.corpus import stopwords 
from sklearn.model_selection import train_test_split
import string
import time
import math

import numpy as np
import os
directory = r"C:\Users\Sparsh Jain\Downloads\20_newsgroups"
folders = [x[0] for x in os.walk(directory)]
folders = np.array(folders)
folders=folders[1:]

print(folders)

files = [x[2] for x in os.walk(directory)]    
files = np.array(files)
files = files[1:]
for i in range(len(files)):
    files[i] = np.array(files[i])

print(files)

dictionary = {}
for i in range(len(folders)):
    dictionary[folders[i]] = files[i]

print(dictionary)

all_list = []
target = []
for i in range(len(folders)):
    for j in range(len(files[i])):
        all_list.append(folders[i] + "\\" + dictionary[folders[i]][j])
        target.append(i)
x = np.array(all_list)
target = np.array(target)

print(x)

print(target)

x_train,x_test,y_train,y_test = train_test_split(x,target,test_size=0.2,random_state=1)

# x_train.shape , y_train.shape

#x_test.shape , y_test.shape

#np.unique(y_test,return_counts=True)

punc = string.punctuation
punc = punc + '0123456789'

def remove_punctuations_from_list(list):
    for i in range(len(list)):
        temp = ""
        for j in list[i]:
            if(j not in punc):
                temp += j
        list[i] = temp
    return list

stop_words = stopwords.words('english')
stop_words = remove_punctuations_from_list(stop_words)
stop_words.append("")

print(punc)

print(stop_words)

def remove_punctuations(filepath):
    word_list = open(filepath).read().lower().split()
    for i in range(len(word_list)):
        temp = ""
        for j in word_list[i]:
            if(j not in punc):
                temp+=j
        word_list[i] = temp
    return word_list

def remove_stop_words(np_list):    
    temp = []
    for i in np_list:
        if(i not in stop_words):
            temp.append(i)
    return temp

def clean(filepath):
    np_list = remove_punctuations(filepath)
    cleaned_list = remove_stop_words(np_list)
    return cleaned_list

def dictionary_file(filepath):
    cleaned_list = clean(filepath)
    unique_list = np.unique(cleaned_list, return_counts=True)
    dic = {}
    for i in range(len(unique_list[0])):
        dic[unique_list[0][i]] = unique_list[1][i]
    return dic

def create_path_dictionary():
    dic = {}
    for i in range(20):
        dic[i] = []
    for i in range(len(x_train)):
        dic[y_train[i]].append(x_train[i])
    return dic

def dictionary_folder(path_dict,target_class):
    folder_dic = {}
    for i in range(len(path_dict[target_class])):        
        temp_dic = dictionary_file(path_dict[target_class][i])
        temp_key_list = list(temp_dic.keys())        
        for key in temp_key_list:
            if(key not in folder_dic):
                folder_dic[key] = temp_dic[key]
            else:
                folder_dic[key] += temp_dic[key]
    return folder_dic

def create_mega_dictionary():
    path_dict = create_path_dictionary()
    mega_dict = {}
    for i in range(len(path_dict)):
        folder_dic = dictionary_folder(path_dict,i)
        folder_dic = list(folder_dic.items())
        folder_dic = sorted(folder_dic,reverse=True,key=lambda x: x[1])
        mega_dict[i] = dict(folder_dic)
        mega_dict[i]["TOTAL COUNT"] = np.array(list(mega_dict[i].values())).sum()
    return mega_dict

# st = time.time()
mega_dict = create_mega_dictionary()
# et = time.time()
# et-st

print(mega_dict)

def create_vocab(n_words,mega_dict):
    size = len(mega_dict)
    n = math.ceil(n_words/size)
    vocab = {}
    for i in range(size):
        key_list = list(mega_dict[i].keys())
        for j in range(n):
            if(key_list[j] not in vocab):
                vocab[key_list[j]] = mega_dict[i][key_list[j]]
            else:
                vocab[key_list[j]] += mega_dict[i][key_list[j]]
    return vocab

print(len(create_vocab(30000,mega_dict)))

vocab = create_vocab(30000,mega_dict)
print(vocab)

def calculate_class_probs():
    unique = np.unique(y_train,return_counts=True)
    target_class = unique[0]
    count = unique[1]
    probs = []
    for i in range(len(count)):
        temp = math.log10(count[i]/count.sum())
        probs.append(temp)
    return probs

def word_prob(word,target_class,vocab):
    if(word in mega_dict[target_class]):
        numerator = mega_dict[target_class][word] + 1
    else:
        numerator = 1
    denominator = mega_dict[target_class]["TOTAL COUNT"] + len(vocab)
    prob = numerator/denominator
    log_prob = math.log10(prob)
    return log_prob

# def word_prob(word, target_class, vocab, n_words=3000):
#     key_list = list(mega_dict[target_class].keys())
#     for i in range(n_words):
#         if(i==len(key_list)):
#             break
#         if(key_list[i] == word):
#             numerator = mega_dict[target_class][word] + 1
#             break
#         else:
#             numerator = 1    
#     denominator = mega_dict[target_class]["TOTAL COUNT"] + len(vocab)
#     prob = numerator/denominator
#     log_prob = math.log10(prob)
#     return log_prob

def step_predict(filepath, target_class, vocab):
    cleaned_list = clean(filepath)
    score = 0
    for i in cleaned_list:
        score+= word_prob(i,target_class,vocab) 
    return score

def predict(x_test,vocab):
    class_probs = calculate_class_probs()
    pred = []
    for i in x_test:
        final_probs = []
        for j in range(len(class_probs)):
            score = step_predict(i,j,vocab)
            score +=class_probs[j]
            final_probs.append(score)
            maxi = 0
        for j in range(len(class_probs)):
            if(final_probs[j]>final_probs[maxi]):
                maxi = j
        pred.append(maxi)
    return pred

# st = time.time()
y_pred = predict(x_test,vocab)
# et = time.time()
# et-st

print(len(y_pred), y_test.shape)

def score(y_pred , y_true):
    true_arr = y_pred == y_true
    score = true_arr.sum()/len(true_arr)*100
    return score

print(score(y_pred,y_test))
