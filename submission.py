## import modules here

import helper
import pandas as pd
import numpy as np
#from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn import tree
from sklearn.metrics import f1_score
import pickle


################# training #################


def obtain_feature(prons,record,word="",train_use=True,pos_tag=[]):
    vowels_list=["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
    consonants_list=["P", "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L","M", "N", "NG", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"]
    total_list=vowels_list+consonants_list #len 39
     
    
    #pos_tag=['NN', 'NNS', 'NNP', 'RB', 'VB', 'CC', 'IN', 'JJ', 'DT', 'FW', 'WP', 'WDT']
    #feature (total_vowel_number,vowel_pos,vowel,prev_pron,next_pron,2prev_pron,2next_pron)
    #rule1=["IC","EE","ER","NG","ES","TY","LE","RY"]
    features=[]
    labels=[]
    prons_length=len(prons)
    vowel_pos=0
    pos=0
    primary_vowel=""
    total_vowels=0
    vowels=[]
    
    for pron in prons:
        if pron not in consonants_list:
            total_vowels+=1
            vowels.append(pron)
    #if total_vowels==1:
     #   return features,labels,record
                
            
            
    for pron in prons:
        feature=[]
        pos+=1
        if pron not in consonants_list:
            if pos==1:
                prev="None"
            else:
                prev=prons[pos-2][:2]
            
            vowel_pos+=1
            
            if train_use==True:
                if int(pron[-1])==1:# In case for the test part
                    labels.append(1)
                else:
                    labels.append(0)
         
            #feature.append(total_vowels)
            
            
            #feature.append(vowel_pos)
            feature.append(vowels_list.index(pron[:2])+1)
            
            if prev=="None":
                feature.append(0)
            else:    
                feature.append(total_list.index(prev)+1)
                
            
                    
            feature.append(total_list.index(pron[:2])+1)
             
            if pos==prons_length:
                feature.append(0)
            else:
                feature.append(total_list.index(prons[pos][:2])+1)
            
            
            if pos==1 or pos==2:
                feature.append(0)
            else:
                feature.append(total_list.index(prons[pos-3][:2])+1)
                
            if pos==prons_length or pos==prons_length-1:
                feature.append(0)
            else:
                feature.append(total_list.index(prons[pos+1][:2])+1)
                
            if vowel_pos==1:
                feature.append(1)
            else:
                feature.append(0)
                
            if vowel_pos==total_vowels-1:
                feature.append(1)
            else:
                feature.append(0)  
                    
            if vowel_pos==total_vowels:
                feature.append(1)
            else:
                feature.append(0) 
                
            
            #feature.append(total_vowels-vowel_pos+1)
          #  if prev=="None":
          #      feature.append(total_list.index(pron[:2]))
          #  else:    
           #     feature.append(total_list.index(prev)*39+total_list.index(pron[:2]))   
            
               
            
            ############################# 
            features.append(feature)
            
            
        
    return features,labels,record     
    

def train(data, classifier_file):
    
    record=dict()
    features=[]
    labels=[]
    words=[]
    vowels=[]
    _class=[]
    pos_tag=[]
    for w in data:

        w=w.split(":")      #['K', 'OW1', 'EH2', 'D']
        word=w[0]
        prons=w[1].split(" ")
        
        
        
        new_features,new_labels,record=obtain_feature(prons,record,word,True)
       # if new_features!=[]:
        features+=new_features
        labels+=new_labels 
        
    #print(pos_tag)    
    #print(features,labels) 
    #print(features,labels)
    #X = np.array([[1,2,3],[2,3,4],[3,4,5]])
    #Y = np.array([1, 2, 3])
    #clf = tree.DecisionTreeClassifier()    
    clf = tree.DecisionTreeClassifier(min_samples_leaf=8,min_samples_split=8)    
    clf.fit(features, labels)
    #print(clf.feature_importances_)
    
    
    output = open(classifier_file, 'wb')
    pickle.dump(clf,output)
    output.close()
    
    
    
################# testing #################

def test(data, classifier_file):
    vowels_list=["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
    consonants_list=["P", "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L","M", "N", "NG", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"]
    total_list=vowels_list+consonants_list
    #rule1=["IC","EE","ER","EY","NG","ES","CK","IS","TY","LE","RS","GH","TI","RY"]
    #pos_tag=['NN', 'NNS', 'NNP', 'RB', 'VB', 'CC', 'IN', 'JJ', 'DT', 'FW', 'WP', 'WDT']
    features=[]
    pkl_file = open(classifier_file, 'rb')
    clf=pickle.load(pkl_file)
    res=[]
    features=[]
    labels=[]
    predict=[]
    
    
    for w in data:
        #print(w)
        
        w=w.split(":")
        
        word=w[0]
        l_word=len(word)
        prons=w[1].split(" ") #[K OW1 EH2 D]
        
        vowel_pos=0
        predict_max=0
        predict_res=1
        total_vowels=0
        vowels=[]
        #count :
        prons_length=len(prons)
        for pron in prons:
            if pron not in consonants_list:
                total_vowels+=1
                vowels.append(pron)
                
        
                
        pos=0        
        for pron in prons:
            pos+=1
            
            if pron not in consonants_list:
                if pos==1:
                    prev="None"
                else:
                    prev=prons[pos-2][:2]
                    
                feature=[]
                vowel_pos+=1
                #feature.append(total_vowels)
            
               #############
          #      if int(pron[-1])==1:
          #          labels.append(vowel_pos)
                ###################
            
                #feature.append(vowel_pos)
                feature.append(vowels_list.index(pron[:2])+1)
                
                if prev=="None":
                    feature.append(0)
                else:
                    feature.append(total_list.index(prev)+1)
                   
                feature.append(total_list.index(pron[:2])+1)
                
                #feature.append(total_list.index(pron[:2]))
                #print(feature)
                if pos==prons_length:
                    feature.append(0)
                else:
                    feature.append(total_list.index(prons[pos][:2])+1)
                
                if pos==1 or pos==2:
                    feature.append(0)
                else:
                    feature.append(total_list.index(prons[pos-3][:2])+1)
                    
                if pos==prons_length or pos==prons_length-1:
                    feature.append(0)
                else:
                    feature.append(total_list.index(prons[pos+1][:2])+1) 
                
                
                if vowel_pos==1:
                    feature.append(1)
                else:
                    feature.append(0)
                    
                    
                if vowel_pos==total_vowels-1:
                    feature.append(1)
                else:
                    feature.append(0)
                    
                if vowel_pos==total_vowels:
                    feature.append(1)
                else:
                    feature.append(0)   
                                                                   
                predict=clf.predict_proba([feature])

                predict=predict[0][1]
                if predict> predict_max:
          
                     predict_max=predict
                     predict_res=vowel_pos                   
                     
        res.append(predict_res)
    
        
#    ground_truth =labels  
#   length=len(res)
    
    
   # file_object = open('wrong.txt', 'w')
    #file_object = open('true.txt', 'w')
    
  #  for i in range(length):
    #    if res[i]!=ground_truth[i]:
            #print(i,ground_truth[i],res[i],data[i])
       #     file_object.write(str(i)+" "+str(ground_truth[i])+" "+str(res[i])+" "+data[i]+"\n")
      #  else:
         #   file_object.write(str(i)+" "+str(ground_truth[i])+" "+str(res[i])+" "+data[i]+str(predict_max)+"\n")
        
#    f1=f1_score(ground_truth, res, average='macro')
#    print(f1)
    
     
    
       
    pkl_file.close()
    return res
   # return res,f1
    
#['COED:K OW1 EH2 D', 'PURVIEW:P ER1 V Y UW2']
    
'''
training_data = helper.read_data('./asset/training_data.txt')
classifier_path = './asset/classifier.dat'
train(training_data, classifier_path)
testing_data = helper.read_data('./asset/tiny_test.txt')
prediction=test(testing_data, classifier_path)
print(prediction)

'''

###cross validation
'''
if_test=1
if if_test==1:
    training_data = helper.read_data('./asset/training_data.txt')


    classifier_path = './asset/classifier.dat'
    length=len(training_data)
    n=5
    print("{:}-folds cross validation".format(n))
    single=length//n
    f=[]
    for i in range(n):
        
        testing_set=training_data[i*single:(i+1)*single]
        training_set=training_data[:i*single]+training_data[(i+1)*single:]
        train(training_set, classifier_path)
    
        prediction,f1= test(testing_set, classifier_path)
        f.append(f1)
        
    print("f1={:}".format(sum(f)/n))
'''