## import modules here 
import pandas as pd

################# Question 1 #################


def tokenize(sms):
    return sms.split(' ')

def get_freq_of_tokens(sms):
    tokens = {}
    for token in tokenize(sms):
        if token not in tokens:
            tokens[token] = 1
        else:
            tokens[token] += 1
    return tokens


def multinomial_nb(docs,sms):
    pxspam = []
    pxham = []
    cspam=0
    cham =0
    total_docs=len(docs)
    nspam=0
    nham=0
    name_list=set()
    for d in docs:
        if d[1]=="ham":
            cham+=1
            for i in d[0]:
                nham+=d[0][i]
                name_list.add(i)
        elif d[1]=="spam":
            cspam+=1
            for i in d[0]:
                nspam+=d[0][i]
                name_list.add(i)
    v=len(name_list)       
    pham=cham/total_docs        
    pspam=cspam/total_docs  
    for s in sms:
        nks=0
        nkh=0
        for d in docs:
            if d[1]=="ham":
                if s in d[0]:
                    nkh+=d[0][s]
            elif d[1]=="spam":
                if s in d[0]:
                    nks+=d[0][s]    
        if nks==0 and nkh==0:
            continue             
        p_h=((nkh+1)/(nham+v))           
        pxham.append(p_h) 
        p_s=((nks+1)/(nspam+v)) 
        pxspam.append(p_s)  
        
    pxspam.append(pspam) 
    pxham.append(pham)  
    p_ham=1
    p_spam=1
    for i in pxspam:
        p_spam*=i
    for j in pxham:
        p_ham*=j    
    return p_spam/p_ham

'''
raw_data = pd.read_csv('./asset/data.txt', sep='\t')
#print(raw_data.head())
    
training_data = []

for index in range(len(raw_data)):
    training_data.append((get_freq_of_tokens(raw_data.iloc[index].text), raw_data.iloc[index].category))
    
    
sms = 'I am not spam'

print(multinomial_nb(training_data, tokenize(sms)))
'''
