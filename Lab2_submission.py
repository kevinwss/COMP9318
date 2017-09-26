## import modules here 
import pandas as pd
import numpy as np
import helper

################# Question 1 #################


##============================================================
# Data file format: 
# * tab-delimited input file
# * 1st line: dimension names and the last dimension is assumed to be the measure
# * rest of the lines: data values.
# helper functions

##### test use , not for submit
def read_data(filename):
    df = pd.read_csv(filename, sep='\t')
    return (df)
####

def buc_rec_optimized(df):
    ll=[]
    
        
    
    
    def buc_rec(input,l=[]):
        dims = input.shape[1]
    # Note that input is a DataFrame
        if input.shape[0]==1 and dims==3:
            a=input.iloc[0,0]
            b=input.iloc[0,1]
            s=input.iloc[0,2]
            ll.extend([l+[a,b,s],l+[a,"ALL",s],l+["ALL",b,s],l+["ALL","ALL",s]])
            return 
        
    
        if dims == 1:
        # only the measure dim
            input_sum = sum( helper.project_data(input, 0) )
            input_sum="%.1f" % input_sum
            ll.append(l+[input_sum])
        else:
        # the general case
            dim0_vals = set(helper.project_data(input, 0).values)

            for dim0_v in dim0_vals:
                sub_data = helper.slice_data_dim0(input, dim0_v)
                buc_rec(sub_data,l+[dim0_v])
        ## for R_{ALL}
            sub_data = helper.remove_first_dim(input)
            buc_rec(sub_data,l+['ALL'])
        
    buc_rec(df)  
    aa=np.array(ll)
    col=[]
    for i in df.columns:
        col.append(i)
        
    bb=pd.DataFrame(aa,columns=col)
    return(bb)
    

########test
input_data = read_data('d.txt')

output = buc_rec_optimized(input_data)

print(output)

################# Question 2 #################

def v_opt_dp(x, num_bins):
    def sse(l):
        n=len(l)
        if n==1:
            return 0 
        mean=sum(l)/n
        ans=sum(map(lambda x:(abs(x-mean))**2,l))
        return ans
        
    bins=[]
    n=len(x)
    matrix=[[-1 for _ in range(n)] for _ in range(num_bins)]   
    record=[[] for _ in range(n+1)]
    for j in range(n-1,-1,-1):
        if j>=num_bins-1:
            seg=x[j:]
            
            
            matrix[0][j]=sse(seg)
    for i in range(1,num_bins):
        last_record=[i for i in record]
        for j in range(n-1,-1,-1):
            if j>=num_bins-1-i and n-j>i:
                seg=x[j:]
                m=2147483647
                m_pos=-1
                for k in range(1,len(seg)):
                    new=sse(seg[:k])+matrix[i-1][j+k]#dp,find the minimum
                    if new<m:
                        m=new
                        m_pos=j+k       
                record[j]=last_record[m_pos]+[m_pos]
                matrix[i][j]=m        

        
    f_record=sorted(record[0])
    left=0
    for r in f_record:
        bins.append(x[left:r])# output the record
        left=r
    bins.append(x[left:])    
    return matrix,bins
    pass 
    
    
########test   
'''
x = [3, 1, 18, 11, 13, 17]
print(x)
num_bins = 4
matrix, bins = v_opt_dp(x, num_bins)
print(bins)
for row in matrix:
    print(row)    
'''
    
    
