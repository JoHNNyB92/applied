import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def generate_noise(min_max,keys,length,df,enable_plot=False):
    noise_array={}
    records={}
    for key in keys:
        mean=round((float)(min_max[key][0]-min_max[key][1])/2,1)
        std=round((float)(min_max[key][0]-min_max[key][1])/6,1)
        print length
        noise_array[key]=np.random.normal(mean, std, length)
        records[key]=np.random.uniform(low=0.0, high=1.0, size=length)
        if enable_plot==True:
            plt.hist(noise_array[key], bins='auto')  # arguments are passed to np.histogram\
            plt.title('NOISE FOR COLUMN:'+str(key)+" WITH MEAN:"+str(mean)+" STD:"+str(std))
            plt.show()
            plt.gcf().clear()
            #noise_array.append(np.random.normal(, , length))
    return (noise_array,records)
            
        
def add_noise(noisy,noisy_probs,dfs,feat_prob):
    cnt=0
    for key in dfs.keys():
        if len(dfs[keys])!=cnt:
            for rec in range(dfs[key].count()):
                prob=noisy_probs[key][rec]
                if prob<feat_prob[key]:
                    print 'Adding noise to '+key+',record '+str(rec)+' ,previously value '+str(dfs[key][rec])+', noise '+str(noisy[key][rec])+',new value '+str(dfs[key][rec]+noisy[key][rec])
                    dfs[key][rec]=dfs[key][rec]+noisy[key][rec]
        cnt=cnt+1
    return dfs

    
def write_excel_pd(pd_noise_array):
    pathname='noisy.xlsx'
    writer = pd.ExcelWriter(pathname)
    pd_noise_array.to_excel(writer,'Sheet1')
    writer.save()


if __name__ == "__main__":
    dfs = pd.read_excel("players_stats.xlsx")
    df = pd.DataFrame(dfs, columns=dfs.keys())
    noise_prop = [0.1 for _ in range(len(dfs.keys()))]
    min_max={}
    noise_prop_dict={}
    
    
    for key in df.keys():
        print 'Filling empty values with 0 for feature '+key
        df[key].fillna(0, inplace=True)
    length=df[key].count()
    cnt=0
    for key in df.keys():
        noise_prop_dict[key]=noise_prop[cnt]
        cnt+=1
        min_max[key]=[df.loc[df[key].idxmax()][key],df.loc[df[key].idxmin()][key]]
        print min_max[key]
    
    headers=df.keys()  
    noisy={}
    noisy_probs={}
    if len(sys.argv)!=1:
        (noisy,noisy_probs)=generate_noise(min_max,headers,length,dfs,True)
    else:
        (noisy,noisy_probs)=generate_noise(min_max,headers,length,dfs)
    dfs=add_noise(noisy,noisy_probs,df,noise_prop_dict)
    write_excel_pd(dfs)
	
 
