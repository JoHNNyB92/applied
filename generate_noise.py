import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def generate_noise(min_max,length,keys,df,enable_plot=False):
    noise_array={}
    records={}
    for key in keys:
        mean=round((float)(min_max[key][0]-min_max[key][1])/2,1)
        std=round((float)(min_max[key][0]-min_max[key][1])/6,1)
        noise_array[key]=np.random.normal(mean, std, length)
        records[key]=np.random.uniform(low=0.0, high=1.0, size=length)
        if enable_plot==True:
            plt.hist(noise_array[key], bins='auto')  # arguments are passed to np.histogram\
            plt.title('NOISE FOR COLUMN:'+str(key)+" WITH MEAN:"+str(mean)+" STD:"+str(std))
            plt.show()
            plt.gcf().clear()
            #noise_array.append(np.random.normal(, , length))
            
        
#def add_noise():
#    #TODO



if __name__ == "__main__":
    dfs = pd.read_excel("players.xlsx")
    length=df['FTA'].count()
    noise_prop = [0.1 for _ in range(length)]
    noise_prop_dic={}
    print dfs.keys()
    df = pd.DataFrame(dfs, columns=dfs.keys())
    min_max={}
    df=df.drop('Player', axis=1)
    cnt=0
    for key in df.keys():
        noise_prop_dict[key]=noise_prop[cnt]
        cnt+=1
        min_max[key]=[df.loc[df[key].idxmax()][key],df.loc[df[key].idxmin()][key]]
        print min_max[key]
    
    headers=df.keys()
    if len(sys.argv)!=1:
        generate_noise(noise_prop_dict,min_max,length,headers,dfs,True)
    else:
        generate_noise(noise_prop_dict,min_max,length,headers,dfs)
    
    '''miss = prop
    outlier = prop
    swap = prop
    mixed = []
    ms=0
    sw=0
    o=0
    for i, v in enumerate(gen):
        cols = len(v)
        uni = np.random.uniform(0, 1, 3)
        
        if uni[0] < miss:
            ms=ms+1
            v[int(np.random.uniform(0, cols, 1)[0])] = 'NA'
            #print('NA in record ', i)
 
        if uni[1] < swap:
            sw=sw+1
            swap_p = np.random.uniform(0, cols, 2)
            first = int(swap_p[0])
            second = int(swap_p[1])
            temp = v[first]
            v[first] = v[second]
            v[second] = temp
            #print('Swapping record ', i, ' columns ', first, '-', second)
 
        if uni[2] < outlier:
            o=o+1
            outlier_indices =[0,1,5,11,12,13]
            n = int(np.random.uniform(0, len(outlier_indices), 1)[0])
            #print("Outlier in record ", i, " with index ", n)
            v = moutlier(v, outlier_indices[n])
 
        mixed.append(v)
    print 'Added ',ms,' NA records, swapped ',sw,' fields in records and inserted outliers in ', o,' records.'
    create_csv(mixed)'''
	
 
