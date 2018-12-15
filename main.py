import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def generate_noise(min_max,keys,length,df,noisyless,n_f,enable_plot=False):
    noise_array={}
    records={}
    cnt=1
    if n_f=='f':
        for key in keys:
            if key not in noisyless and len(keys)!=cnt:
                print key
                mean=round((float)(min_max[key][0]-min_max[key][1])/2,1)
                std=round((float)(min_max[key][0]-min_max[key][1])/6,1)
                noise_array[key]=np.random.normal(mean, std, length)
                records[key]=np.random.uniform(low=0.0, high=1.0, size=length)
                if enable_plot==True:
                    plt.hist(noise_array[key], bins='auto')  # arguments are passed to np.histogram\
                    plt.title('NOISE FOR COLUMN:'+str(key)+" WITH MEAN:"+str(mean)+" STD:"+str(std))
                    plt.show()
                    plt.gcf().clear()
            cnt=cnt+1
    else:
        target=keys[len(keys)-1]
        mean=round((float)(min_max[target][0]-min_max[target][1])/2,1)
        std=round((float)(min_max[target][0]-min_max[target][1])/6,1)
        noise_array[target]=np.random.normal(mean, std, length)
        records[target]=np.random.uniform(low=0.0, high=1.0, size=length)
        if enable_plot==True:
            plt.hist(noise_array[target], bins='auto')  # arguments are passed to np.histogram\
            plt.title('NOISE FOR COLUMN:'+str(target)+" WITH MEAN:"+str(mean)+" STD:"+str(std))
            plt.show()
            plt.gcf().clear()
        
    return (noise_array,records)
            
        
def add_noise(noisy,noisy_probs,dfs,feat_prob,noisyless,n_f):
    cnt=1
    if n_f!="c":
        for key in dfs.keys():
            if len(dfs.keys())!=cnt and key not in noisyless:
                for rec in range(dfs[key].count()):
                    prob=noisy_probs[key][rec]
                    if prob<feat_prob[key]:
                        print 'Adding noise to '+key+',record '+str(rec)+' ,previously value '+str(dfs[key][rec])+', noise '+str(noisy[key][rec])+',new value '+str(dfs[key][rec]+noisy[key][rec])
                        dfs[key][rec]=dfs[key][rec]+noisy[key][rec]
            cnt=cnt+1
    else:
        keys=df.keys()
        target=keys[len(keys)-1]
        for rec in range(dfs[target].count()):
            prob=noisy_probs[target][rec]
            if prob<feat_prob[target]:
                print 'Adding noise to '+target+',record '+str(rec)+' ,previously value '+str(dfs[target][rec])+', noise '+str(noisy[target][rec])+',new value '+str(dfs[target][rec]+noisy[target][rec])
                dfs[target][rec]=dfs[target][rec]+noisy[target][rec]
    return dfs

    
def write_excel_pd(pd_noise_array,folder,name,noisy,n_f):
    pathname=folder+'/'+name.split('.')[0]+'_'+str(noisy)+'_'+n_f+'_noisy.xlsx'
    writer = pd.ExcelWriter(pathname)
    pd_noise_array.to_excel(writer,'Sheet1')
    writer.save()


if __name__ == "__main__":
    noisy_folder=sys.argv[1]
    clean_dataset=sys.argv[2]
    noisy_pct=float(sys.argv[3])
    n_f=sys.argv[4]
    print noisy_pct
    dfs = pd.read_excel(noisy_folder+'/'+clean_dataset)
    df = pd.DataFrame(dfs, columns=dfs.keys())
    min_max={}
    
    noise_prop_dict={}
    noisyless=[]
    cnt=0
    for key in df.keys():
        if(str(df[key].dtype) != 'datetime64[ns]' and str(df[key].dtype) != 'object'):
            print 'Filling empty values with 0 for feature '+key
            cnt=1+cnt
            df[key].fillna(0, inplace=True)
        else:
            noisyless.append(key)
            print 'Column '+key+' type is not of a number type'
    noise_prop = [noisy_pct for _ in range(cnt)]
    length=df[key].count()
    cnt=0
    for key in df.keys():
        if key not in noisyless:
            noise_prop_dict[key]=noise_prop[cnt]
            cnt+=1
            min_max[key]=[df.loc[df[key].idxmax()][key],df.loc[df[key].idxmin()][key]]
            print min_max[key]
    
    headers=df.keys()  
    noisy={}
    noisy_probs={}
    if len(sys.argv)!=5:
        (noisy,noisy_probs)=generate_noise(min_max,headers,length,dfs,noisyless,n_f,True)
    else:
        (noisy,noisy_probs)=generate_noise(min_max,headers,length,dfs,noisyless,n_f)
    dfs=add_noise(noisy,noisy_probs,df,noise_prop_dict,noisyless,n_f)
    write_excel_pd(dfs,noisy_folder,clean_dataset,noisy_pct,n_f)
	
 
