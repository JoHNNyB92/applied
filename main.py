import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import sys
from sklearn.model_selection import train_test_split
import random
import decimal 
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
    dfs_return=dfs
    if n_f!="c":
        for key in dfs.keys():
            if len(dfs.keys())!=cnt and key not in noisyless:
                rec=0
                row_iterator = dfs.iterrows()
                for i,elem in row_iterator:
                    prob=noisy_probs[key][rec]
                    if prob<feat_prob[key]:
                        digits = len(str(elem[key]-int(elem[key]))[1:])-1
                        if digits>4:
                            digits=3
                        print 'Adding noise to '+key+',record '+str(rec)+' ,previously value '+str(elem[key])+', noise '+str(noisy[key][rec])+',new value '+str(round(noisy[key][rec],digits))
                        element_noise=round(noisy[key][rec],digits)
                        dfs.at[i,key] = element_noise
                        for key2 in dfs.keys():
                             if key2!=key:   
                                 dfs.at[i,key2] = elem[key2]
                    rec=rec+1
            cnt=cnt+1
    else:
        keys=dfs.keys()
        target=keys[len(keys)-1]
        rec=0
        row_iterator = dfs.iterrows()
        for i,elem in row_iterator:
            prob=noisy_probs[target][rec]
            if prob<feat_prob[target]:
                digits = len(str(elem[target]-int(elem[target]))[1:])-1
                if digits>4:
                    digits=3
                print 'Adding noise to '+target+',record '+str(rec)+' ,previously value '+str(elem[target])+', noise '+str(noisy[target][rec])+',new value '+str(round(noisy[target][rec],digits))
                elem[target]=round(noisy[target][rec],digits)
                #elem[target]=elem[target]+noisy[target][rec]
                dfs.at[i,target] = elem[target]
                for key2 in dfs.keys():
                    if key2!=target:
                        dfs.at[i,key2] = elem[key2] 
            rec=rec+1
    return dfs

    
def write_excel_pd(pd_noise_array,folder,name,noisy,n_f,case_dc,case_tt):
    pathname=folder+'/'+name.split('.')[0]+'_'+str(noisy)+'_'+n_f+'_'+case_tt+'_'+case_dc+'.xlsx'
    writer = pd.ExcelWriter(pathname)
    pd_noise_array.to_excel(writer,'Sheet1')
    writer.save()

def skf_regression(dfs,pct):
    tmp=[]
    tens=[]
    cols=dfs.keys()
    target=cols[len(dfs.keys())-1]
    dfs=dfs.sort_values(by=[target])
    df_train = pd.DataFrame(pd.np.empty((0,len(cols))))
    df_train.columns=cols[:]
    df_test=df = pd.DataFrame(pd.np.empty((0,len(cols))))
    df_test.columns=cols[:]
    amount=int(pct*10)
    for x in range(len(dfs[cols[0]])):
        tmp.append(dfs.iloc[x])
        if x%10==9:
            tens.append(tmp)
            tmp=[]
    if len(tmp)!=0:
        tens.append(tmp)
    for o_l in tens:
        l=o_l[:]
        for m in range(amount):
            index = random.randrange(len(l))
            if index!=(len(l)-1):
                elem = l[index]
                l[index] = l[len(l)-1]
                df_test=df_test.append(elem)
                del l[-1]
            else:
                elem = l[index]
                df_test=df_test.append(elem)
                del l[-1]
        for x in l:
            df_train=df_train.append(x)
    return (df_train,df_test)

         
if __name__ == "__main__":
    noisy_folder=sys.argv[1]
    clean_dataset=sys.argv[2]
    dfs = pd.read_excel(noisy_folder+'/'+clean_dataset)
    df = pd.DataFrame(dfs, columns=dfs.keys())
    min_max={}
    noise_prop_dict={}
    noisyless=[]
    for key in df.keys():
        if(str(df[key].dtype) != 'datetime64[ns]' and str(df[key].dtype) != 'object'):
            print 'Filling empty values with 0 for feature '+key
            df[key].fillna(0, inplace=True)
        else:
            noisyless.append(key)
            print 'Column '+key+' type is not of a number type'
    length=df[key].count()
    for key in df.keys():
        if key not in noisyless:
            min_max[key]=[df.loc[df[key].idxmax()][key],df.loc[df[key].idxmin()][key]]
    pct=0.2
    (df_train,df_test)=skf_regression(df,pct)
    write_excel_pd(df_train,noisy_folder,clean_dataset,'','','clean','train')
    write_excel_pd(df_test,noisy_folder,clean_dataset,'','','','test')
    c=[0.05,0.10,0.35,0.5]
    case=['c','f']
    for noisy_pct in c:
        length=df[key].count()
        noise_prop = [noisy_pct for _ in range(length)]
        cnt=0
        for key in df.keys():
            if key not in noisyless:
                noise_prop_dict[key]=noise_prop[cnt]
                cnt=cnt+1
        for n_f in case:
            print 'Noise:'+str(noisy_pct)+' Case:'+n_f
            df_train_n=df_train.copy()
            headers=df.keys()  
            noisy={}
            noisy_probs={}
            if len(sys.argv)!=3:
                (noisy,noisy_probs)=generate_noise(min_max,headers,length,df_train_n,noisyless,n_f,True)
            else:
                (noisy,noisy_probs)=generate_noise(min_max,headers,length,df_train_n,noisyless,n_f)
            df_train_n_n=add_noise(noisy,noisy_probs,df_train_n,noise_prop_dict,noisyless,n_f)
            write_excel_pd(df_train_n_n,noisy_folder,clean_dataset,noisy_pct,n_f,'noisy','train')
    
    
    
	
 
