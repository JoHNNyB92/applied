import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import xlsxwriter
import os

path=''
filename=''

def generate_noise(min_max,length,keys,df,enable_plot):
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
        pd_noise_array=pd.DataFrame.from_dict(noise_array)    
        write_excel_pd(pd_noise_array)
 
def write_excel_pd(pd_noise_array):      
    os.chdir('/home/mscuser/Desktop/applied Data science/applied/'+path)
    pathname=filename+'_noisy.xlsx'
    
    writer = pd.ExcelWriter(pathname)
    pd_noise_array.to_excel(writer,'Sheet1')
    writer.save()


#def add_noise():
#    #TODO

def main(argv):

    global path
    global filename
    file_xlsx=sys.argv[1]
    path=file_xlsx.split('/')
    filename=path[1]
    filename=filename.split('.')
    filename=filename[0]
    path=path[0]
    column=sys.argv[2]
    print ("Add noise on file:",file_xlsx)
    print ("Add noise on feature:", column)
    print ("Path:",path)
    dfs = pd.read_excel(file_xlsx)
    length=dfs[column].count()
    print ("length:",length)
    noise_prop = [0.1 for _ in range(length)]
    noise_prop_dict={}
    df = pd.DataFrame(dfs, columns=dfs.keys())
    min_max={}
    df=df.drop('Player', axis=1)
    cnt=0
    for key in df.keys():
        noise_prop_dict[key]=noise_prop[cnt]
        cnt+=1
        min_max[key]=[df.loc[df[key].idxmax()][key],df.loc[df[key].idxmin()][key]]
        print (min_max[key])
    
    headers=df.keys()
    if len(sys.argv)!=3:
        generate_noise(min_max,length,headers,dfs,True)
    else:
        generate_noise(min_max,length,headers,dfs,False)
    
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

if __name__ == "__main__":
    main(sys.argv[1:])
    
    
    
	
 
