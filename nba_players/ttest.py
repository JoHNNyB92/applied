import sys
import csv
import os
import numpy as np
from scipy import stats



def main(argv):

 print 'Number of arguments:', len(sys.argv), 'arguments.'
 print 'Argument List:', str(sys.argv)

 path= sys.argv[1].split('/')
 os.chdir(path[0])
 print path[0]
 print path[1]
 
 
 
 with open(path[1]) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    data=''
    for row in csv_reader:
       data = row
    LR=data[0]
    Lasso=data[1]
    Ridge=data[2]
    Ytest=data[3]
    
    LR=LR.replace("]", ",")
    LR=LR.replace("[", "")
    LR=LR.replace(",,", "")
    LR=LR.replace("\n", "")   
    LR=map(float, LR.split(','))
    
    
    Lasso=Lasso.replace("]", ",")
    Lasso=Lasso.replace("[", "")
    Lasso=Lasso.replace(",,", "")
    Lasso=Lasso.replace("\n", "")   
    Lasso=map(float, Lasso.split(','))
    
    Ridge=Ridge.replace("]", ",")
    Ridge=Ridge.replace("[", "")
    Ridge=Ridge.replace(",,", "")
    Ridge=Ridge.replace("\n", "")   
    Ridge=map(float, Ridge.split(','))


 '''
    print Ytest

    Ytest=Ytest.replace("]", "")
    Ytest=Ytest.replace("[", "")
    Ytest=Ytest.replace(" ", ",")
    Ytest=Ytest.replace(",,", ",")
    Ytest=Ytest.replace(" ", "")
    Ytest=Ytest.replace("\n", "")   
    Ytest=Ytest.split(',')
    print Ytest
    for i in range(len(Ytest)):
        print Ytest[i]
        Ytest[i]=Ytest[i].replace(' ', '')
        Ytest[i]=float(Ytest[i])
    
    
    print Ytest
    
 '''



 N = 84

 a = Lasso
 b = Ridge

 var_a = np.var(a)
 var_b = np.var(b)
 s = np.sqrt((var_a + var_b)/2)
 t = (np.mean(a) - np.mean(b))/(s*np.sqrt(2/N))
 df = 2*N - 2
 p = 1 - stats.t.cdf(t,df=df)



 t2, p2 = stats.ttest_ind(a,b)
 print("t = " + str(t2))
 print("p = " + str(2*p2))




if __name__ == "__main__":
   main(sys.argv[1:])
