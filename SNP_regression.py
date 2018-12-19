import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
import pylab

import sklearn.feature_selection as sk
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from scipy.stats  import probplot

from pandas import DataFrame

import matplotlib.pyplot as plt

#
#
#X = full[[]]
#tup = [2,3,4,5,6,7]
#for k in range(len(full)):
#    tup.append(k)
#    
#tup = tuple(tup)
#X = full.iloc[:,[3,4,5,6,7,9,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]]
#print(X)
#
#
##print(full)
#
#X = np.array(X)
#
#SNP15 = pd.read_csv("AIHack-SNP-Matrix-15.csv")
#Clin = pd.read_csv("AIHack-Clinical.csv")
#PC = pd.read_csv("AIHack-PCs.csv")
#Clin = Clin.replace('nan', np.NaN)
#Clin.fillna(Clin.mean(), inplace = True)
#full = pd.concat([Clin, SNP15, PC], axis=1, join_axes=[Clin.index])
#full = full.drop("Unnamed: 0", axis=1)


low = 8
high = 16391

#===========================================================================================
# 1. Data Loading 
def load_data_CAD():
    SNP15 = pd.read_csv("D:/AI_Hack/GWAS_files/AIHack-SNP-Matrix-15.csv")
    Clin = pd.read_csv("D:/AI_Hack/GWAS_files/AIHack-Clinical.csv")
    PC = pd.read_csv("D:/AI_Hack/GWAS_files/AIHack-PCs.csv")
    Clin = Clin.replace('nan', np.NaN)
    Clin.fillna(Clin.mean(), inplace = True)
    full = pd.concat([Clin, SNP15, PC], axis=1, join_axes=[Clin.index])
    full = full.drop("Unnamed: 0", axis=1)
    CAD = full.iloc[:,1]
    return CAD
    
def load_data_HDL():
    SNP15 = pd.read_csv("AIHack-SNP-Matrix-15.csv")
    Clin = pd.read_csv("AIHack-Clinical.csv")
    PC = pd.read_csv("AIHack-PCs.csv")
    Clin = Clin.replace('nan', np.NaN)
    Clin.fillna(Clin.mean(), inplace = True)
    full = pd.concat([Clin, SNP15, PC], axis=1, join_axes=[Clin.index])
    full = full.drop("Unnamed: 0", axis=1)
    HDL = full.iloc[:,6]
    return HDL

def load_data_LDL():
    SNP15 = pd.read_csv("AIHack-SNP-Matrix-15.csv")
    Clin = pd.read_csv("AIHack-Clinical.csv")
    PC = pd.read_csv("AIHack-PCs.csv")
    Clin = Clin.replace('nan', np.NaN)
    Clin.fillna(Clin.mean(), inplace = True)
    full = pd.concat([Clin, SNP15, PC], axis=1, join_axes=[Clin.index])
    full = full.drop("Unnamed: 0", axis=1)
    LDL = full.iloc[:,7]
    return LDL
#===========================================================================================
# 2. Statistical learning for prediction of significant SNPs 

# Logistic Regression Model 
def calc_CAD(CAD):  
    SNP_pvalues = []
    for i in range(low, high):

        X = full.iloc[:,[2,3,4,5,6,7,i,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2]] #X Values for 1 SNP 
        
        X = np.array(X) #convert dataframe to array
       
        #create logisitic regression model, and find weights 
        clf = LogisticRegression(random_state=0, solver='lbfgs',
                                 multi_class='multinomial').fit(X, CAD)
        
        
        denom = (2.0*(1.0+np.cosh(clf.decision_function(X))))
        F_ij = np.dot((X/denom[:,None]).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.array([np.sqrt(Cramer_Rao[i,i]) for i in range(Cramer_Rao.shape[0])]) # sigma for each coefficient
        z_scores = clf.coef_[0]/sigma_estimates # z-score for eaach model coefficient
        p_values = [stats.norm.sf(abs(x))*2 for x in z_scores] ### `two tailed test for p-values
        SNP_pvalues.append(-np.log(p_values[5]))
    return SNP_pvalues
    
    
# Alternative methods but none gave successful Manhattan Plots 
# def svm(CAD):
#     SNP_pvalues = []
#     for i in range(low, high):
#         X = full.iloc[:,[2,3,4,5,6,7,i,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2]] #X Values for 1 SNP 
        
#         X = np.array(X) #convert dataframe to array
       
#         #create logisitic regression model, and find weights 
#         clf = LinearSVC(random_state=0, tol=1e-5).fit(X, CAD)
        
#         denom = (2.0*(1.0+np.cosh(clf.decision_function(X))))
#         F_ij = np.dot((X/denom[:,None]).T,X) ## Fisher Information Matrix
#         Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
#         sigma_estimates = np.array([np.sqrt(Cramer_Rao[i,i]) for i in range(Cramer_Rao.shape[0])]) # sigma for each coefficient
#         z_scores = clf.coef_[0]/sigma_estimates # z-score for eaach model coefficient
#         p_values = [stats.norm.sf(abs(x))*2 for x in z_scores] ### `two tailed test for p-values
#         SNP_pvalues.append(-np.log(p_values[5]))
#     return SNP_pvalues

# def forest(CAD):
#     SNP_pvalues = []
#     for i in range(8,9):
# #    range(low, high):
#         print (i)
#         X = full.iloc[:,[2,3,4,5,6,7,i,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2]] #X Values for 1 SNP 
        
#         X = np.array(X) #convert dataframe to array
       
#         #create logisitic regression model, and find weights 
#         clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0).fit(X, CAD)
        
        
#         denom = (2.0*(1.0+np.cosh(clf.oob_decision_function_[:,0])))
# #        F_ij = np.dot((X/denom[:,None]).T,X) ## Fisher Information Matrix
# #        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
# #        sigma_estimates = np.array([np.sqrt(Cramer_Rao[i,i]) for i in range(Cramer_Rao.shape[0])]) # sigma for each coefficient
# #        z_scores = clf.coef_[0]/sigma_estimates # z-score for eaach model coefficient
# #        p_values = [stats.norm.sf(abs(x))*2 for x in z_scores] ### `two tailed test for p-values
# #        SNP_pvalues.append(-np.log(p_values[5]))
#     return clf.oob_decision_function_


#===========================================================================================
# Flagging of missing data 
def flag_alleles(SNP_pvalues):
    lst = []
    col = []
    for i in range(len(SNP_pvalues)):
        if SNP_pvalues[i] > -np.log10((10**(-5)/(10**1))):
            col.append(full.columns[i+8]) 
    return col



def calc_HDL(HDL):
    SNP_pvalues = []
    for i in range(low,high):
        X = full.iloc[:,[1,2,3,4,5,7,i,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2]] #X Values for 1 SNP 
        
#        X = np.array(X, dtype = float) #convert dataframe to array
        
#        X = X + np.ones(shape = np.shape(X))
        fvalue, pvalue = sk.f_regression(X, HDL)
        SNP_pvalues.append(-np.log(pvalue[5]))
       
    return SNP_pvalues

def calc_LDL(LDL):
    SNP_pvalues = []
    for i in range(low,high):
        X = full.iloc[:,[2,3,4,5,6,i,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2]] #X Values for 1 SNP 
        X = np.array(X) #convert dataframe to array
        fvalue, pvalue = sk.f_regression(X, LDL)
        
#        for j in range(len(X)):
#            scipy.stats.chi2()
       
        SNP_pvalues.append(-np.log(pvalue[5]))
       
    return SNP_pvalues


#===========================================================================================
3. Manhattan Plots
def graph_CAD(SNP_pvalues):
    df = DataFrame({'gene' : ['gene-%i' % i for i in np.arange(len(SNP_pvalues))],
    'minuslog10pvalue' : SNP_pvalues})

    df['ind'] = range(len(df))
    print (df)
#    ax = plt.fig.add_subplot(111)
    plt.figure(1)
    colors = 'red'
    plt.scatter(df['ind'],df['minuslog10pvalue'],s=1)
    plt.plot(np.arange(0,17000,1), np.ones(17000)*-np.log10((10**(-5)/(17))),color = 'red', linewidth = '1')
    plt.gca()
    plt.title("Deviation between control and CAD phenotype allele frequencies")
    plt.xlabel(r'SNP site $i$')
    plt.ylabel(r'$-log(p_{i})$')
    plt.show()


def graph_HDL(SNP_pvalues):
    df = DataFrame({'gene' : ['gene-%i' % i for i in np.arange(len(SNP_pvalues))],
    'minuslog10pvalue' : SNP_pvalues})
    
    df['ind'] = range(len(df))
    plt.figure(2)
    ax = fig.add_subplot(111)
    plt.scatter(df['ind'],df['minuslog10pvalue'],s=1)
    plt.plot(np.arange(0,17000,1), np.ones(17000)*-np.log10((10**(-5)/(17))),color = 'red', linewidth = '1')
    plt.gca()
    plt.title("Deviation between control and HDL phenotype allele frequencies")
    plt.xlabel(r'SNP site $i$')
    plt.ylabel(r'$-log(p_{i})$')
    plt.show()

def graph_LDL(SNP_pvalues):
    df = DataFrame({'gene' : ['gene-%i' % i for i in np.arange(len(SNP_pvalues))],
    'minuslog10pvalue' : SNP_pvalues})
    
    df['ind'] = range(len(df))
    plt.figure(3)
    ax = fig.add_subplot(111)
    colors = 'red'
    plt.scatter(df['ind'],df['minuslog10pvalue'],s=1)
    plt.plot(np.arange(0,17000,1), np.ones(17000)*-np.log10((10**(-5)/(17))),color = 'red', linewidth = '1')
    plt.gca()
    plt.title("Deviation between control and LDL phenotype allele frequencies")
    plt.xlabel(r'SNP site $i$')
    plt.ylabel(r'$-log(p_{i})$')
    plt.show()
    
# def QQ_plots_CAD(x):
    
#     stats.probplot(x, dist = 'norm', plot=pylab)
#     pylab.show()
#     pass
