# Required function for running this assignment
# Written by Mehdi Rezvandehy


import numpy as np
import pandas as pd
import math
from matplotlib.offsetbox import AnchoredText
from typing import Callable
from scipy.stats import gaussian_kde
import scipy.linalg 
import scipy.stats as ss
from scipy.stats import norm
from scipy.stats import reciprocal, uniform
from scipy.stats import randint
from scipy.stats import mode
from IPython.display import display, Math, Latex
from matplotlib.ticker import PercentFormatter
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from IPython.display import HTML
import itertools
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold


##########################################################################################################

class EDA_plot:
    def histplt (val: list,bins: int,title: str,xlabl: str,ylabl: str,xlimt: list,
                 ylimt: list=False, loc: int =1,legend: int=1,axt=None,days: int=False,
                 class_: int=False,scale: int=1,x_tick: list=False,
                 nsplit: int=1,font: int=5,color: str='b') -> None :
        
        """ Make histogram of data"""
        
        ax1 = axt or plt.axes()
        font = {'size'   : font }
        plt.rc('font', **font) 
        
        val=val[~np.isnan(val)]
        val=np.array(val)
        plt.hist(val, bins=bins, weights=np.ones(len(val)) / len(val),ec='black',color=color)
        n=len(val[~np.isnan(val)])
        Mean=np.nanmean(val)
        Median=np.nanmedian(val)
        SD=np.sqrt(np.nanvar(val))
        Max=np.nanmax(val)
        Min=np.nanmin(val)
    
        
        txt='n=%.0f\nMean=%0.2f\nMedian=%0.1f\nσ=%0.1f\nMax=%0.1f\nMin=%0.1f'       
        anchored_text = AnchoredText(txt %(n,Mean,Median,SD,Max,Min), borderpad=0, 
                                     loc=loc,prop={ 'size': font['size']*scale})    
        if(legend==1): ax1.add_artist(anchored_text)
        if (scale): plt.title(title,fontsize=font['size']*(scale+0.15))
        else:       plt.title(title)
        plt.xlabel(xlabl,fontsize=font['size']) 
        ax1.set_ylabel('Frequency',fontsize=font['size'])
        if (scale): ax1.set_xlabel(xlabl,fontsize=font['size']*scale)
        else:       ax1.set_xlabel(xlabl)
    
        try:
            xlabl
        except NameError:
            pass    
        else:
            if (scale): plt.xlabel(xlabl,fontsize=font['size']*scale) 
            else:        plt.xlabel(xlabl)   
            
        try:
            ylabl
        except NameError:
            pass      
        else:
            if (scale): plt.ylabel(ylabl,fontsize=font['size']*scale)  
            else:         plt.ylabel(ylabl)  
            
        if (class_==True): plt.xticks([0,1])
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        ax1.grid(linewidth='0.1')
        try:
            xlimt
        except NameError:
            pass  
        else:
            plt.xlim(xlimt) 
            
        try:
            ylimt
        except NameError:
            pass  
        else:
            plt.ylim(ylimt)  
            
        if x_tick: plt.xticks(x_tick,fontsize=font['size']*scale)    
        plt.yticks(fontsize=font['size']*scale)               
    
    ######################################################################### 
            
    def KDE(xs: list,data_var: list,nvar: int,clmn: [str],color: [str],xlabel: str='DE Length',
            title: str='Title',ylabel: str='Percentage',LAMBA: float =0.3,linewidth: float=2.5,
            loc: int=0,axt=None,xlim: list=(0,40),ylim: list=(0,0.1),x_ftze: float =13,
            y_ftze: float=13,tit_ftze: float=13,leg_ftze: float=9) -> None :
        
        """
        Kernel Density Estimation (Smooth Histogram)
         
        """
        ax1 = axt or plt.axes()
        var_m=[]
        var_med=[]
        var_s=[]
        var_n=[]
        s1=[]
        data_var_=np.array([[None]*nvar]*len(xs), dtype=float)
        # Loop over variables
        for i in range (nvar):
            data = data_var[i]
            var_m.append(np.mean(data).round(2))
            var_med.append(np.median(data).round(2))
            var_s.append(np.var(data).round(1))
            var_n.append(len(data))
            density = gaussian_kde(data)
            density.set_bandwidth(LAMBA)
            density_=density(xs)/sum(density(xs))
            data_var_[:,i]=density_
            linestyle='solid'
            plt.plot(xs,density_,color=color[i],linestyle=linestyle, linewidth=linewidth)
            
        #############
        
        data_var_tf=np.array([[False]*nvar]*len(data_var_))
        for j in range(len(data_var_)):
            data_tf_t=[]
            for i in range (nvar):
                if (data_var_[j,i]==max(data_var_[j,:])):
                    data_var_tf[j,i]=True     
        #############            
        for i in range (nvar):
            plt.fill_between(np.array(xs),np.array(data_var_[:,i]),where=np.array(data_var_tf[:,i]),
                             color=color[i],alpha=0.9,label=clmn[i]+': n='+str(var_n[i])+
                             ', mean= '+str(var_m[i])+', median= '+str(var_med[i])+
                             ', '+r"$\sigma^{2}$="+str(var_s[i]))
        
        plt.xlabel(xlabel,fontsize=x_ftze, labelpad=6)
        plt.ylabel(ylabel,fontsize=y_ftze)
        plt.title(title,fontsize=tit_ftze)
        plt.legend(loc=loc,fontsize=leg_ftze,markerscale=1.2)
        
        ax1.grid(linewidth='0.2')
        plt.xlim(xlim) 
        plt.ylim(ylim) 

    ######################################################################### 
                        
    def CDF_plot(data_var: list,nvar: int,label:str,colors:str,title:str,xlabel:str,
                 ylabel:str='Cumulative Probability', bins: int =1000,xlim: list=(0,100),
                 ylim: list=(0,0.01),linewidth: float =2.5,loc: int=0,axt=None,
                 x_ftze: float=12,y_ftze: float=12,tit_ftze: float=12,leg_ftze: float=9) -> None:
        
        """
        Cumulative Distribution Function
         
        """
        ax1 = axt or plt.axes() 
        def calc(data:[float])  -> [float]:
            var_mean=np.nanmean(data).round(2)
            var_median=np.nanmedian(data).round(2)
            var_s=np.var(data).round(1)
            var_n=len(data)
            val_=np.array(data)
            counts, bin_edges = np.histogram(val_[~np.isnan(val_)], bins=bins,density=True)
            cdf = np.cumsum(counts)
            tmp=max(cdf)
            cdf=cdf/float(tmp)
            return var_mean,var_median,var_s,var_n,bin_edges,cdf
               
        if nvar==1:
            var_mean,var_median,var_s,var_n,bin_edges,cdf=calc(data_var)
            if label:
                label_=f'{label} : n={var_n}, mean= {var_mean}, median= {var_median}, $\sigma^{2}$={var_s}'
                plt.plot(bin_edges[1:], cdf,color=colors, linewidth=linewidth,
                    label=label_)                
            else:
                plt.plot(bin_edges[1:], cdf,color=colors, linewidth=linewidth)

        else:    
            # Loop over variables
            for i in range (nvar):
                data = data_var[i]
                var_mean,var_median,var_s,var_n,bin_edges,cdf=calc(data)
                label_=f'{label[i]} : n={var_n}, mean= {var_mean}, median= {var_median}, $\sigma^{2}$={var_s}'
                plt.plot(bin_edges[1:], cdf,color=colors[i], linewidth=linewidth,
                        label=label_)
         
        plt.xlabel(xlabel,fontsize=x_ftze, labelpad=6)
        plt.ylabel(ylabel,fontsize=y_ftze)
        plt.title(title,fontsize=tit_ftze)
        if label:
            plt.legend(loc=loc,fontsize=leg_ftze,markerscale=1.2)
        
        ax1.grid(linewidth='0.2')
        plt.xlim(xlim) 
        plt.ylim(ylim)         

    ######################################################################### 
            
    def CrossPlot (x:list,y:list,title:str,xlabl:str,ylabl:str,loc:int,
                   xlimt:list,ylimt:list,axt=None,scale: float=0.8,alpha: float=0.6,
                   markersize: float=6,marker: str='ro', fit_line: bool=False, 
                   font: int=5) -> None:
        """
        Cross plto between two variables
         
        """
        ax1 = axt or plt.axes()
        font = {'size'   : font }
        plt.rc('font', **font) 

        x=np.array(x)
        y=np.array(y)    
        no_nan=np.where((~np.isnan(x)) & (~np.isnan(y)))[0]
        Mean_x=np.mean(x)
        SD_x=np.sqrt(np.var(x)) 
        #
        n_x=len(x)
        n_y=len(y)
        Mean_y=np.mean(y)
        SD_y=np.sqrt(np.var(y)) 
        corr=np.corrcoef(x[no_nan],y[no_nan])
        n_=len(no_nan)
        #txt=r'$\rho_{x,y}=$%.2f'+'\n $n=$%.0f '
        #anchored_text = AnchoredText(txt %(corr[1,0], n_),borderpad=0, loc=loc,
        #                         prop={ 'size': font['size']*0.95, 'fontweight': 'bold'})  
        
        txt=r'$\rho_{x,y}}$=%.2f'+'\n $n$=%.0f \n $\mu_{x}$=%.0f \n $\sigma_{x}$=%.0f \n '
        txt+=' $\mu_{y}$=%.0f \n $\sigma_{y}$=%.0f'
        anchored_text = AnchoredText(txt %(corr[1,0], n_x,Mean_x,SD_x,Mean_y,SD_y), loc=4,
                                prop={ 'size': font['size']*1.1, 'fontweight': 'bold'})    
            
        ax1.add_artist(anchored_text)
        Lfunc1=np.polyfit(x,y,1)
        vEst=Lfunc1[0]*x+Lfunc1[1]    
        try:
            title
        except NameError:
            pass  # do nothing! 
        else:
            plt.title(title,fontsize=font['size']*(scale))   
    #
        try:
            xlabl
        except NameError:
            pass  # do nothing! 
        else:
            plt.xlabel(xlabl,fontsize=font['size']*scale)            
    #
        try:
            ylabl
        except NameError:
            pass  # do nothing! 
        else:
            plt.ylabel(ylabl,fontsize=font['size']*scale)        
            
        try:
            xlimt
        except NameError:
            pass  # do nothing! 
        else:
            plt.xlim(xlimt)   
    #        
        try:
            ylimt
        except NameError:
            pass  # do nothing! 
        else:
            plt.ylim(ylimt)   
          
        plt.plot(x,y,marker,markersize=markersize,alpha=alpha)  
        if fit_line:
            ax1.plot(x, vEst,'k-',linewidth=2)   
        ax1.grid(linewidth='0.1') 
        plt.xticks(fontsize=font['size']*0.85)    
        plt.yticks(fontsize=font['size']*0.85)    
        
    #########################################################################         
        
class Correlation_plot:
    def corr_mat(df: pd.DataFrame, title: str, corr_val_font: float=False, y_l: list=1.2,axt: plt.Axes=None,
                titlefontsize: int=10, xyfontsize: int=6, xy_title: list=[-22,1.2],
                vlim=[-0.8,0.8]) -> [float]:
        
        """Plot correlation matrix between features"""
        ax = axt or plt.axes()
        colmn=list(df.columns)
        corr=df.corr().values
        corr_array=[]
        for i in range(len(colmn)):
            for j in range(len(colmn)):
                c=corr[j,i]
                if (corr_val_font):
                        ax.text(j, i, str(round(c,2)), va='center', ha='center',fontsize=corr_val_font)
                if i>j:
                    corr_array.append(c)

        im =ax.matshow(corr, cmap='jet', interpolation='nearest',vmin=vlim[0], vmax=vlim[1])
        
        cbar =plt.colorbar(im,shrink=0.5,label='Correlation Coefficient')
        cbar.ax.tick_params(labelsize=10) 
        
        ax.set_xticks(np.arange(len(corr)))
        ax.set_xticklabels(colmn,fontsize=xyfontsize, rotation=90)
        ax.set_yticks(np.arange(len(corr)))
        ax.set_yticklabels(colmn,fontsize=xyfontsize)
        ax.grid(color='k', linestyle='-', linewidth=0.025)
        plt.text(xy_title[0],xy_title[1],title, 
                 fontsize=titlefontsize,bbox=dict(facecolor='white', alpha=0.2))
        return corr_array
        plt.show()
        
        
    #########################  
    
    def corr_bar(corr: list, clmns: str,title: str, select: bool= False
                ,yfontsize: float=4.6, xlim: list=[-0.5,0.5], ymax_vert_lin: float= False) -> None:
        
        """Plot correlation bar with target"""
        
        r_ = pd.DataFrame( { 'coef': corr, 'positive': corr>=0  }, index = clmns )
        r_ = r_.sort_values(by=['coef'])
        if (select):
            selected_features=abs(r_['coef'])[:select].index
            r_=r_[r_.index.isin(selected_features)]
    
        r_['coef'].plot(kind='barh',edgecolor='black',linewidth=0.8
                        , color=r_['positive'].map({True: 'r', False: 'b'}))
        plt.xlabel('Correlation Coefficient',fontsize=6)
        if (ymax_vert_lin): plt.vlines(x=0,ymin=-0.5, ymax=ymax_vert_lin, color = 'k',linewidth=1.2)
        plt.yticks(np.arange(len(r_.index)), r_.index,rotation=0,fontsize=yfontsize,x=0.01)
        plt.title(title)
        plt.xlim((xlim[0], xlim[1])) 
        ax1 = plt.gca()
        ax1.xaxis.grid(color='k', linestyle='-', linewidth=0.1)
        ax1.yaxis.grid(color='k', linestyle='-', linewidth=0.1)
        plt.show()  
        
##############################################################

def Conf_Matrix(y_train: [float],y_train_pred:[float], label: [str] ,perfect: int= 0,axt=None,plot: bool =True,
               title: bool =False,t_fontsize: float =8.5,t_y: float=1.2,x_fontsize: float=6.5,
               y_fontsize: float=6.5,trshld: float=0.5) -> [float]:
    
    '''Plot confusion matrix'''
    
    if (y_train_pred.shape[1]==2):
        y_train_pred=[0 if y_train_pred[i][0]>trshld else 1 for i in range(len(y_train_pred))]
    elif (y_train_pred.shape[1]==1):
        y_train_pred=[1 if y_train_pred[i][0]>trshld else 0 for i in range(len(y_train_pred))] 
    else:    
        y_train_pred=[1 if i>trshld else 0 for i in y_train_pred]       
    conf_mx=confusion_matrix(y_train,y_train_pred)
    acr=accuracy_score(y_train,y_train_pred)
    conf_mx =confusion_matrix(y_train,y_train_pred)
    prec=precision_score(y_train,y_train_pred) # == TP/(TP+FP) 
    reca=recall_score(y_train,y_train_pred) # == TP/(TP+FN) ) 
    TN=conf_mx[0][0] ; FP=conf_mx[0][1]
    spec= TN/(TN+FP)        
    if(plot):
        ax1 = axt or plt.axes()
        
        if (perfect==1): y_train_pred=y_train
        
        x=[f'Predicted {label[0]}', f'Predicted {label[1]}']; y=[f'Actual {label[0]}', f'Actual {label[1]}']
        ii=0 
        im =ax1.matshow(conf_mx, cmap='jet', interpolation='nearest') 
        for (i, j), z in np.ndenumerate(conf_mx): 
            if(ii==0): al='TN= '
            if(ii==1): al='FP= '
            if(ii==2): al='FN= '
            if(ii==3): al='TP= '          
            ax1.text(j, i, al+'{:0.0f}'.format(z), color='w', ha='center', va='center', fontweight='bold',fontsize=6.5)
            ii=ii+1
     
        txt='$ Accuracy\,\,\,$=%.2f\n$Sensitivity$=%.2f\n$Precision\,\,\,\,$=%.2f\n$Specificity$=%.2f'
        anchored_text = AnchoredText(txt %(acr,reca,prec,spec), loc=10, borderpad=0)
        ax1.add_artist(anchored_text)    
        
        ax1.set_xticks(np.arange(len(x)))
        ax1.set_xticklabels(x,fontsize=x_fontsize,y=0.97, rotation='horizontal')
        ax1.set_yticks(np.arange(len(y)))
        ax1.set_yticklabels(y,fontsize=y_fontsize,x=0.035, rotation='horizontal') 
        
        cbar =plt.colorbar(im,shrink=0.3,
                           label='Low                              High',orientation='vertical')   
        cbar.set_ticks([])
        plt.title(title,fontsize=t_fontsize,y=t_y)
    return acr, prec, reca, spec 

############################################################
        
def AUC(prediction: [float],y_train: [float], n_algorithm: int
       ,label:[str],title: str='Receiver Operating Characteristic (ROC)'
       ,linewidth=2) -> None:
    
    '''Plot Receiver Operating Characteristic (ROC) for predictors'''
    
    color=['b','r','g','y','m','c']
    for i in range(n_algorithm):
        fpr, tpr, thresold = roc_curve(y_train, prediction[i][:,1])
        roc_auc = auc(fpr, tpr)
        if (i==0):
            tmp_linewidth=4
            cm='k--'
        else:
            tmp_linewidth=linewidth
            cm= f'{color[i]}-'
            
        plt.plot(fpr, tpr,cm, linewidth=tmp_linewidth,
                 label=label[i]+' (AUC =' + r"$\bf{" + str(np.round(roc_auc,3)) + "}$"+')')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (1-Specificity) FP/(FP+TN)',fontsize=12)
    plt.ylabel('True Positive Rate (Sensistivity) TP/(TP+FN)',fontsize=12)
    plt.title(title,fontsize=15)
    plt.grid(linewidth='0.25')
    plt.legend(loc="lower right",fontsize=11)
    plt.show()     
    
############################################################################# 

class prfrmnce_plot(object):
    """Plot performance of features to predict a target"""
    def __init__(self,importance: list, title: str, ylabel: str,clmns: str, 
                titlefontsize: int=10, xfontsize: int=5, yfontsize: int=8) -> None:
        self.importance    = importance
        self.title         = title 
        self.ylabel        = ylabel  
        self.clmns         = clmns  
        self.titlefontsize = titlefontsize 
        self.xfontsize     = xfontsize 
        self.yfontsize     = yfontsize
        
    #########################    
    
    def bargraph(self, select: bool= False, fontsizelable: bool= False, xshift: float=-0.1, nsim: int=False
                 ,yshift: float=0.01,perent: bool=False, xlim: list=False,axt=None,
                 ylim: list=False, y_rot: int=0) -> pd.DataFrame():
        ax1 = axt or plt.axes()
        if not nsim:
            # Make all negative coefficients to positive
            sort_score=sorted(zip(abs(self.importance),self.clmns), reverse=True)
            Clmns_sort=[sort_score[i][1] for i in range(len(self.clmns))]
            sort_score=[sort_score[i][0] for i in range(len(self.clmns))]
        else:
            importance_agg=[]
            importance_std=[]
            for iclmn in range(len(self.clmns)):
                tmp=[]
                for isim in range(nsim):
                    tmp.append(abs(self.importance[isim][iclmn]))
                importance_agg.append(np.mean(tmp))
                importance_std.append(np.std(tmp))
                
            # Make all negative coefficients to positive
            sort_score=sorted(zip(importance_agg,self.clmns), reverse=True)
            Clmns_sort=[sort_score[i][1] for i in range(len(self.clmns))]
            sort_score=[sort_score[i][0] for i in range(len(self.clmns))]                
            

        index1 = np.arange(len(self.clmns))
        # select the most important features
        if (select):
            Clmns_sort=Clmns_sort[:select]
            sort_score=sort_score[:select]
        ax1.bar(Clmns_sort, sort_score, width=0.6, align='center', alpha=1, edgecolor='k', capsize=4,color='b')
        plt.title(self.title,fontsize=self.titlefontsize)
        ax1.set_ylabel(self.ylabel,fontsize=self.yfontsize)
        ax1.set_xticks(np.arange(len(Clmns_sort)))
        
        ax1.set_xticklabels(Clmns_sort,fontsize=self.xfontsize, rotation=90,y=0.02)   
        if (perent): plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        ax1.xaxis.grid(color='k', linestyle='--', linewidth=0.2) 
        if (xlim): plt.xlim(xlim)
        if (ylim): plt.ylim(ylim)
        if (fontsizelable):
            for ii in range(len(sort_score)):
                if (perent):
                    plt.text(xshift+ii, sort_score[ii]+yshift,f'{"{0:.1f}".format(sort_score[ii]*100)}%',
                    fontsize=fontsizelable,rotation=y_rot,color='k')     
                else:
                    plt.text(xshift+ii, sort_score[ii]+yshift,f'{"{0:.3f}".format(sort_score[ii])}',
                fontsize=fontsizelable,rotation=y_rot,color='k')                        
                    
        
        dic_Clmns={}
        for i in range(len(Clmns_sort)):
            dic_Clmns[Clmns_sort[i]]=sort_score[i]
            
        return  pd.DataFrame(dic_Clmns.items(), columns=['Features', 'Scores'])  
        plt.show()   
        
    #########################    
    
    def Conf_Matrix(y_train: [float],y_train_pred:[float],perfect: int= 0,axt=None,plot: bool =True,
                   title: bool =False,t_fontsize: float =8.5,t_y: float=1.2,x_fontsize: float=6.5,
                   y_fontsize: float=6.5,trshld: float=0.5) -> [float]:
        
        '''Plot confusion matrix'''
        
        if (y_train_pred.shape[1]==2):
            y_train_pred=[0 if y_train_pred[i][0]>trshld else 1 for i in range(len(y_train_pred))]
        elif (y_train_pred.shape[1]==1):
            y_train_pred=[1 if y_train_pred[i][0]>trshld else 0 for i in range(len(y_train_pred))] 
        else:    
            y_train_pred=[1 if i>trshld else 0 for i in y_train_pred]       
        conf_mx=confusion_matrix(y_train,y_train_pred)
        acr=accuracy_score(y_train,y_train_pred)
        conf_mx =confusion_matrix(y_train,y_train_pred)
        prec=precision_score(y_train,y_train_pred) # == TP/(TP+FP) 
        reca=recall_score(y_train,y_train_pred) # == TP/(TP+FN) ) 
        TN=conf_mx[0][0] ; FP=conf_mx[0][1]
        spec= TN/(TN+FP)        
        if(plot):
            ax1 = axt or plt.axes()
            
            if (perfect==1): y_train_pred=y_train
            
            x=['Predicted \n Negative', 'Predicted \n Positive']; y=['Actual \n Negative', 'Actual \n Positive']
            ii=0 
            im =ax1.matshow(conf_mx, cmap='jet', interpolation='nearest') 
            for (i, j), z in np.ndenumerate(conf_mx): 
                if(ii==0): al='TN= '
                if(ii==1): al='FP= '
                if(ii==2): al='FN= '
                if(ii==3): al='TP= '          
                ax1.text(j, i, al+'{:0.0f}'.format(z), color='w', ha='center', va='center', fontweight='bold',fontsize=6.5)
                ii=ii+1
         
            txt='$ Accuracy\,\,\,$=%.2f\n$Sensitivity$=%.2f\n$Precision\,\,\,\,$=%.2f\n$Specificity$=%.2f'
            anchored_text = AnchoredText(txt %(acr,reca,prec,spec), loc=10, borderpad=0)
            ax1.add_artist(anchored_text)    
            
            ax1.set_xticks(np.arange(len(x)))
            ax1.set_xticklabels(x,fontsize=x_fontsize,y=0.97, rotation='horizontal')
            ax1.set_yticks(np.arange(len(y)))
            ax1.set_yticklabels(y,fontsize=y_fontsize,x=0.035, rotation='horizontal') 
            
            cbar =plt.colorbar(im,shrink=0.3,
                               label='Low                              High',orientation='vertical')   
            cbar.set_ticks([])
            plt.title(title,fontsize=t_fontsize,y=t_y)
        return acr, prec, reca, spec    
        
    #########################    
    
    def AUC(prediction_prob: [float],y_train: [float], n_algorithm: int
           ,label:[str],title: str='Receiver Operating Characteristic (ROC)'
           ,linewidth=2) -> None:
        
        '''Plot Receiver Operating Characteristic (ROC) for predictors'''
        
        color=['b','r','g','y','m','c']
        for i in range(n_algorithm):
            if (prediction_prob[i].shape[1]==2):
                fpr, tpr, thresold = roc_curve(y_train, prediction_prob[i][:,1])
            else:
                fpr, tpr, thresold = roc_curve(y_train, prediction_prob[i])
            roc_auc = auc(fpr, tpr)
            if (i==0):
                tmp_linewidth=4
                cm='k--'
            else:
                tmp_linewidth=linewidth
                cm= f'{color[i-1]}-'
                
            plt.plot(fpr, tpr,cm, linewidth=tmp_linewidth,
                     label=label[i]+' (AUC =' + r"$\bf{" + str(np.round(roc_auc,3)) + "}$"+')')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (1-Specificity) FP/(FP+TN)',fontsize=12)
        plt.ylabel('True Positive Rate (Sensistivity) TP/(TP+FN)',fontsize=12)
        plt.title(title,fontsize=15)
        plt.grid(linewidth='0.25')
        plt.legend(loc="lower right",fontsize=11)
        plt.show()    
        
        
        
        
        
        
#############################################################################

def ANN (input_dim,neurons=50,loss="binary_crossentropy",activation="relu",Nout=1,
             metrics=['accuracy'],activation_out='sigmoid',init_mode=None,BatchOpt=False,dropout_rate=False):
    """ Function to run Neural Network for different hyperparameters"""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    if(activation=='Leaky_ReLU'): activation = keras.layers.LeakyReLU(alpha=0.2)
        
    # create model
    model = keras.models.Sequential()
    
    # Input & Hidden Layer 1
    model.add(keras.layers.Dense(neurons,input_dim=input_dim, activation=activation, kernel_initializer=init_mode))
        
    # Hidden Layer 2
    model.add(keras.layers.Dense(neurons,activation=activation)) 
    if(BatchOpt): model.add(keras.layers.BatchNormalization())
    if(dropout_rate):  model.add(keras.layers.Dropout(dropout_rate))
        
    # Hidden Layer 3    
    model.add(keras.layers.Dense(neurons,activation=activation)) 
    if(BatchOpt): model.add(keras.layers.BatchNormalization())
    if(dropout_rate):  model.add(keras.layers.Dropout(dropout_rate))
    
    # Hidden Layer 4    
    model.add(keras.layers.Dense(neurons,activation=activation)) 
    if(BatchOpt): model.add(keras.layers.BatchNormalization())
    if(dropout_rate):  model.add(keras.layers.Dropout(dropout_rate))    
    
    # Hidden Layer 5    
    model.add(keras.layers.Dense(neurons,activation=activation)) 
    if(BatchOpt): model.add(keras.layers.BatchNormalization())
    if(dropout_rate):  model.add(keras.layers.Dropout(dropout_rate))        
    
    # Output Layer 
    model.add(keras.layers.Dense(Nout,activation=activation_out)) 
        
    # Compile model
    model.compile(optimizer='adam',loss=loss,metrics=metrics)
    return model

#######################################################################   
   
def NN_plot(history):

    font = {'size'   : 10}
    plt.rc('font', **font)
    fig, ax=plt.subplots(figsize=(12, 4), dpi= 110, facecolor='w', edgecolor='k')
    
    ax1 = plt.subplot(1,2,1)
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'r--o', markersize=8, label='Training loss')          
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss',linewidth=2)    
    plt.title('Training and Validation Loss',fontsize=14)
    plt.xlabel('Epochs (Early Stopping)',fontsize=12)
    plt.ylabel('Loss',fontsize=11)
    plt.legend(fontsize='12')
    #plt.ylim((0.387, 0.405))
    
    ax2 = plt.subplot(1,2,2)    
    history_dict = history.history
    loss_values = history_dict['accuracy']
    val_loss_values = history_dict['val_accuracy']
    epochs = range(1, len(loss_values) + 1)
    ax2.plot(epochs, loss_values, 'r--o', markersize=8, label='Training accuracy')          
    ax2.plot(epochs, val_loss_values, 'b', label='Validation accuracy',linewidth=2)    
    plt.title('Training and Validation Accuracy',fontsize=14)
    plt.xlabel('Epochs (Early Stopping)',fontsize=12)
    plt.ylabel('Accuracy',fontsize=12)
    plt.legend(fontsize='12')
    #plt.ylim((0.79, 0.799))
    plt.show()
    
#######################################################################   
   
def BS_ANN (x_train,y_train,x_Validation,y_Validation,neurons=50,loss="binary_crossentropy",activation="relu"
            ,Nout=1,metrics=['accuracy'],activation_out='sigmoid',init_mode=None,BatchOpt=False,dropout_rate=False):
    """ Function to run Neural Network for different hyperparameters for Bootstraping"""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    if(activation=='Leaky_ReLU'): activation = keras.layers.LeakyReLU(alpha=0.2)
        
    # create model
    model = keras.models.Sequential()
    
    # Input & Hidden Layer 1
    model.add(keras.layers.Dense(neurons,input_dim=np.array(x_train).shape[1], activation=activation, kernel_initializer=init_mode))
        
    # Hidden Layer 2
    model.add(keras.layers.Dense(neurons,activation=activation)) 
    if(BatchOpt): model.add(keras.layers.BatchNormalization())
        
    # Hidden Layer 3    
    model.add(keras.layers.Dense(neurons,activation=activation)) 
    if(BatchOpt): model.add(keras.layers.BatchNormalization())
    if(dropout_rate):  model.add(keras.layers.Dropout(dropout_rate))
    
    # Output Layer 
    model.add(keras.layers.Dense(Nout,activation=activation_out)) 
        
    # Compile model
    model.compile(optimizer='adam',loss=loss,metrics=metrics)
    
    # Early stopping to avoid overfitting
    monitor= tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5,patience=5, mode='auto')
    history=model.fit(x_train,y_train,batch_size=32,validation_data=
              (x_Validation,y_Validation),callbacks=[monitor],verbose=0,epochs=1000)
    history_dict = history.history
    loss_values = history_dict['loss']
    epochs = len(loss_values)-1    
    
    return model, epochs    


#####################################################################

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, x=0.8):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "y-", label="Sensitivity", linewidth=2)
    plt.legend(loc="center right", fontsize=16) 
    plt.xlabel("Threshold", fontsize=12) 
    plt.ylabel("Precision/Sensitivity", fontsize=12)     
    plt.title("Precision and Sensitivity versus Threshold", fontsize=12)   
    plt.grid(linewidth='0.25')   
    plt.axvline(x=x,linestyle='--',color='k',label='Probability '+str(np.round(x,2)))
    plt.legend(loc=3, ncol=3,fontsize=10,markerscale=1.2, edgecolor="black",framealpha=0.9)
    plt.axis([0, 1, 0, 1])  

##########################################################################################################

def Over_Under_Sampling(x,y,ind,r1,r2,corr,seed,mr=False,r3=False):
    """
    Oversampling and undersampling:
    
    x    : Distribution of each feature for x
    y    : Distribution of each binary class for target
    corr : Correlation matrix between features
    ind  : oversampling if ind=0, undersampling if ind=1, both if ind=2
    r1   : Oversampling ratio for minory/majority 
    r2   : Undersampling ratio for minory/majority 
    seed : Random number seed
    mr   : Using missing ratio for each row for undersampling
    r3   : Missing ratio for each row
    
    """
    
    # Counts number of classes for target
    counts=pd.Series(y).value_counts()
    maj_class=counts.keys()[0]
    min_class=counts.keys()[1]
    maj_counts=counts.values[0]
    min_counts=counts.values[1]    
    
    if (ind==0 or ind==2): 
        # Number of oversamplings from minority class
        over_sample=int(maj_counts*r1-min_counts) 
        if (over_sample<0):
            raise ValueError ('Oversampling ratio (r1) is lower than the ratio in data') 
    
    # Number of undersamplings from majority class
    if (ind==1 or ind==2):
        under_sample=int(min_counts/r2)
        if (maj_counts<under_sample):
            raise ValueError ('Undersampling ratio (r2) is higher than majority class') 
        
    #print('Over: ',over_sample,', Under: ',under_sample)    
    ########################## Oversampling from minority class ################################        
        
    # Divide training sets into minority class
    x_min_class=x[np.where(y==min_class)[0]]
    
    if (ind==0 or ind==2):
        # LU Simulation for Standard Gaussian
        np.random.seed(seed)
        t_dist = scipy.stats.t(seed)
        L=scipy.linalg.cholesky(corr, lower=True, overwrite_a=True)
        mu=0
        sigma=1
        nvar=len(corr)
        w=np.zeros((nvar,over_sample)) 
        N_Sim_R_val=[]
        
        for i in range (nvar):
            for j in range(over_sample):
                Dist = np.random.normal(mu, sigma, over_sample)
                w[i,:]=Dist
        #
        N_var=[]
        for i in range(over_sample):
            tmp=(np.matmul(L,w[:,i]))
            N_var.append(tmp)       
        N_var=np.array(N_var).transpose() 
        #
        Sim_R_val=np.zeros((over_sample,nvar))
        for i1 in range(nvar):
            R_tmp=[]
            for i2 in range(over_sample):
                prob=t_dist.cdf(N_var[i1][i2])
                R_tmp=np.quantile(x_min_class[:,i1], prob, axis=0, keepdims=True)[0]
                Sim_R_val[i2,i1]=R_tmp
        
        # Concatenate simulate over samples from minority class with original date
        over_sample_f=np.concatenate((Sim_R_val,x_min_class), axis=0)
        over_sample_y=np.array(len(over_sample_f)*[min_class])

    if (ind==1 or ind==2):    
        ########################## Undersampling from majority class ################################
        
        # Divide training sets into majority class
        x_maj_class=x[np.where(y==maj_class)[0]]
        
        if (mr):
            # Sort training set based on missing ratio for each row of majority class
            idx_maj_class=np.arange(len(x_maj_class))
            ratio_maj_class=r3[np.where(y==maj_class)[0]]
            idx_sort = [x for _,x in sorted(zip(ratio_maj_class,idx_maj_class),reverse=True)]
            under_sample_f=x_maj_class[idx_sort[:under_sample]]    
        else:
            under_sample_shuflle=np.random.permutation(np.arange(under_sample))
            under_sample_f=x_maj_class[under_sample_shuflle]            
        under_sample_y=np.array(len(under_sample_f)*[maj_class])   
    
    # Final data 
    if (ind==2):
        x_=np.concatenate((over_sample_f,under_sample_f), axis=0)
        y_=np.concatenate((over_sample_y,under_sample_y), axis=0)
        idx=np.random.permutation(np.arange(len(x_)))
        x_f=x_[idx]
        y_f=y_[idx]
    elif (ind==0):
        x_maj_class=x[np.where(y==maj_class)[0]]
        y_maj_class=y[np.where(y==maj_class)[0]]
        x_=np.concatenate((x_maj_class,over_sample_f), axis=0)
        y_=np.concatenate((y_maj_class,over_sample_y), axis=0)
        idx=np.random.permutation(np.arange(len(x_)))
        x_f=x_[idx]
        y_f=y_[idx]        
    elif (ind==1):
        x_min_class=x[np.where(y==min_class)[0]]
        y_min_class=y[np.where(y==min_class)[0]]
        x_=np.concatenate((x_min_class,under_sample_f), axis=0)
        y_=np.concatenate((y_min_class,under_sample_y), axis=0)        
        
        idx=np.random.permutation(np.arange(len(x_)))
        x_f=x_[idx]
        y_f=y_[idx]                 
        
    return x_f, y_f

##########################################################################################################

def Ove_Und_During_Cross(x,y,model,ind,r1,r2,corr,seed=34,cv=3,
                         NN=False,batch_size=32,verbose=0,epochs=10):
    """
    Implement K-Fold cross validation during oversampling
    """
    from sklearn.base import clone

    # Cross-validate
    # Use for StratifiedKFold classification
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) 
    oos_pred = []
    y_real=[]
    model_=[]
    # Must specify y StratifiedKFold for
    ir=1
    for train, test in kf.split(x,y):   
        x_trn = x[train]
        y_trn = y[train]
        #
        x_tst = x[test]
        y_tst = y[test]  
        
        x_over_under,y_over_under=Over_Under_Sampling(x_trn,y_trn,ind,r1
                            ,r2,corr,seed+ir,r3=None)
    
        # Fit Model for over undersample data
        if (NN==True):
            model=DNN (init_mode='uniform', activation= 'relu',dropout_rate= False, neurons= 50 ) 
            model.fit(x_over_under,y_over_under,batch_size=batch_size,verbose=verbose,epochs=epochs) 
            model_.append(model)            
        else:
            model.random_state=seed+ir
            model=clone(model)
            model.fit(x_over_under,y_over_under)
            model_.append(model)
        
        # Predict on test fold
        pred = list(np.ravel(np.array(model.predict(x_tst), dtype='float')))
        oos_pred.append(pred)
        y_real.append(y_tst)
        ir+=1
    oos_pred=np.concatenate(oos_pred).ravel()
    y_real=np.concatenate(y_real).ravel()
    return oos_pred,y_real,model_