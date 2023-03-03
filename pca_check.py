# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 09:48:58 2023

@author: a_deswal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
#from mpl_toolkits import mplot3d
from datetime import datetime as dt
#from datetime import timedelta as td
from xbbg import blp
import re
import streamlit as st

curves = pd.read_excel('C:\\BLP\\data\\pca.xlsx', index_col = 0)

class curve():
    
    def __init__(self, name,spot = False):
        self.name = name.lower()
        self.kind = 'spot' if spot else 'forward'    
        c = curves.loc[name.lower()+'_spot' if spot else name.lower()].dropna().values
        start = dt(year = dt.today().year - 3, month = dt.today().month, day = dt.today().day ).strftime('%Y-%m-%d')
        df = blp.bdh(tickers=c,start_date = start, end_date ='today')
        
        if spot:
            t = [i+'Y' for j in [re.findall(r'\d+',i) for i in c] for i in j]
        elif self.name in ['euribor', 'stibor']:
            t = ['1Y']+[str(int(i[:2]))+'Y'+str(int(i[2:]))+'Y' for j in [re.findall(r'\d+',i) for i in c[1:]] for i in j]
        else:
            t = ['1Y']+[i.split()[1] for i in c[1:]]
        
        df.columns = t
        
        self.data = df
        
        
        
    def set_lookback(self,days):
        
        self.lookback = self.data.loc[self.data.index[-(days+1):]].copy(deep = True)
        
        dm = pd.DataFrame({})
        for i in self.lookback.columns:
            dm[i] = self.lookback[i] - self.lookback[i].mean()
        
        self.demeaned = dm 
        
    
    def run_PCA(self, retained_components = 3):
        
        self.eigval,self.eigvec = np.linalg.eig(self.demeaned.cov().values)
        self.explained_prop = pd.DataFrame({'Eigenvalues': self.eigval, 'Explained Proportion': self.eigval/np.sum(self.eigval)})
        self.loadings = pd.DataFrame(self.eigvec.T[:retained_components], index = ['PC'+str(i+1) for i in range(retained_components)], columns= self.data.columns)
        self.components =  self.lookback.dot(self.loadings.values.T)
        self.components.columns = self.loadings.index
        self.projection = self.loadings.transpose().dot(self.loadings)
        
        self.recon  =  pd.DataFrame({})
        for i in self.data.columns:
            self.recon[i] = self.demeaned.dot(self.projection)[i] + self.lookback[i].mean()
        
        
        self.residuals = self.demeaned  - self.demeaned.dot(self.projection)
        #return self.explained_prop.head(5).style.format({"Explained Proportion": "{:.2%}"})

    
    def fix_loadings(self,*pcs):
        
        for i in pcs:
            self.loadings.loc['PC'+str(i)] *= -1
         
    def plot_loadings(self):
        if self.loadings.shape[0] == 2:
            fig,(ax1,ax2) = plt.subplots(nrows = 2, ncols = 1)

            plt.subplots_adjust(left=0, right=2)
            plt.subplots_adjust(bottom=0, top=4)

            ax1.bar(self.loadings.columns, self.loadings.loc['PC1'], color = 'b', edgecolor = 'k')
            ax1.set_title('LOADINGS FOR FIRST PRINCIPAL COMPONENT')
            ax1.set_xlabel('Tenor')
            ax1.set_ylabel('Loading')

            ax2.bar(self.loadings.columns, self.loadings.loc['PC2'], color = 'g', edgecolor = 'k')
            ax2.set_title('LOADINGS FOR SECOND PRINCIPAL COMPONENT')
            ax2.set_xlabel('Tenor')
            ax2.set_ylabel('Loading')

        
        else:
            fig,(ax1,ax2,ax3) = plt.subplots(nrows = 3, ncols = 1)

            plt.subplots_adjust(left=0, right=2)
            plt.subplots_adjust(bottom=0, top=4)

            ax1.bar(self.loadings.columns, self.loadings.loc['PC1'], color = 'b', edgecolor = 'k')
            ax1.set_title('LOADINGS FOR FIRST PRINCIPAL COMPONENT')
            ax1.set_xlabel('Tenor')
            ax1.set_ylabel('Loading')

            ax2.bar(self.loadings.columns, self.loadings.loc['PC2'], color = 'g', edgecolor = 'k')
            ax2.set_title('LOADINGS FOR SECOND PRINCIPAL COMPONENT')
            ax2.set_xlabel('Tenor')
            ax2.set_ylabel('Loading')

            ax3.bar(self.loadings.columns, self.loadings.loc['PC3'], color = 'r', edgecolor = 'k')
            ax3.set_title('LOADINGS FOR THIRD PRINCIPAL COMPONENT')
            ax3.set_xlabel('Tenor')
            ax3.set_ylabel('Loading')

            plt.show()

    def plot_components(self):
        
        if self.loadings.shape[0] == 2:    
        
            fig,(ax1,ax2) = plt.subplots(nrows = 2, ncols = 1)

            plt.subplots_adjust(left=0, right=2)
            plt.subplots_adjust(bottom=0, top=4)

            ax1.plot(self.components['PC1'], color = 'b')
            ax1.set_title('FIRST PRINCIPAL COMPONENT')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Level')

            ax2.plot(self.components['PC2'], color = 'g')
            ax2.set_title('SECOND PRINCIPAL COMPONENT')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Level')

            plt.show()
        
        else:
            fig,(ax1,ax2,ax3) = plt.subplots(nrows = 3, ncols = 1)
    
            plt.subplots_adjust(left=0, right=2)
            plt.subplots_adjust(bottom=0, top=4)

            ax1.plot(self.components['PC1'], color = 'b')
            ax1.set_title('FIRST PRINCIPAL COMPONENT')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Level')

            ax2.plot(self.components['PC2'], color = 'g')
            ax2.set_title('SECOND PRINCIPAL COMPONENT')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Level')

            ax3.plot(self.components['PC3'], color = 'r')
            ax3.set_title('THIRD PRINCIPAL COMPONENT')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Level')

            plt.show()
    

    def plot_residuals(self):
        f = self.residuals.loc[self.residuals.index[-1]]
        title = f'{self.name} {self.kind} residuals as of {self.residuals.index[-1].strftime("%d %B, %Y")}'
        fig,ax1 = plt.subplots(nrows = 1, ncols = 1)

        plt.subplots_adjust(left=0, right=2)
        plt.subplots_adjust(bottom=0, top=1)

        ax1.bar(self.residuals.columns,f, color = 'blue', edgecolor = 'k')
        ax1.set_title(title.upper())
        ax1.set_xlabel('Tenor')
        ax1.set_ylabel('Residual')


        plt.show()
              
underlying = st.selectbox('Underlying: ',['ESTR','SOFR', 'SONIA'])
fwd_curve = curve(underlying, spot = False)
fwd_curve.set_lookback(250)
fwd_curve.run_PCA(3)
res = fwd_curve.residuals.loc[fwd_curve.residuals.index[-1]]
st.title(f'{underlying} forward residuals as of {fwd_curve.residuals.index[-1].strftime("%d %B, %Y")}')
st.bar_chart(res.values)        
