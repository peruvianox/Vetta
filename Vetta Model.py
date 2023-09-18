# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:04:55 2023

@author: richa
"""


#%%
import os
import numpy as np 
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
import tkinter as tk
import seaborn as sb
import json
os.chdir('C:/Users/richa/Documents/Packages/Vetta/Database')
import VettaUtils as VU

# get current sql table names
TNames = VU.AllData().getTableNames()
Subs = np.unique([x.split('_')[0] for x in TNames]).tolist()

Folder = 'C:/Users/richa/Documents/Packages/Vetta/Database'
D = VU.AllData()


#%% load/create dataframe with all vgrf and waist data

# get current sql table names
TNames = D.getTableNames()
Subs = np.unique([x.split('_')[0] for x in TNames]).tolist()
GRFnames = [x for x in TNames if 'GCsyncTM' in x]
Accnames = [x for x in TNames if 'GCsyncAcc' in x]
TotalTrials = 20 * 15 * 2
# print('Found:', str(len(GRFnames)), 'GRF trials out of', str(TotalTrials), 'Total Trials')
# print('Found:', str(len(Accnames)), 'Acc trials out of', str(TotalTrials), 'Total Trials')
 
CreateDF = 1
if CreateDF == 1:
    # create dataframe
    meta = ['Subj','Speed','Load','Side','Step']
    vGRFcols = ['VGRF' + str(x) for x in range(100)]
    wAcccols = ['wAcc' + str(x) for x in range(100)]
    StepTable = pd.DataFrame()
    
    MetaDF = pd.DataFrame(columns=['Subj','Speed','Load','Side','Nsteps'])
    
    for v in GRFnames: 
        print('Processing:', v)
        vGRFsteps = VU.sql_to_pd(f"SELECT * FROM {v}", D.sqlDB) # get vGRF data
        
        # get meta data from table name
        Full = v.split('_')
        subj = Full[0]
        speed = Full[1]
        if speed == '000':
            speed = 'pref'
        load = Full[2]
        if 'pref' in load:
            load = 'baseline'
        
        if 'LGC' in v:
            side = 'L'
        else:
            side = 'R'
        
        # get matching Acc data
        AccName = [x for x in Accnames if x[:-3] in v][0]
        wAccsteps = VU.sql_to_pd(f"SELECT * FROM {AccName}", D.sqlDB)
        # plot to check values
        # ax1 = plt.subplot(211)
        # ax2 = plt.subplot(212)
        # ax1.plot(vGRFsteps.T)
        # ax2.plot(wAccsteps.T)
        
        if len(vGRFsteps) != len(wAccsteps):
            print('Steps not equal length', v)
            
            
        # save step counts to Meta dataframe
        MetaDF.loc[len(MetaDF)] = [subj, speed, load, side, len(vGRFsteps)]
        
        for i in range(len(vGRFsteps)):
        
            # save full waveforms to step table
            StepTable.loc[len(StepTable)] = [subj, speed, load, side, i] + vGRFsteps.iloc[i].tolist() + wAccsteps.iloc[i].tolist()
        
        
    MetaDF = MetaDF.sort_values(by=['Subj', 'Speed'])
    MetaDF.to_csv('StepMetaData.csv')
    
    StepTable.columns = [meta + vGRFcols + wAcccols][0]
    StepTable = StepTable.sort_values(by=['Subj', 'Speed', 'Step'])
    StepTable.to_csv('StepData.csv')


# otherwise load saved sheets
MetaDF = pd.read_csv('StepMetaData.csv')
MetaDF.drop([x for x in MetaDF.columns if 'Unnamed' in x], axis=1)
StepTable = pd.read_csv('StepData.csv')    
StepTable.drop([x for x in StepTable.columns if 'Unnamed' in x], axis=1)

#%% Plot Step Counts
plt.close('all')
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10,10))

ax1 = axs[0,0]
ax2 = axs[0,1]
gs = axs[1,0].get_gridspec()
ax3 = axs[1,0]
ax4 = axs[1,1]
ax3.remove()
ax4.remove()
ax3 = fig.add_subplot(gs[1,:])
sb.boxplot(MetaDF, x='Speed', y='Nsteps', ax=ax1, 
           order=['080','preferred','160'])
sb.boxplot(MetaDF, x='Load', y='Nsteps', ax=ax2, 
           order=['under3','baseline','over3','over6','over9'])
sb.boxplot(MetaDF, x='Subj', y='Nsteps', ax=ax3)
ax1.set_ylim([0, 130])
ax2.set_ylim([0, 130])
ax3.set_ylim([0, 130])

plt.savefig('Figures/StepCounts.png')

#%% Plot new vGRFs and waist Accs

plt.close('all')
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(10,10))

ax1 = axs[0]
ax2 = axs[1]

vGRFcols = [i for i, x in enumerate(list(StepTable.columns)) if 'VGRF' in x]
wAcccols = [i for i, x in enumerate(list(StepTable.columns)) if 'wAcc' in x]
cmap = mpl.colormaps['plasma']
A = 0.1

for i in range(len(StepTable)):
    
    subj = StepTable['Subj'].iloc[i]
    subjInd = int(subj[1:])
    
    vGRF = StepTable.iloc[i, vGRFcols]
    wAcc = StepTable.iloc[i, wAcccols]
    
    ax1.plot(vGRF, color=cmap(12*subjInd), alpha=A)


    ax2.plot(wAcc, color=cmap(12*subjInd), alpha=A)
    
    
ax1.set_xticklabels([x for x in range(100)])
ax1.set_xlim([0, 99])
ax1.set_xticks([0, 20, 40, 60, 80, 99])
ax1.set_xticklabels(['0', '20', '40', '60', '80', '100'])
ax1.set_title('Measured vGRF')
ax1.set_ylabel('BW')

ax2.set_xticklabels([x for x in range(100)])
ax2.set_xlim([0, 99])
ax2.set_xticks([0, 20, 40, 60, 80, 99])
ax2.set_xticklabels(['0', '20', '40', '60', '80', '100'])
ax2.set_title('Waist Acc')
ax2.set_xlabel('% GC')
ax2.set_ylabel('g')


# sb.boxplot(MetaDF, x='Speed', y='Nsteps', ax=ax1, 
#            order=['080','preferred','160'])
# sb.boxplot(MetaDF, x='Load', y='Nsteps', ax=ax2, 
#            order=['under3','baseline','over3','over6','over9'])
# sb.boxplot(MetaDF, x='Subj', y='Nsteps', ax=ax3)
# ax1.set_ylim([0, 130])
# ax2.set_ylim([0, 130])
# ax3.set_ylim([0, 130])

plt.savefig('Figures/Preds.png')


#%% Plot all measured vGRFs by loading profile 
plt.close('all')

vGRFcols = [i for i, x in enumerate(list(StepTable.columns)) if 'VGRF' in x]
wAcccols = [i for i, x in enumerate(list(StepTable.columns)) if 'wAcc' in x]
cmap = mpl.colormaps['plasma']
A = 0.2
LW = 3
Loads = ['under3', 'over3', 'over6', 'over9', 'baseline']
Colors = ['b','r','r','r','k']
Alphas = [1, 0.33, 0.66, 1, 1]
Speeds = ['160', '80', 'preferred']

for S in Speeds: 
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(10, 6))
    ax1 = axs[0]
    ax2 = axs[1]
    for i, L in enumerate(Loads): 
        SpeedInds = [i for i, x in enumerate(StepTable['Speed']) if S in x]
        LoadInds = [i for i, x in enumerate(StepTable['Load']) if L in x]
        Match = list(set(SpeedInds) & set(LoadInds))
        
        All = StepTable.iloc[Match, vGRFcols].to_numpy()
        Avg = np.mean(All, axis=0)
        Std = np.std(All, axis=0)
        ax1.plot(Avg, color=Colors[i], lw=LW, alpha=Alphas[i], label=L)
    # ax1.fill_between(range(100), Avg-Std, Avg+Std, color='k', alpha=A)
    
        
        AllW = StepTable.iloc[Match, wAcccols].to_numpy()
        AvgW = np.mean(AllW, axis=0)
        StdW = np.std(AllW, axis=0) 
        ax2.plot(AvgW, color=Colors[i], lw=LW, alpha=Alphas[i])
    
    ax1.legend()
    ax1.set_xticklabels([x for x in range(100)])
    ax1.set_xlim([0, 99])
    ax1.set_xticks([0, 20, 40, 60, 80, 99])
    ax1.set_xticklabels(['0', '20', '40', '60', '80', '100'])
    ax1.set_title('Measured vGRF, Speed: ' + S)
    ax1.set_ylabel('BW')
    
    ax2.set_xticklabels([x for x in range(100)])
    ax2.set_xlim([0, 99])
    ax2.set_xticks([0, 20, 40, 60, 80, 99])
    ax2.set_xticklabels(['0', '20', '40', '60', '80', '100'])
    ax2.set_title('Waist Acc, Speed: ' + S)
    ax2.set_xlabel('% GC')
    ax2.set_ylabel('g')
    
    plt.savefig('Figures/Averages' + S + 'Spd.png')

    