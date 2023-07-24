# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 07:43:53 2023

@author: richa
"""

#%%
import os
import numpy as np 
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import tkinter as tk
import json
os.chdir('C:/Users/richa/Documents/Packages/Vetta/Database')
import VettaUtils as VU

# get current sql table names
TNames = VU.AllData().getTableNames()
Subs = np.unique([x.split('_')[0] for x in TNames]).tolist()

Folder = 'C:/Users/richa/Documents/Packages/Vetta/Database'
D = VU.AllData()
D.getTableNames()


#%% Load Raw Data into SQL Database 
ReProcessRaw = 0
if ReProcessRaw == 1:
    DataFolder = 'C:/Users/richa/Documents/Packages/Vetta/Database/PilotData'
    for F in os.listdir(DataFolder):
        if F not in Subs: 
            print('Processing:' ,F)
            SubFolder = DataFolder + '/' + F
            P = VU.PilotData(SubFolder)
            AllRawData = P.getRawData()


#%% Process Raw Data and get ML estimates
os.chdir('C:/Users/richa/Documents/Packages/Vetta/Database')
D = VU.AllData()

# get waist, Lank, and Rank data from raw data
NoGCtableNames = [x for x in D.getTableNames() if 'GC' not in x]

for f in NoGCtableNames: # loop through sql tables of raw data
    print('Processing:', f)
    S = f.split('_')
    D.L, D.L_DQ = D.loadSensor(S[0], S[1], S[2], '34')
    D.R, D.R_DQ = D.loadSensor(S[0], S[1], S[2], '35')
    D.W, D.W_DQ = D.loadSensor(S[0], S[1], S[2], '36')
    
    # identify gait cycles using ankle sensors
    D.L_GC = VU.GetGCAcc(D.L)
    D.R_GC = VU.GetGCAcc(D.R)
    
    # get waist data and estimate vGRF
    D.W_LGC = VU.PredvGRF(D.W, D.L_GC, D.model)
    D.W_RGC = VU.PredvGRF(D.W, D.R_GC, D.model)
    
    # save estimates to SQL database
    C = ['Pred' + str(x) for x in range(100)]
    VU.pd_to_sql(pd.DataFrame(D.W_LGC['Preds'], columns=C), f + '_L_GC', D.sqlDB)
    VU.pd_to_sql(pd.DataFrame(D.W_RGC['Preds'], columns=C), f + '_R_GC', D.sqlDB)
    
    # get diagnostics on data quality
    Cols = ['NumSamp','RecDur','MeanSampFreq','MeanDropLen','StdDropLen','NumDrops']
    LDiag = ['Left'] + [D.L_DQ[x] for x in Cols]
    RDiag = ['Right'] + [D.R_DQ[x] for x in Cols]
    WDiag = ['Waist'] + [D.W_DQ[x] for x in Cols]
    Diag = pd.DataFrame([LDiag, RDiag, WDiag], columns=['Sensor'] + Cols)
    VU.pd_to_sql(Diag, f + '_GC_Diag', D.sqlDB)        # export diagnostics 
    
    # aggregate and export vGRF peaks
    if D.W_LGC['Num_GCs'] < 3:
        continue
    ColNames = ['StartTimes','EndTimes','Side','Peaks']   
    PksL = np.array([list(D.W_LGC['StartTimes']), 
                     list(D.W_LGC['EndTimes']), 
                    np.repeat('Left', D.W_LGC['Num_GCs']).tolist(), 
                    list(D.W_LGC['Peaks'])]).T
    OutPksL = pd.DataFrame(data=PksL, columns=ColNames)
    PksR = np.array([list(D.W_RGC['StartTimes']), 
                     list(D.W_RGC['EndTimes']), 
                    np.repeat('Right', D.W_RGC['Num_GCs']).tolist(), 
                    list(D.W_RGC['Peaks'])]).T
    OutPksR = pd.DataFrame(data=PksR, columns=ColNames)
    OutPks = pd.concat([OutPksL, OutPksR])
    VU.pd_to_sql(OutPks, f + '_GC_Peaks', D.sqlDB)      # export peaks

    # export vGRF peaks to CSV?
    ToCSV = 0
    if ToCSV == 1:
        OutPks.to_csv(f + '_GC_Peaks.csv')

        

#%% Incorporate TM data and compare with estimates

# create dataframe to save trial info
cols = ['Subj','TrialName','Synced','ExclRsn']
if 'TrialInfo.csv' in os.listdir(Folder):
    TrialInfo = pd.read_csv('TrialInfo.csv')
else:
    TrialInfo = pd.DataFrame(columns=cols)

for s in Subs:
    
    if 's014' not in s:
        continue
    
    SubjPath = os.path.join(os.getcwd(), 'PilotData', s)
    if 'Figures' not in os.listdir(SubjPath):
        os.mkdir(os.path.join(SubjPath, 'Figures'))

    for fn in os.listdir(SubjPath):
        if '.mat' in fn:
            
            if 'static' in fn:      # skip static trial
                continue
                
            print('Processing:  ', fn)
            SQLname = D.getSQLfromMatFile(fn) # match TM trial with SQL table
            print(SQLname)
            if not SQLname:
                continue
            # check to see if TM data already in SQL database
            # skip if already present
            
            # load associate sensor data from SQL db
            S = SQLname.split('_')
            L, L_unF, L_DQ = D.loadSensor(S[0], S[1], S[2], '34')
            R, R_unF, R_DQ = D.loadSensor(S[0], S[1], S[2], '35')
            W, W_unF, W_DQ = D.loadSensor(S[0], S[1], S[2], '36')

            # load TM data
            TM = D.loadTMdata(fn)
            
            # plot data for quality assurance
            a = 0
            b = 30
            plt.close('all')
            plt.figure(figsize=(12, 8))
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212)
            
            ax1.plot(TM['Time'], TM['GRFv_Rn'], label='Right', color='b')
            ax1.plot(TM['Time'], TM['GRFv_Ln'], label='Left', color='g')
            ax1.set_title('Full vGRFs ' + fn)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('vGRF (BWs)')
            ax1.set_xlim((a,b))
            ax1.legend()
            
            ax2.plot(R.index, R_unF.y, label='RSensor', color='b')
            ax2.plot(L.index, L_unF.y, label='LSensor', color='g')
            ax2.plot(W.index, W.y, label='WSensor', color='r')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Acc (g)')
            ax2.set_xlim((a,b))
            ax2.legend()
            plt.show()
            
            
            # align signals via stomp
            
            # locate stomps in plot
            t = 'Stomp ID'
            m = 'Can you fully view the vGRF (top) and acc(bottom) stomps?'
            ans = tk.messagebox.askyesno(title=t, message=m, icon='question')
            
            if ans == False:
                b = 60
                ax1.set_xlim((a,b))
                ax2.set_xlim((a,b))
                plt.show()
                m = 'Now can you fully view the vGRF (top) and acc(bottom) stomps?'
                ans2 = tk.messagebox.askyesno(title=t, message=m, icon='question')
                if ans2 == False:
                    # save trial info
                    TrialInfo.loc[len(TrialInfo)] = [s, fn, 'No', 'No stomp view']
                    continue
                
            
            print('Select vGRF stomp start & stop (top subplot)')
            selvGRF1, selvGRF2 = plt.ginput(2)
            ax1.vlines([selvGRF1[0], selvGRF2[0]],ax1.get_ylim()[0], ax1.get_ylim()[1], 'k')
            
            print('Select acc stomp start & stop (bottom subplot)')
            selAcc1, selAcc2 = plt.ginput(2)
            ax2.vlines([selAcc1[0], selAcc2[0]],ax2.get_ylim()[0], ax2.get_ylim()[1], 'k')
            
            # get max vGRF
            GRFind1 = np.where(TM['Time'] == round(selvGRF1[0], 2))[0][0]
            GRFind2 = np.where(TM['Time'] == round(selvGRF2[0], 2))[0][0]
            RvGRFindMax = np.argmax(TM['GRFv_Rn'][GRFind1:GRFind2])
            LvGRFindMax = np.argmax(TM['GRFv_Ln'][GRFind1:GRFind2])
            
            RvGRFind = GRFind1 + RvGRFindMax
            ax1.plot(TM['Time'][RvGRFind], TM['GRFv_Rn'][RvGRFind], 'ob')
            LvGRFind = GRFind1 + LvGRFindMax
            ax1.plot(TM['Time'][LvGRFind], TM['GRFv_Ln'][LvGRFind], 'og')
            
            # get max acc
            GRFind1 = np.where(L.index == round(selAcc1[0], 2))[0][0]
            GRFind2 = np.where(L.index == round(selAcc2[0], 2))[0][0]
            RAccindMax = np.argmax(R_unF.y.tolist()[GRFind1:GRFind2])
            LAccindMax = np.argmax(L_unF.y.tolist()[GRFind1:GRFind2])
            
            RAccind = GRFind1 + RAccindMax
            ax2.plot(R.index[RAccind], R_unF.y.tolist()[RAccind], 'ob')
            LAccind = GRFind1 + LAccindMax
            ax2.plot(L.index[LAccind], L_unF.y.tolist()[LAccind], 'og')
            
            # visually check stomp alignment
            pad = 1
            ax1.set_xlim((selvGRF1[0]-pad, selvGRF2[0]+pad))
            ax2.set_xlim((selAcc1[0]-pad, selAcc2[0]+pad))
            plt.show()
            
            ans = tk.messagebox.askyesno('Correct Stomps', 
                                         'Are the stomps identified correctly?')
            
            if ans == False:
                m = 'Re-select vGRF stomp start & stop (top subplot)'
                tk.messagebox.showinfo(t, m)
                selvGRF1, selvGRF2 = plt.ginput(2)
                ax1.vlines([selvGRF1[0], selvGRF2[0]],ax1.get_ylim()[0], ax1.get_ylim()[1], 'k')
                
                m = 'Re-select acc stomp start & stop (bottom subplot)'
                tk.messagebox.showinfo(t, m)
                selAcc1, selAcc2 = plt.ginput(2)
                ax2.vlines([selAcc1[0], selAcc2[0]],ax2.get_ylim()[0], ax2.get_ylim()[1], 'k')
                
                # get max vGRF
                GRFind1 = np.where(TM['Time'] == round(selvGRF1[0], 2))[0][0]
                GRFind2 = np.where(TM['Time'] == round(selvGRF2[0], 2))[0][0]
                RvGRFindMax = np.argmax(TM['GRFv_Rn'][GRFind1:GRFind2])
                LvGRFindMax = np.argmax(TM['GRFv_Ln'][GRFind1:GRFind2])
                
                RvGRFind = GRFind1 + RvGRFindMax
                ax1.plot(TM['Time'][RvGRFind], TM['GRFv_Rn'][RvGRFind], 'ob')
                LvGRFind = GRFind1 + LvGRFindMax
                ax1.plot(TM['Time'][LvGRFind], TM['GRFv_Ln'][LvGRFind], 'og')
                
                # get max acc
                GRFind1 = np.where(L.index == round(selAcc1[0], 2))[0][0]
                GRFind2 = np.where(L.index == round(selAcc2[0], 2))[0][0]
                RAccindMax = np.argmax(R_unF.y.tolist()[GRFind1:GRFind2])
                LAccindMax = np.argmax(L_unF.y.tolist()[GRFind1:GRFind2])
                
                RAccind = GRFind1 + RAccindMax
                ax2.plot(R.index[RAccind], R_unF.y.tolist()[RAccind], 'ob')
                LAccind = GRFind1 + LAccindMax
                ax2.plot(L.index[LAccind], L_unF.y.tolist()[LAccind], 'og')
                plt.show()
                
                ans2 = tk.messagebox.askyesno('Correct Stomps', 
                                             'Are the stomps identified correctly?')
                
                if ans2 == False:
                    m = 'Skipping this trial...'
                    tk.messagebox.showinfo(t, m)
                    
                    # save trial info
                    TrialInfo.loc[len(TrialInfo)] = [s, fn, 'No','No stomp id']
                    continue
                
            # save figure of stomp ID
            plt.savefig(os.path.join(SubjPath, 'Figures', SQLname + '_StompID.png'))
            
            # raise StopIteration
            
            
            # determine if sensors are on the correct side 
            if RvGRFind < LvGRFind:
                FirstvGRFStomp = 'R'
            else:
                FirstvGRFStomp = 'L'
                
            if RAccind < LAccind:
                FirstAccStomp = 'R'
            else:
                FirstAccStomp = 'L'
            
            if FirstvGRFStomp == FirstAccStomp:
                print('Sensors are on the correct feet')
                SensorSwap = 'No'
            else:
                print('Sensors are swapped')
                SensorSwap = 'Yes'
                # swap sensors
                newL, newL_unF, newL_DQ = R, R_unF, R_DQ
                newR, newR_unF, newR_DQ = L, L_unF, L_DQ         
                R, R_unF, R_DQ = newR, newR_unF, newR_DQ
                L, L_unF, L_DQ = newL, newL_unF, newL_DQ
                RAccind, LAccind = LAccind, RAccind

            
            # save sensor orientation in demo file
            DemoFile = open(os.path.join(SubjPath, 'Demo.json'), 'r')
            JSdemo = json.load(DemoFile)
            DemoFile.close()
            
            JSdemo['SensorSwap'] = SensorSwap
            DemoFile = open(os.path.join(SubjPath, 'Demo.json'), 'w')
            json.dump(JSdemo, DemoFile, indent=5)
            DemoFile.close()
            
            # synchronize vGRFs and Accs
            Offset = 3000   # trial start offset
            EndTrim = 1000 # trim end of trials to account for TM slowing down
            
            # right
            RvGRFsync = TM['GRFv_Rn'][RvGRFind + Offset :]
            RTimeSync = TM['Time'].tolist()[RvGRFind + Offset :]
            RAccsync = R.y.tolist()[RAccind + Offset :]
            RAccUFsync = R_unF.y.tolist()[RAccind + Offset :]
            if len(RvGRFsync) > len(RAccsync):
                RvGRFsync = RvGRFsync[:len(RAccsync) - EndTrim]
                RAccsync = RAccsync[:len(RAccsync) - EndTrim]
                RAccUFsync = RAccUFsync[:len(RAccsync)]
                RTimeSync = RTimeSync[:len(RAccsync)]
            elif len(RvGRFsync) < len(RAccsync):
                RvGRFsync = RvGRFsync[:len(RvGRFsync) - EndTrim]
                RAccsync = RAccsync[:len(RvGRFsync)]
                RAccUFsync = RAccUFsync[:len(RvGRFsync)]
                RTimeSync = RTimeSync[:len(RvGRFsync)]
                
            # left
            LvGRFsync = TM['GRFv_Ln'][LvGRFind + Offset :]
            LTimeSync = TM['Time'].tolist()[LvGRFind + Offset :]
            LAccsync = L.y.tolist()[LAccind + Offset :]
            LAccUFsync = L_unF.y.tolist()[LAccind + Offset :]
            if len(LvGRFsync) > len(LAccsync):
                LvGRFsync = LvGRFsync[:len(LAccsync) - EndTrim]
                LAccsync = LAccsync[:len(LAccsync) - EndTrim]
                LAccUFsync = LAccUFsync[:len(LAccsync)]
                LTimeSync = LTimeSync[:len(LAccsync)]
            elif len(LvGRFsync) < len(LAccsync):
                LvGRFsync = LvGRFsync[:len(LvGRFsync) - EndTrim]
                LAccsync = LAccsync[:len(LvGRFsync)]
                LAccUFsync = LAccUFsync[:len(LvGRFsync)]
                LTimeSync = LTimeSync[:-EndTrim]

            
            # check synchronization and extract steps
            # RIGHT
            RvGRF_stance, RvGRF_GC = VU.GetGCs(RvGRFsync)
            ToDel = [x for x in RvGRF_GC['Start'] if x > len(RTimeSync)]
            while len(ToDel) > 0:
                ind = RvGRF_GC['Start'].index(ToDel[0])
                RvGRF_GC['Start'].pop(ind)
                ToDel.pop(0)
            R_GCs = [RTimeSync[x] for x in RvGRF_GC['Start']]
            RAcc = pd.DataFrame(dict(y=RAccsync), index=RTimeSync)
            RAccUF = pd.DataFrame(dict(y=RAccUFsync), index=RTimeSync)
            RAcc_GC = VU.GetGCAcc(RAcc, RAccUF) # identify gait cycles using ankle sensors
            
            # plot synchronized vGRF and Acc
            # plt.close('all')
            plt.figure(figsize=(8, 6))
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212)
            
            ax1.plot(RTimeSync, RvGRFsync)
            ax1.vlines(R_GCs, ax1.get_ylim()[0], ax1.get_ylim()[1], 'k')
            ax1.set_xlim((RTimeSync[0], RTimeSync[500]))
            ax1.set_title('Right vGRF')
            
            ax2.plot(RTimeSync, RAccsync, label='Filtered')
            ax2.plot(RTimeSync, RAccUFsync, label='Unfiltered')
            ax2.plot(RTimeSync[:-1], np.diff(RAccUFsync), label='jerk')
            ax2.vlines(R_GCs, ax2.get_ylim()[0], ax2.get_ylim()[1], 'k')
            ax2.vlines(RAcc_GC['StartTimes'], ax2.get_ylim()[0], ax2.get_ylim()[1], 'b')
            ax2.legend()
            ax2.set_xlim((RTimeSync[0], RTimeSync[500]))
            ax2.set_title('Right Acc')
            plt.savefig(os.path.join(SubjPath, 'Figures', SQLname + '_Rsync.png')) 
            
            # LEFT
            LvGRF_stance, LvGRF_GC = VU.GetGCs(LvGRFsync)
            ToDel = [x for x in LvGRF_GC['Start'] if x > len(LTimeSync)]
            while len(ToDel) > 0:
                ind = LvGRF_GC['Start'].index(ToDel[0])
                LvGRF_GC['Start'].pop(ind)
                ToDel.pop(0)
            L_GCs = [LTimeSync[x] for x in LvGRF_GC['Start']]
            LAcc = pd.DataFrame(dict(y=LAccsync), index=LTimeSync)
            LAccUF = pd.DataFrame(dict(y=LAccUFsync), index=LTimeSync)
            LAcc_GC = VU.GetGCAcc(LAcc, LAccUF) # identify gait cycles using ankle sensors
            
            # plot synchronized vGRF and Acc
            plt.figure(figsize=(8, 6))
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212)
            
            ax1.plot(LTimeSync, LvGRFsync)
            ax1.vlines(L_GCs, ax1.get_ylim()[0], ax1.get_ylim()[1], 'k')
            ax1.set_xlim((LTimeSync[0], LTimeSync[500]))
            ax1.set_title('Left vGRF')
            
            ax2.plot(LTimeSync, LAccsync, label='Filtered')
            ax2.plot(LTimeSync, LAccUFsync, label='Unfiltered')
            ax2.plot(LTimeSync[:-1], np.diff(LAccUFsync), label='jerk')
            ax2.vlines(L_GCs, ax2.get_ylim()[0], ax2.get_ylim()[1], 'k')
            ax2.vlines(LAcc_GC['StartTimes'], ax2.get_ylim()[0], ax2.get_ylim()[1], 'b')
            ax2.legend()
            ax2.set_xlim((LTimeSync[0], LTimeSync[500]))
            ax2.set_title('Left Acc')
            plt.savefig(os.path.join(SubjPath, 'Figures', SQLname + '_Lsync.png')) 
            
            t = 'Successful Sync?'
            m = 'Are the vGRFs and Accs synchronized correctly?'
            ans = tk.messagebox.askyesno(t, m, icon='question')
            if ans == False:
                # save trial info
                TrialInfo.loc[len(TrialInfo)] = [s, fn, 'No','No sync']
                continue
            
            
            # get all steps present in both vGRFs and Acc
            Thresh = 0.05
            vGRFstarts = []
            Accstarts = []
            for g in LAcc_GC['StartTimes']:
                # get step within threshold
                t = [i for i, x in enumerate(np.subtract(L_GCs, g)) if abs(x) < Thresh]
                if t:
                    # print('vGRF start time:  ', L_GCs[t[0]])
                    # print('Acc start time:  ', g)
                    vGRFstarts.append(L_GCs[t[0]])
                    Accstarts.append(g)
            
            # compile all steps into DF
            
            # resample to 100 points
            
            # save in SQL db
            
            
            # save trial info
            TrialInfo.loc[len(TrialInfo)] = [s, fn,'Yes','N/A']
                
            
    # raise StopIteration
    
TrialInfo.to_csv('TrialInfo.csv')


#%% Plot Estimates SQL data
# D.plotPreds()




        



