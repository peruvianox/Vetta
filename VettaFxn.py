#!/usr/bin/env python
# coding: utf-8

# Vetta Functions


#%% Gait cycle ID functions

def GetGCs(D):
    # get gait cycles from vGRF signal
    
    from scipy import signal
    import numpy as np
    
    # peak-finding parameters
    wid = 15
    p = 0.8
    ht = 0.95
    rh = 0.98
    d = 50
    # find peaks, widths, and bases
    Pks, PkVals = signal.find_peaks(D, height=ht, prominence=p, width=wid, distance=d)
    # results = signal.peak_widths(D, Pks, rel_height=rh)
    widths, width_heights, l_ips, r_ips = signal.peak_widths(D, Pks, rel_height=rh)
    
    ## plot curve, peaks, starts & ends
    # plt.plot(D)
    # plt.plot(Pks, D[Pks], 'om')
    # plt.hlines(*results[1:], 'k')
    # plt.plot(l_ips, width_heights, 'og')
    # plt.plot(r_ips, width_heights, 'or')
    
    Rising = [round(x) for x in l_ips]
    Falling = [round(x) for x in r_ips]
    Starts = []
    Ends = []
    
    # Get Stance Phases (heel strike to toe off)
    # loop through rise cycles and interpolate & save
    L = min(len(Rising), len(Falling))
    All = np.zeros([L, 100])
    a = 0
    for i in range(L):
        Raw = D[Rising[i]:Falling[i]]
        RawX = range(len(Raw))
        X = np.linspace(0, len(Raw), 100)
        
        if len(Raw) < 10: # exclude step if too short
          continue
    
        A = np.interp(X, RawX, Raw)
        
        # exclude invalid steps if: 
        if np.mean(A[10:20]) < 0.5 or np.mean(A[20:30]) < 0.5: # too low during loading peak
          continue
        if np.mean(A[30:40]) < 0.5 or np.mean(A[45:55]) < 0.5: # too low during mid stance
          continue
        if np.mean(A[65:75]) < 0.5 or np.mean(A[75:85]) < 0.5: # too low during propulsive peak
          continue
    
        All[i, :] = A # otherwise save step
        Starts.append(Rising[i])
        Ends.append(Falling[i])
        a += 1
        del Raw, RawX, X, A
    
    if a < L:
      All = All[0:a, :]
    Avg = np.mean(100*All, axis=0)
    Std = np.std(100*All, axis=0)
    Stance = {} # save data in dict for export
    Stance['All'] = All
    Stance['Avg'] = Avg
    Stance['Std'] = Std
    Stance['Start'] = Starts
    Stance['End'] = Ends
    Stance['Num_GCs'] = len(Starts)
     
    
    # Get Gait Cycles (heel strike to heel strike)
    Starts = []
    Ends = []
    NumGCs = len(Rising)-1 # set number of gait cycles
    All = np.zeros([NumGCs, 100])
    for i in range(NumGCs):
        Raw = D[Rising[i]:Rising[i+1]]
        RawX = range(len(Raw))
        X = np.linspace(0, len(Raw), 100)
        if len(Raw) < 10: # exclude step if too short
          continue
    
        A = np.interp(X, RawX, Raw)
       # exclude invalid steps if: 
        if np.mean(A[10:20]) < 0.5 or np.mean(A[20:30]) < 0.5: # too low during loading peak
          continue
        if np.mean(A[30:40]) < 0.5 or np.mean(A[45:55]) < 0.5: # too low during propulsive peak
          continue
        if np.mean(A[75:85]) > 0.25 or np.mean(A[85:95]) > 0.25: # too high during swing
          continue
    
        All[i, :] = A # otherwise save step
        Starts.append(Rising[i])
        Ends.append(Rising[i+1])
        del Raw, RawX, X, A
    
    Avg = np.mean(100*All, axis=0)
    Std = np.std(100*All, axis=0)
    GC = {} # save data in dict for export
    GC['All'] = All
    GC['Avg'] = Avg
    GC['Std'] = Std
    GC['Start'] = Starts
    GC['End'] = Ends
    GC['Num_GCs'] = NumGCs
    
    return Stance, GC


# get predicted gait cycles from cutting the same points in the measured signal
def GetPGCs(GCs, P):

    # uses time points from GCs (see GetGCs function) to make the same time cuts in predicted gait cycles (P)
    
    # from scipy import signal
    import numpy as np
    
    # Get Stance Phases
    # loop through rise cycles and interpolate & save
    All = np.zeros(np.shape(GCs['All']))
    for j in range(len(GCs['Start'])):
      a = GCs['Start'][j]
      b = GCs['End'][j]
      Raw = P[a:b]
      if len(Raw) < 10:
          continue
      RawX = range(len(Raw))
      X = np.linspace(0, len(Raw), 100)
      All[j,:] = np.interp(X, RawX, Raw)
      del Raw, RawX, X
    
    Avg = np.mean(100*All, axis=0)
    Std = np.std(100*All, axis=0)
    Stance = {} # save data in dict for export
    Stance['All'] = All
    Stance['Avg'] = Avg
    Stance['Std'] = Std
    Stance['Start'] = GCs['Start']
    Stance['End'] = GCs['End']
    Stance['Num_GCs'] = len(GCs['Start']) - 1
 
    # Get Gait Cycles
    Rising = GCs['Start']
    # loop through rise-fall cycles and interpolate & save
    All = np.zeros([len(Rising)-1, 100])
    NumGCs = len(Rising)-1
    for i in range(NumGCs):
        a = GCs['Start'][j]
        b = GCs['End'][j]
        Raw = P[a:b]
        RawX = range(len(Raw))
        if len(Raw) < 10:
          continue
        X = np.linspace(0, len(Raw), 100)
        All[i, :] = np.interp(X, RawX, Raw)
        del Raw, RawX, X
    
    Avg = np.mean(100*All, axis=0)
    Std = np.std(100*All, axis=0)
    GC = {} # save data in dict for export
    GC['All'] = All
    GC['Avg'] = Avg
    GC['Std'] = Std
    GC['Start'] = Rising[0:-1]
    GC['End'] = Rising[1:-2]
    GC['Num_GCs'] = NumGCs

    return Stance, GC



def GetGCAccFoot(D):
    # get gait cycles from acceleration signal on foot
    
    import numpy as np
    from scipy import signal
    
    # calculate jerk (derivative of acceleration)
    # J = np.diff(D)
    
    # peak-finding parameters
    wid = 15
    p = 0.8
    ht = 0.95
    rh = 0.95
    d = 50
    
    # find peaks, widths, and bases
    Pks, PkVals = signal.find_peaks(D, height=ht, prominence=p, width=wid, distance=d)
    # results = signal.peak_widths(D, Pks, rel_height=rh)
    widths, width_heights, l_ips, r_ips = signal.peak_widths(D, Pks, rel_height=rh)
    
    ## plot curve, peaks, starts & ends
    # plt.plot(D)
    # plt.plot(Pks, D[Pks], 'om')
    # plt.hlines(*results[1:], 'k')
    # plt.plot(l_ips, width_heights, 'og')
    # plt.plot(r_ips, width_heights, 'or')
    
    Rising = [round(x) for x in l_ips]
    Falling = [round(x) for x in r_ips]
    Starts = []
    Ends = []
    
    # Get Stance Phases (heel strike to toe off)
    # loop through rise cycles and interpolate & save
    L = min(len(Rising), len(Falling))
    All = np.zeros([L, 100])
    a = 0
    for i in range(L):
        Raw = D[Rising[i]:Falling[i]]
        RawX = range(len(Raw))
        X = np.linspace(0, len(Raw), 100)
        
        if len(Raw) < 10: # exclude step if too short
          continue
    
        A = np.interp(X, RawX, Raw)
        
        # exclude invalid steps if: 
        if np.mean(A[10:20]) < 0.5 or np.mean(A[20:30]) < 0.5: # too low during loading peak
          continue
        if np.mean(A[30:40]) < 0.5 or np.mean(A[45:55]) < 0.5: # too low during mid stance
          continue
        if np.mean(A[65:75]) < 0.5 or np.mean(A[75:85]) < 0.5: # too low during propulsive peak
          continue
    
        All[i, :] = A # otherwise save step
        Starts.append(Rising[i])
        Ends.append(Falling[i])
        a += 1
        del Raw, RawX, X, A
    
    if a < L:
      All = All[0:a, :]
    Avg = np.mean(100*All, axis=0)
    Std = np.std(100*All, axis=0)
    Stance = {} # save data in dict for export
    Stance['All'] = All
    Stance['Avg'] = Avg
    Stance['Std'] = Std
    Stance['Start'] = Starts
    Stance['End'] = Ends
    Stance['Num_GCs'] = len(Starts)


#%% Other Functions

def ccc(x,y):
  # calculate concordance correlation coefficient

    import numpy as np
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc


#%% Data loading funcions
def LoadCSV(FName):
    import pandas as pd
    from datetime import datetime
    import numpy as np
    from scipy import interpolate
    
    Data = pd.read_csv(FName) # Load data

    # Translate Time Signal to seconds
    T = Data['time'].to_list()
    StartTime = datetime.fromtimestamp(T[0])
    time = []
    for x in T:
        Time = datetime.fromtimestamp(x)
        t = Time - StartTime
        time.append(t.total_seconds()) # save seconds elapsed

    Freq = 0.01
    NewTime = np.arange(0, round(time[-2], 2), Freq) 
    F = interpolate.interp1d(time, Data, axis=0)
    NewData = F(NewTime) 

    OutData = pd.DataFrame(data = NewData, columns=Data.columns)
    OutData['time'] = NewTime
    return OutData

def LoadTSV(FName):
    import pandas as pd
    # import datetime
    import numpy as np
    from scipy import interpolate
    
    Data = pd.read_csv(FName, skiprows=26, sep='\t') # Load data
    Freq = 1200
    NewFreq = 0.01
    Time = np.arange(0, len(Data) / Freq, 1 / Freq)
    NewTime = np.arange(0, round(Time[-1], 2), NewFreq) 
    F = interpolate.interp1d(Time, Data, axis=0)
    NewData = F(NewTime) 

    OutData = pd.DataFrame(data = NewData, columns=Data.columns)
    OutData['time'] = NewTime
    return OutData


#%% Check FP signals
def CheckFP(FP1, FP2):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12,8))
    plt.plot(FP1, label='FP1')
    plt.plot(FP2, label='FP2')
    plt.legend()
    
    