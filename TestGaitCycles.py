
#%% Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# from datetime import datetime #, timedelta
from scipy import signal
# from scipy import interpolate
import VettaFxn as vf
os.chdir(r'C:\Users\richa\Documents\Packages\Vetta\Piloting with IMUs\Testing 1_20')

# get files & trials
Files = os.listdir(os.getcwd())
CSVFiles = [x for x in Files if '.csv' in x]
TSVFiles = [x for x in Files if '.tsv' in x]
Trials = ['normal','over','under']
BodyMass = 78


#%% Processing Loop
T = 'under'
# for T in Trials: 
for f in CSVFiles: 
    if T in f:
        if 'waist' in f:
            Waist = vf.LoadCSV(f)
        if 'Lshank' in f:
            Lshank = vf.LoadCSV(f)
        if 'Rshank' in f:
            Rshank = vf.LoadCSV(f)
    
for f in TSVFiles: 
    if T in f: 
        if '_f_1' in f:
            Lfp = vf.LoadTSV(f)
        if '_f_2' in f:
            Rfp = vf.LoadTSV(f)

#%% Plot data to ensure accuracy
plt.close('all')
Accs = ['accel x','accel y','accel z']
Forces = ['Force_Z']
plt.figure(figsize=(14,10))
# plot waist
ax1 = plt.subplot(5,1, 1)
for i, a in enumerate(Accs):
    ax1.plot(Waist['time'], Waist[a], label=a)
ax1.legend()
ax1.set_ylabel('Waist (m/ss)')

# left shank
ax2 = plt.subplot(5, 1, 2)
for i, a in enumerate(Accs):
    ax2.plot(Lshank['time'], Lshank[a], label=a)
ax2.legend()
ax2.set_ylabel('Lshank (m/ss)')

# right shank
ax3 = plt.subplot(5, 1, 3)
for i, a in enumerate(Accs):
    ax3.plot(Rshank['time'], Rshank[a], label=a)
ax3.legend()
ax3.set_ylabel('Rshank (m/ss)')

# left FP
ax4 = plt.subplot(5, 1, 4)
for i, a in enumerate(Forces):
    ax4.plot(Lfp['time'], Lfp[a], label=a)
ax4.legend()
ax4.set_ylabel('L FP (N)')

# right FP
ax5 = plt.subplot(5, 1, 5)
for i, a in enumerate(Forces):
    ax5.plot(Rfp['time'], Rfp[a], label=a)
ax5.legend()
ax5.set_ylabel('Rs FP (N)')

os.chdir(r'C:\Users\richa\Documents\Packages\Vetta\Piloting with IMUs')
plt.show()
plt.savefig('Figs/' + T + 'Walking.png')

# Check Fp Signals
# vf.CheckFP(Lfp['Force_Z'], Rfp['Force_Z'])

#%% Trim Signals
# get start & stop times
# AccStart = input('What time (s) to start ACC signal: ')
# AccEnd = input('What time (s) to end ACC signal: ')
# FpStart = input('What time (s) to start FP signal: ')
# FpEnd = input('What time (s) to end FP signal: ')
if T == 'normal':
    AccStart = 25
    AccEnd = 125
    FpStart = 58
    FpEnd = 155
elif T == 'over':
    AccStart = 5
    AccEnd = 125
    FpStart = 45
    FpEnd = 170
elif T == 'under':
    AccStart = 12
    AccEnd = 124
    FpStart = 8
    FpEnd = 130

#%% run cross correlation between signals
plt.close('all')
# left
FaL = Lfp[Lfp['time'] == float(FpStart)].index[0]
FbL = Lfp[Lfp['time'] == float(FpEnd)].index[0]
x = Lfp['Force_Z'][FaL:FbL]
aL = Lshank[Lshank['time'] == float(AccStart)].index[0]
bL = Lshank[Lshank['time'] == float(AccEnd)].index[0]
y = Lshank['accel y'][aL:bL]
# Lcorr = signal.correlate(x, y)
# lags = signal.correlation_lags(x.size, y.size, mode="full")
# Llag = lags[np.argmax(Lcorr)]

# right
FaR = Rfp[Rfp['time'] == float(FpStart)].index[0]
FbR = Rfp[Rfp['time'] == float(FpEnd)].index[0]
x = Rfp['Force_Z'][FaR:FbR]
aR = Rshank[Rshank['time'] == float(AccStart)].index[0]
bR = Rshank[Rshank['time'] == float(AccEnd)].index[0]
y = Rshank['accel y'][aR:bR]
# Rcorr = signal.correlate(x, y)
# lags = signal.correlation_lags(x.size, y.size, mode="full")
# Rlag = lags[np.argmax(Rcorr)]

# trim FP signals
LfpTrim = Lfp.iloc[FaL:FbL, :].reset_index(drop=True)
LfpTrim['time'] = LfpTrim['time'] - LfpTrim['time'].to_list()[0]
RfpTrim = Rfp.iloc[FaR:FbR, :].reset_index(drop=True)
RfpTrim['time'] = RfpTrim['time'] - RfpTrim['time'].to_list()[0]

# reset time for acc signals
LshankTrim = Lshank.iloc[aL:bL, :].reset_index(drop=True)
LshankTrim['time'] = LshankTrim['time'] - LshankTrim['time'].to_list()[0] 
RshankTrim = Rshank.iloc[aR:bR, :].reset_index(drop=True)
RshankTrim['time'] = RshankTrim['time'] - RshankTrim['time'].to_list()[0] 

raise StopIteration


#%%
plt.close('all')
class PeakFinder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

fig, ax = plt.subplots()
npoints = 1
a = 0
b = 2000

# pick left stomp
ax.plot(np.multiply(0.01, LfpTrim['Force_Z'][a:b].tolist()), picker=npoints, label='FP')
plt.title('Select the stomp peak for Fp signal')
line, = ax.plot([0], [0])  # empty line
FpPks = PeakFinder(line)
plt.show()

#%%
fig, ax = plt.subplots()
ax.plot(LshankTrim['accel y'][a:b].tolist(), picker=npoints, label='Acc')
plt.title('Select the stomp peak for Acc signal')
line, = ax.plot([0], [0])  # empty line
AccPks = PeakFinder(line)
plt.show()

#%% align signals based on lag
plt.close('all')
plt.figure(figsize=(12,8))
ax1 = plt.subplot(211)
win = 10 # frames
AccPk = round(AccPks.xs[1])
AccPkInd = signal.find_peaks(LshankTrim['accel y'][AccPk-win:AccPk+win], distance=2*win)
ax1.plot(LshankTrim['accel y'][a:b].tolist(), label='L Acc')
AccPkFullInd = AccPk - win + int(AccPkInd[0])
ax1.plot(AccPkFullInd, LshankTrim['accel y'][AccPkFullInd], 'og')

FpPk = round(FpPks.xs[1])
FpPkInd = signal.find_peaks(LfpTrim['Force_Z'][FpPk-win:FpPk+win], distance=2*win)
ax1.plot(np.multiply(0.01, LfpTrim['Force_Z'][a:b].tolist()), label='L FP')
FpPkFullInd = FpPk - win + int(FpPkInd[0])
ax1.plot(FpPkFullInd, np.multiply(0.01, LfpTrim['Force_Z'][FpPkFullInd]), 'og')
ax1.set_title('Original Signals')

# delete 
Dif = abs(AccPkFullInd - FpPkFullInd)
if AccPkFullInd > FpPkFullInd:
    LshankTrim = LshankTrim.iloc[Dif:, :].reset_indexd(drop=True)
    RshankTrim = RshankTrim.iloc[Dif:, :].reset_index(drop=True)
    # WaistTrim = WaistTrim.iloc[Dif:, :].reset_index(drop=True)
elif FpPkFullInd > AccPkFullInd:
    LfpTrim = LfpTrim.iloc[Dif:, :].reset_index(drop=True)
    RfpTrim = RfpTrim.iloc[Dif:, :].reset_index(drop=True)
    
    
# plot trimmed signals
ax2 = plt.subplot(212)
ax2.plot(LshankTrim['accel y'][a:b].tolist(), label='L Acc')
ax2.plot(np.multiply(0.01, LfpTrim['Force_Z'][a:b].tolist()), label='L FP')  

# set Forces < Thresh == 0
Thresh = 30
Inds = LfpTrim['Force_Z'][LfpTrim['Force_Z'] < Thresh].index.to_list()
LfpTrim['Force_Z'].iloc[Inds] = 0
Inds = RfpTrim['Force_Z'][RfpTrim['Force_Z'] < Thresh].index.to_list()
RfpTrim['Force_Z'].iloc[Inds] = 0

# reset time again
LshankTrim['time'] = LshankTrim['time'] - LshankTrim['time'].to_list()[0] 
RshankTrim['time'] = RshankTrim['time'] - RshankTrim['time'].to_list()[0] 
LfpTrim['time'] = LfpTrim['time'] - LfpTrim['time'].to_list()[0] 
RfpTrim['time'] = RfpTrim['time'] - RfpTrim['time'].to_list()[0] 

# cut back side if one signal is longer than the other
M = np.min([len(LfpTrim), len(LshankTrim), len(RfpTrim), len(RshankTrim)])
LshankTrim = LshankTrim.iloc[0:M]
RshankTrim = RshankTrim.iloc[0:M]
LfpTrim = LfpTrim.iloc[0:M]
RfpTrim = RfpTrim.iloc[0:M]


# plot signals up close
scale = 0.02
# win = 2000
# a = 0
# b = a + win
# plt.figure(figsize=(12,8))
# plt.plot(LshankTrim['accel y'][a:b].to_list(), label='L shank')
# plt.plot(np.multiply(scale, LfpTrim['Force_Z'][a:b].to_list()), label='L FP')
# plt.legend()

# plt.figure(figsize=(12,8))
# plt.plot(RshankTrim['accel y'][a:b].to_list(), label='R shank')
# plt.plot(np.multiply(scale, RfpTrim['Force_Z'][a:b].to_list()), label='R FP')
# plt.legend()

#%% define function for Selles gait cycle ID function
def SellesID(FpTrim, AccTrim, name, plot=False):
    # use Selles algorithm to identify start/stop gait cycle times
    # based on: doi: 10.1109/TNSRE.2004.843176

    # part 1 - get gait cycle duration
    cutoff = 0.75 # cutoff frequency
    gain =  cutoff / np.sqrt(2)
    sos = signal.butter(2, gain, fs=100, output='sos') # 2nd-order low-pass zero-phase-lag Butterworth filter 
    F = signal.sosfiltfilt(sos, AccTrim, axis=0)
    Filt = pd.DataFrame(data=F, columns=AccTrim.columns)
    
    # get gait cycle duration
    prom = (0.05, 0.5)
    [locs, props] = signal.find_peaks(-Filt['accel y'], prominence=prom)
    GCDur = np.mean(np.diff(locs)) / 100
    
    if plot == True:
        win = 4000
        a = 0
        b = a + win
        plt.figure(figsize=(15,10))
        plt.plot(FpTrim['time'][a:b].to_list(), np.multiply(scale, FpTrim['Force_Z'][a:b].to_list()), label='FP')
        plt.plot(AccTrim['time'][a:b].to_list(), AccTrim['accel y'][a:b].to_list(), label='Acc')
        plt.plot(Filt['time'][a:b].to_list(), Filt['accel y'][a:b].to_list(), label='Acc Filt')
        plt.legend()
        plt.title(name + ' Gait Cycle Duration')
        pks = Filt['accel y'].iloc[locs]
        plt.plot(Filt['time'].iloc[locs], pks, 'ok')
        plt.xlim((0, 30))
    
    
    # Part 2 - identify initial contact and final contact times
    cutF = 2.5 - 0.4*GCDur
    gain =  cutF / np.sqrt(2)
    sos = signal.butter(2, gain, fs=100, output='sos') # 2nd-order low-pass zero-phase-lag Butterworth filter 
    GCF = signal.sosfiltfilt(sos, AccTrim, axis = 0)
    GCFilt = pd.DataFrame(data=GCF, columns=AccTrim.columns)
    
    # get initial contact times
    prom = (0.01, 0.5) # specify promimence for small peak
    [IClocs, ICprops] = signal.find_peaks(GCFilt['accel y'], prominence=prom)
    ICpks = GCFilt['accel y'].iloc[IClocs]

    # get final contact times
    prom = (1, 5)  # specify promimence for large peak
    [FClocs, FCprops] = signal.find_peaks(GCFilt['accel y'], prominence=prom)
    FCpks = GCFilt['accel y'].iloc[FClocs]
    
    if plot == True:
        win = 2000
        a = 2000
        b = a + win
        plt.figure(figsize=(15,10))
        plt.plot(FpTrim['time'][a:b].to_list(), np.multiply(scale, FpTrim['Force_Z'][a:b].to_list()), label='FP')
        plt.plot(GCFilt['time'][a:b].to_list(), GCFilt['accel y'][a:b].to_list(), label='Acc Filt')
        plt.legend()
        plt.title(name)
        plt.plot(GCFilt['time'].iloc[IClocs], ICpks, 'og')
        plt.plot(GCFilt['time'].iloc[FClocs], FCpks, 'or')
        plt.xlim((20, 20+win/100))
        
    # save outputs in dict
    Out = {}
    Out['GC Duration'] = GCDur
    Out['IC'] = IClocs
    Out['FC'] = FClocs
    Out['Filt'] = GCFilt
    return Out


#%% Get standard gait events from force plates
# LEFT Analysis
LStance, GCs = vf.GetGCs(np.divide(LfpTrim['Force_Z'].to_list(), BodyMass*9.81))
LSelles = SellesID(LfpTrim, LshankTrim, T)

plt.close('all')
plt.figure(figsize=(12,8))
win = 1000
a = 2000
b = a + win
# plot signals
time = LshankTrim['time']
plt.plot(time, np.divide(LfpTrim['Force_Z'].to_list(), BodyMass), label='vGRF')
plt.plot(time, LshankTrim['accel y'].to_list(), label='Raw Acc')
plt.plot(time, LSelles['Filt']['accel y'], label='Filt Acc')
# add gait cycles
plt.vlines(np.divide(LStance['Start'], 100), -5, 25, ls='-', label='vGRF IC', color='k')
plt.vlines(np.divide(LStance['End'], 100), -5, 25, ls='--', label='vGRF FC', color='k')
IC = LSelles['IC']
FC = LSelles['FC']
plt.plot(LSelles['Filt']['time'].iloc[IC], LSelles['Filt']['accel y'].iloc[IC], 'og', label='S IC')
plt.plot(LSelles['Filt']['time'].iloc[FC], LSelles['Filt']['accel y'].iloc[FC], 'or', label='S FC')
plt.xlim((30, 42))
plt.xlabel('Time (s)')
plt.legend()
plt.savefig('Figs/GC_Selles_' + T + '_Left.png')

# RIGHT Analysis
RStance, GCs = vf.GetGCs(np.divide(RfpTrim['Force_Z'].to_list(), BodyMass*9.81))
RSelles = SellesID(RfpTrim, RshankTrim, T)

plt.figure(figsize=(12,8))
win = 1000
a = 2000
b = a + win
# plot signals
time = RshankTrim['time']
plt.plot(time, np.divide(RfpTrim['Force_Z'].to_list(), BodyMass), label='vGRF')
plt.plot(time, RshankTrim['accel y'].to_list(), label='Raw Acc')
plt.plot(time, RSelles['Filt']['accel y'], label='Filt Acc')
# add gait cycles
plt.vlines(np.divide(RStance['Start'], 100), -5, 25, ls='-', label='vGRF IC', color='k')
plt.vlines(np.divide(RStance['End'], 100), -5, 25, ls='--', label='vGRF FC', color='k')
IC = RSelles['IC']
FC = RSelles['FC']
plt.plot(RSelles['Filt']['time'].iloc[IC], RSelles['Filt']['accel y'].iloc[IC], 'og', label='S IC')
plt.plot(RSelles['Filt']['time'].iloc[FC], RSelles['Filt']['accel y'].iloc[FC], 'or', label='S FC')
plt.xlim((30, 42))
plt.xlabel('Time (s)')
plt.legend()
plt.savefig('Figs/GC_Selles_' + T + '_Right.png')

#%% Analyze Gait Event Detection
def AnalyzeGE(Events, Test):
    # analyze gait events between sensor based (test) and force plate (events) detection methods
    
    Thresh = 50 # set gait event detection threshold (only analyzes events that are THRESH apart in frames (assumed 100 Hz))
    Missed = []
    Found = []
    NearErr = []
    for x in Events:
        Errs = abs(np.subtract(Test, x))
        ValidErrs = [i for i in Errs if abs(i) < Thresh]
        if len(ValidErrs) > 0:
            NearErr.append(ValidErrs[0])
            Found.append(1)
        else:
            Missed.append(1)

    return Missed, Found, NearErr
        

Missed, Found, NearErr = AnalyzeGE(LStance['Start'], LSelles['IC'])
MeanICErr = round(np.mean(NearErr) / 100, 2)
StdICErr = round(np.std(NearErr) / 100, 2)
print('Mean IC Err: ', MeanICErr, 's')
print('Std IC Err: ', StdICErr, 's')
print('Missed ICs:', sum(Missed))
print('Found ICs:', sum(Found))

Missed, Found, NearErr = AnalyzeGE(LStance['End'], LSelles['FC'])
MeanFCErr = round(np.mean(NearErr) / 100, 2)
StdFCErr = round(np.std(NearErr) / 100, 2)
print('Mean FC Err: ', MeanFCErr, 's')
print('Std FC Err: ', StdFCErr, 's')
print('Missed FCs:', sum(Missed))
print('Found FCs:', sum(Found))


#%% Vector Magnitude Acc & Jerk gait events
plt.close('all')

def VectMagGE(Shank, StanceData, Plot=False):
    
    # calculate ACC vector magnitude 
    x = Shank['accel x']
    y = Shank['accel y']
    z = Shank['accel z']
    VM = np.sqrt(np.multiply(x,x) + np.multiply(y,y) + np.multiply(z,z)).tolist()
    Jerk = np.diff(VM)
    
    # filter signals
    cutoff = 6 # cutoff frequency
    gain =  cutoff / np.sqrt(2)
    sos = signal.butter(2, gain, fs=100, output='sos') # 2nd-order low-pass zero-phase-lag Butterworth filter 
    F = signal.sosfiltfilt(sos, LshankTrim, axis=0)
    Filt = pd.DataFrame(data=F, columns=LshankTrim.columns)
    xF = Filt['accel x']
    yF = Filt['accel y']
    zF = Filt['accel z']
    VMF = np.sqrt(np.multiply(xF,xF) + np.multiply(yF,yF) + np.multiply(zF,zF)).tolist()
    JerkF = np.diff(VMF)
    
    # get final contact times
    prom = 5 # specify promimence for small peak
    [FClocs, FCprops] = signal.find_peaks(VMF, prominence=prom)
    FCpks = [VMF[x] for x in FClocs]
    
    # get initial contact times
    prom = (1, 5)  # specify promimence for large peak
    wid = (5, 20)
    [IClocs, ICprops] = signal.find_peaks(np.multiply(-1, VMF), prominence=prom, width=wid)
    ICpks = [VMF[x] for x in IClocs]
    
    if Plot == True:
        # plot Gait events
        a = 3000
        b = 4000
        plt.figure(figsize=(12,6))
        ax1 = plt.subplot(211)
        ax1.plot(VM, label='VM')
        ax1.vlines(StanceData['Start'], 0, 30, 'k', ls='-', label='IC')
        ax1.vlines(StanceData['End'], 0, 30, 'k', ls='--', label='FC')
        ax1.set_title('Acc Vector Magnitude')
        ax1.set_xlim((a,b))
        ax1.plot(VMF, label='VM Filtered')
        ax1.plot(IClocs, ICpks, 'og', label='VM IC')
        ax1.plot(FClocs, FCpks, 'or', label='VM FC')
        ax1.legend()
        
        ax2 = plt.subplot(212)
        ax2.plot(Jerk, label='Jerk')
        ax2.vlines(StanceData['Start'], -20, 20, 'k', ls='-')
        ax2.vlines(StanceData['End'], -20, 20, 'k', ls='--')
        ax2.set_title('Jerk Vector Magnitude')
        ax2.set_xlim((a,b))
        ax2.plot(JerkF, label='Jerk Filtered')
        ax2.legend()
        
        plt.savefig('Figs/GC_VM_' + T + '_Left.png')
    
    
    # print accuracies
    print(' ')
    print('VM results: ', T, ' condition')
    Missed, Found, NearErr = AnalyzeGE(LStance['Start'], IClocs)
    MeanICErr = round(np.mean(NearErr) / 100, 2)
    StdICErr = round(np.std(NearErr) / 100, 2)
    print('Mean IC Err: ', MeanICErr, 's')
    print('Std IC Err: ', StdICErr, 's')
    print('Missed ICs:', sum(Missed))
    print('Found ICs:', sum(Found))
    
    Missed, Found, NearErr = AnalyzeGE(LStance['End'], FClocs)
    MeanFCErr = round(np.mean(NearErr) / 100, 2)
    StdFCErr = round(np.std(NearErr) / 100, 2)
    print('Mean FC Err: ', MeanFCErr, 's')
    print('Std FC Err: ', StdFCErr, 's')
    print('Missed FCs:', sum(Missed))
    print('Found FCs:', sum(Found))


# run VM gait event detection for left side
VectMagGE(LshankTrim, LStance, True)

#%% Trojaniello Gait event detection
# based on: doi: 10.1186/1743-0003-11-152
# pros: no filtering,
# cons: requires accurate sensing of both limbs (L&R) to narrow down GE times, requires specific sensor orientation
plt.close('all')

LShank = LshankTrim
RShank = RshankTrim
Plot = True

# define inputs
SagPlane = 'z'
APPlane = 'x'
LAx, LAy, LAz  = LShank['accel x'], LShank['accel y'], LShank['accel z']
LGx, LGy, LGz = LShank['gyro x'], LShank['gyro y'], LShank['gyro z']
RAx, RAy, RAz  = RShank['accel x'], RShank['accel y'], RShank['accel z']
RGx, RGy, RGz = RShank['gyro x'], RShank['gyro y'], RShank['gyro z']

# identify trusted swing phases
# time interval with ωz larger than a set threshold (20%) of its local maximum value Mp.
# get local maximum values of swing phases
ht = 2
d = 50
Llocs, Lprops = signal.find_peaks(LGz, height=2, distance=d)
# Mp = props['peak_heights'].mean()
# SwingThresh = Mp * 0.25
Lwid, Lwidht, L_Lips, L_Rips = signal.peak_widths(LGz, Llocs, rel_height=0.8)
L_Lip = L_Lips.round()
L_Rip = L_Rips.round()
L_ICT = np.zeros((len(LGz), 1))

# IC = minimum value of the ML angular velocity occurring in TIC before the instant of maximum AP acceleration. 


# FC = instant of minimum AP acceleration in the TFC, 
# since it is expected to occur at the time of a sudden motion of the shank preceding the instant of the last maximum AP acceleration value in TFC


if Plot == True:
    # plot gyros
    plt.figure(figsize=(12, 6))
    plt.plot(LGz, label='L gyro z')
    plt.plot(RGz, label='R gyro z')
    plt.legend()
    plt.xlim((3000, 4000))
    plt.plot(Llocs, LGz[Llocs], 'xk')
    plt.plot(L_Lip, LGz[L_Lip], 'og')
    plt.plot(L_Rip, LGz[L_Rip], 'or')
    plt.xlim((3000, 4000))
    plt.savefig('Figs/GC_Troj_' + T + '.png')
    
    
#%% Ledoux Gait Event Detection
    
# needs real time angle

