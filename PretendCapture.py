# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 07:54:20 2023

@author: richa
"""

# import serial
# import serial.tools.list_ports
import struct
import time
import json
import sys
import numpy as np
# import scipy
from scipy import signal
import joblib
# import random
import os
from os.path import exists
from enum import Enum
import warnings
import VettaFxn as vf
import matplotlib.pyplot as plt

#For reasons related to serial interfaces in Unity, it is easiest for demo
#purposes to externalize
#communication with the Stryde hub to this process.  Anything related to the
#USB device, including detecting connection/removal
#should be in this process.  Status and data should be communicated via simple
#API and/or redirecting StandardIO from the Unity side.

#TODO (TIME PERMITTING): Fully embedded interpreter to allow graceful shutdown
#rather than killing the whole subprocess


#%% Constants

#Iterate This!
fileCount = "new_Raw"
stepsFile = "new_Steps"
waveFormFile = "new_WaveForms"

modelFile = 'MLPmodel_W' 
model = None

#Time-based (0) or an attempted real time construction (1)
StepDetectionMethod = 1

#Sensor unicode values
# waistID = "34"
# leftShankId = "35"
# rightShankID = "36"

#Sample lists
# leftWaistSamples = []
# rightWaistSamples = []
# leftShankSamples = []
# rightShankSamples = []

#Should be true on first event.  Set to false and samples are cleared on second event
#"Stance" is probably a misnomer at this point due to changes in event detection.
leftStance = False
rightStance = False

# minShankSampleCount = 20

#Used for waveform id
leftPeakId = 0
rightPeakId = 0

sampleCount = 0

# VMJerk values: Initialize once
cutoff =6
gain = cutoff / np.sqrt(2)
sos = signal.butter(2, gain, fs =100, output='sos')

#%% Define Funcitons 

# #Actually passes values to the model
# def PredictPeakVGRF(waistSamples,id,side):

#     global vgrfWaveForms
#     if model == None:
#         print("No Model Loaded")
#         return

#     magnitudes = []
#     for sample in waistSamples:
#         magnitudes.append(GetMagnitude(sample.accel))
#     #print(len(magnitudes))
#     #print(magnitudes)
#     inter_magnitudes = signal.resample(magnitudes,100) 
#     #print(len(inter_magnitudes))
#     #print(inter_magnitudes)
#     vgrf = model.predict([inter_magnitudes])[0]
#     #print(vgrf)

#     #save full wave form
#     vgrfWaveForm = VGRFWaveForm(id,time.time(),side,vgrf)
#     jsonData = str(vgrfWaveForm)
#     vgrfWaveForms.append(jsonData)

#     #Grab peak vgrf for stimulus
#     height = .8         # These parameters will have to be doublechecked
#     distance = 10
#     prominence = .15
#     width = 2
#     peaks,properties = signal.find_peaks(vgrf, height = height, prominence = prominence, width = width, distance = distance)
#     peakSample = VGRFSample(id,time.time(),side,properties['peak_heights'][0])
#     return peakSample

def LoadModel():
    return joblib.load(modelFile) 

# def GetMagnitude(sample):
#     return np.sqrt(sample[0] ** 2 + sample[1] ** 2 + sample[2] ** 2)

# #Derived from Ricky's example
# def VectorMagJerk(samples):
#     global sos
#     x = []
#     y = []
#     z = []
#     for sample in samples:
#         x.append(sample.accel[0])
#         y.append(sample.accel[1])
#         z.append(sample.accel[2])
#     x = np.array(x)
#     y = np.array(y)
#     z = np.array(z)
#     shankValues = np.array([x,y,z])
#     jerk = np.diff(np.linalg.norm(shankValues,axis=0))     # Vector Norm Jerk
#     F = signal.sosfiltfilt(sos,shankValues.T, axis = 0)
#     #Filt = pd.DataFrame(data=F)
#     xF = F[0]
#     yF = F[1]
#     zF = F[2]
#     VMF = np.sqrt(np.multiply(xF,xF) + np.multiply(yF,yF) + np.multiply(zF,zF)).tolist()
    
#     return VMF, jerk

def GetVMAJ(samples):
    # return vector magnitude acceleration and jerk from list-based input
    global SampleCols 
    
    Sam = np.reshape(samples, (len(samples), len(SampleCols)))
    x = np.array(Sam[:, 2])
    y = np.array(Sam[:, 3])
    z = np.array(Sam[:, 4])
    Values = np.array([x,y,z])
    VMA = np.linalg.norm(Values,axis=0)
    jerk = np.diff(VMA)     # Vector Norm Jerk
    
    return VMA, jerk


#Currently no references.  Initial and Final event detection.  FindHeelStrikes is what's currently used
# def FindGaitEvents(VMF, jerk):
#     prom = 5 # specify promimence for small peak
#     [FClocs, FCprops] = signal.find_peaks(VMF, prominence=prom)
#     FCpks = [VMF[x] for x in FClocs]
    
#     # get initial contact times
#     prom = (1, 5)  # specify promimence for large peak
#     wid = (5, 20)
#     [IClocs, ICprops] = signal.find_peaks(np.multiply(-1, VMF), prominence=prom, width=wid)
#     ICpks = [VMF[x] for x in IClocs]

#     return FCpks,ICpks

def FindHeelStrikes(jerk):
    ht = 3
    dis = 50
    IClocs, ICprops = signal.find_peaks(jerk, height=ht, distance=dis)
    ICpks = [jerk[x] for x in IClocs]

    return IClocs, ICpks


# def FindHeelStrikes(VMF):
#     prom = (.5, 4)  # specify promimence for large peak
#     wid = (5, 30)
#     IClocs, ICprops = signal.find_peaks(np.multiply(-1, VMF), prominence=prom, width=wid)
#     ICpks = [VMF[x] for x in IClocs]
#     #peaks,_ = scipy.signal.find_peaks(jerk,height = 5,distance=10)
#     return ICpks

#Need to test height band parameters tomorrow morning.
# def FindToeOffs(VMF):
#     prom = 5 # specify promimence for small peak
#     [FClocs, FCprops] = signal.find_peaks(VMF, prominence=prom)
#     FCpks = [VMF[x] for x in FClocs]

#     return FCpks


#%% Run through already captured data 
# pretend we are live streaming to test gait event detection


# load sensors & model
DataFolder = 'Test_Feb8_2023'
Waist = vf.LoadCSV(os.path.join(DataFolder, 'Normal_Waist.csv'))
LShank = vf.LoadCSV(os.path.join(DataFolder, 'Normal_LShank.csv'))
RShank = vf.LoadCSV(os.path.join(DataFolder, 'Normal_RShank.csv'))
# model = LoadModel()

shankSampleTarget = 100
leftShankSamples = []
LStepTimes = []
leftWaistSamples = []
sampleCount = 0
timeThresh = 2

# check length
Dur = min(len(Waist), len(LShank), len(RShank))

global SampleCols
SampleCols = LShank.columns

# define loop
for t in range(Dur):
    
    new_sample = LShank.iloc[t, :].to_list()
    
    # Create Shank Signals for step counting
    if len(leftShankSamples) > 300:
        leftShankSamples = []
        leftWaistSamples =[]
    leftShankSamples.append(new_sample)

    # Create Waist Signals for predictions
    leftWaistSamples.append(Waist.iloc[t, :])
    sampleCount+=1
    
    #StepCheck
    if StepDetectionMethod == 1:
        if len(leftShankSamples) >= shankSampleTarget: # if sufficient samples to check for a gait event
            VMA, jerk = GetVMAJ(leftShankSamples)
            Lind, pks = FindHeelStrikes(jerk)
            
            if len(Lind) > 0:
                LStepTimes.append(leftShankSamples[Lind[0]][1])
                print("Left Step Found at: t = " + str(leftShankSamples[Lind[0]][1]))
            leftShankSamples = []
            
            
            if len(LStepTimes) > 2:
            
                # get previous two heel strike events
                Curr = LStepTimes[-1]
                Prev = LStepTimes[-2]
            
                if Curr - Prev < timeThresh: # ensure they are no more than 2 sec apart
      
                    # get waist samples during that time
                    GCStart = Waist[Waist['time'] == Prev].index[0]
                    GCEnd = Waist[Waist['time'] == Curr].index[0]
                    
                    # get VMA for Waist
                    x = Waist['accel x'][GCStart:GCEnd]  
                    y = Waist['accel y'][GCStart:GCEnd]  
                    z = Waist['accel z'][GCStart:GCEnd]  
                    W_VMA = np.linalg.norm(np.array([x,y,z]),axis=0)
                
                    # resample to 100 points
                    ReSamWaist = signal.resample(W_VMA, 100)
            
                    # load into model
                    # model()
    
                          
print('Num Left Steps:', len(LStepTimes))

