# VetaUtils.py
"""
Utility functions to process Vetta Pilot data and set up Vetta SQL database

"""

import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import datetime 
import sqlite3
import time 
from scipy import signal
# import sklearn 
import onnxruntime as rt
from scipy.io import loadmat

#%%
class AllData:
    
    def __init__(self):
        
        if hasattr(self, 'sqlDB') == 0:
            self.sqlDB = 'VettaPilot_Pred.db'
            
        if hasattr(self, 'model') == 0:
            # self.model = joblib.load('MLPmodel_Wnew')
            mdlName = 'MLPmodelW.onnx'
            self.model = rt.InferenceSession(mdlName, providers=["CPUExecutionProvider"])


    
    
    def getTableNames(self):

        # create connection to sql db
        con = sqlite3.connect(self.sqlDB)
     
        # Getting all tables from sqlite_master
        sql_query = """SELECT name FROM sqlite_master WHERE type='table';"""
        cursor = con.cursor() # Create cursor object using connection object
        cursor.execute(sql_query) # execute our sql query
        tableNames = [str(np.squeeze(x)) for x in cursor.fetchall()]
        # print(tableNames)
        self.TableNames = tableNames
        
        return tableNames
    


    def checkSensorData(self, ElapsedData):
        """
        Check sensor data quality by calculating 
        average sampling frequency and dropout analysis
        """
        # T = ElapsedData['Elapsed'] # define timeseries data 
        T = ElapsedData.index.to_list()
        NumSamp = len(T) # num samples
        RecDur = T[-1] # recording duration      
        MeanSampFreq = round(NumSamp / RecDur, 2) # mean sampling frequency

        # dropout analysis
        Thresh = 0.025 # define a dropout threshold of _ seconds
        DropInds = np.zeros((NumSamp - 1, 1))
        DropDurs = []
        for i, t in enumerate(T): 
            if i == len(T)-1:
                break
            SampDiff = T[i+1] - t
            if SampDiff > Thresh:
                DropInds[i] = 1
                DropDurs.append(SampDiff)

        MeanDropLen = np.nanmean(DropDurs)
        StdDropLen = np.nanstd(DropDurs)
        NumDrops = len(DropDurs)

        DQ = {}
        DQ['TimeData'] = T
        DQ['NumSamp'] = NumSamp
        DQ['RecDur'] = RecDur
        DQ['MeanSampFreq'] = MeanSampFreq
        DQ['DropInds'] = DropInds
        DQ['DropDurs'] = DropDurs
        DQ['MeanDropDur'] = np.mean(DropDurs)
        DQ['MeanDropLen'] = MeanDropLen
        DQ['StdDropLen'] = StdDropLen
        DQ['NumDrops'] = NumDrops
        DQ['Dropout Threshold'] = Thresh
        self.DQ = DQ
        
        return DQ

    
    def loadSensor(self, user, speed, cond, sensorID):
        # download sql data based on selections
        table_name = '_'.join([user, speed, cond])
        sqlStr = f"SELECT * FROM {table_name}"
        df = sql_to_pd(sqlStr, self.sqlDB)
        DF = df[df.id == sensorID].sort_values(by='time')
        # transform time from unix standard to real time
        TimeStr = [datetime.datetime.fromtimestamp(x) for x in DF.time]
        Start = TimeStr[0]
        Elapsed = []
        for x in TimeStr:
            ts = x - Start
            Elapsed.append(ts.seconds + ts.microseconds / 1000000)
            
        ElapsedData = pd.DataFrame(dict(y=DF['accel'].to_list()), index=Elapsed)
        
        # resample to 100 Hz and normalize to Gs from m/s^2
        newX = np.arange(Elapsed[0], Elapsed[-1], 0.01).round(2)
        Resam = np.interp(newX, ElapsedData.index, ElapsedData.y) / 9.81
        ResamData = pd.DataFrame(dict(y=Resam.tolist()), index=newX)

        # check sensor data quality
        DQ = self.checkSensorData(ElapsedData)

        # filter data
        sos = signal.butter(4, 0.1, output='sos') # 4th order, 10 Hz, lowpass filter
        Filt = signal.sosfiltfilt(sos, Resam)
        
        # plot filtering
        PlotFilt = 0
        if PlotFilt == 1:
            plt.figure()
            plt.plot(Resam, label='Reampled')
            plt.plot(Filt, label='Filtered')
            plt.legend()
        
        # save filtered data in dataframe
        FiltData = pd.DataFrame(dict(y=Filt.tolist()), index=newX) 
        
        return FiltData, ResamData, DQ
    


    def checkPreds(self):
        """ 
        Check SQL database for outcome predictions and return lists of all 
        unique subjects, speeds, and conditions. 
        """
         
        if hasattr(self, 'TableNames') is False:
            self.getTableNames()

        GCNames = [x for x in self.TableNames if 'GC' in x]
        if len(GCNames) > 0:
            print('Found:', str(len(GCNames)), 'predictions in database')

        # save prediction metrics into self
        Subjs = np.unique([x.split('_')[0] for x in GCNames])
        print('Subjects:', Subjs)

        Speeds = np.unique([x.split('_')[1] for x in GCNames])
        print('Speeds:', Speeds)

        Conds = np.unique([x.split('_')[2] for x in GCNames])
        print('Conds:', Conds)

        PD = {} # PD = prediction data
        PD['Subjs'] = Subjs
        PD['Speeds'] = Speeds
        PD['Conds'] = Conds

        self.PD = PD

        return PD



    def plotPreds(self,):
        """
        Plot vGRF predictions for each speed and condition, 
        averaged across all subjects
        """

        if hasattr(self, 'PD') is False:
            self.checkPreds()

        # plt.figure(figsize=(10,10))
        # ax1 = plt.subplot(3,2,1)
        # ax2 = plt.subplot(3,2,2)
        # ax3 = plt.subplot(3,2,3)
        # ax4 = plt.subplot(3,2,4)
        # ax5 = plt.subplot(3,2,5)
        # ax6 = plt.subplot(3,2,6)

        # get SQL table names for that condition
        GCNames = [x for x in self.TableNames if 'GC' in x]
        speed = [i for i,x in enumerate(GCNames) if '160' in x]
        cond = [i for i,x in enumerate(GCNames) if 'baseline' in x]
        side = [i for i,x in enumerate(GCNames) if 'L_GC' in x]
        inds = list(set(speed) & set(side) & set(cond))

        P = np.zeros((len(inds), 100))
        for i, x in enumerate(inds):
            # get SQL tables and append to array
            print(GCNames[x])

            sqlStr = f"SELECT * FROM {GCNames[x]}"
            print(sqlStr)
            df = sql_to_pd(sqlStr, self.sqlDB)

            # omit rows with no data 
            
            # df
            print(df)
            raise StopIteration

        # get loading conditions



    def exportCSV(self, TrialNames):
        """ Export CSV files of Raw, Waves, Peaks for all listed Trial Names """
        
        
        
        
    def getMass(self, Subj, PlotStatic=0):
        """ Get Subject Mass from Static Trial """
        
        SubjPath = os.path.join(os.getcwd(), 'PilotData',Subj)
        
        # check to see if Demo.json file exists
        DemoFile = [x for x in os.listdir(SubjPath) if 'Demo.json' in x]
        if not DemoFile:
            print('Analyzing Static Trial to get subject mass ...')
        else:
            f = open(os.path.join(SubjPath, DemoFile[0]), 'r') 
            DemoData = json.load(f)
            f.close()
            return DemoData['Mass']
        
        StaticFile = [x for x in os.listdir(SubjPath) if 'static' in x]
        if len(StaticFile) > 1 :
            print('Too many static trials for subj:', Subj)
            return []
        elif len(StaticFile) == 0:
            print('No Static file for subj:', Subj)
            return []
        else:
            StaticMatFile = StaticFile[0]
        
        # load vGRFs
        MatData = loadmat(os.path.join(SubjPath, StaticMatFile))
        GRF_R = MatData[StaticMatFile[0:-4]]['Force'][0][0]['Force'][0][0]
        GRFv_R = GRF_R[2,:].T
        GRF_L = MatData[StaticMatFile[0:-4]]['Force'][0][0]['Force'][0][1]
        GRFv_L = GRF_L[2,:].T
        
        # get time info
        NFrames =  len(GRFv_L)
        fr = int(MatData[StaticMatFile[0:-4]]['Force'][0][0]['Frequency'][0][0].squeeze()) # set frame rate
        Duration = NFrames / fr
        Time = np.linspace(0, Duration, num = NFrames) 
        Total = GRFv_L + GRFv_R
        BodyMass = round(np.nanmean(Total), 2)
        
        if PlotStatic == 1: # plot static data to ensure correct 
            plt.plot(Time, GRFv_R, label='Right')
            plt.plot(Time, GRFv_L, label='Left')
            plt.plot(Time, Total, label='Total')
            plt.title('Whole Trial - ' + StaticMatFile)
            plt.xlabel('Time (s)')
            plt.ylabel('vGRF (N)')
            plt.legend()
            plt.show()
        
        # print('Average Left side = ' + str(np.round(np.nanmean(GRFv_L))), ' N')
        # print('Average Right side = ' + str(np.round(np.nanmean(GRFv_R))), ' N')
        print('Body Mass = ' + str(BodyMass), ' N')
        
        # save to json file 
        OutDict = {'Mass': BodyMass}
        OutFile = open(os.path.join(SubjPath, 'Demo.json'), 'w')
        json.dump(OutDict, OutFile, indent=5)
        OutFile.close()
        # print('Saved body mass to demofile')
        
        return BodyMass


    def getSQLfromMatFile(self, MatFName):
        
        Subj = MatFName.split('_')[1]
        Speed = MatFName.split('_')[3][0:8]
        LoadCond = MatFName.split('_')[2]
        
        # determine speed and loading condition naming match
        if 'speed000' in Speed: 
            SpeedW = 'preferred'
        else:
            SpeedW = Speed[-3:]
 
        if 'pref' in LoadCond:
            LoadCondW = 'baseline'
        # elif 'pref' in LoadCond:
        #     LoadCondW = 'baseline'
        elif 'under' in LoadCond:
             LoadCondW = 'under3'
        else: 
            LoadCondW = LoadCond
            
        SubjInd = [i for i, x in enumerate(self.TableNames) if Subj in x] 
        LCInd = [i for i, x in enumerate(self.TableNames) if LoadCondW in x] 
        SpdInd = [i for i, x in enumerate(self.TableNames) if SpeedW in x] 
        TypeInd = [i for i, x in enumerate(self.TableNames) if 'GC' not in x] 
        # TypeInd = [i for i, x in enumerate(self.TableNames) if 'L_GC' in x] 
        Ind = list(set(SubjInd) & set(LCInd) & set(SpdInd) & set(TypeInd))
        if not Ind:          # if no associated sensor data, skip trial
            print('No associated sensor data.')
            SQLName = []
            return SQLName
        
        # load associated sensor data
        SQLName = self.TableNames[Ind[0]]
        
        return SQLName
        

    def loadTMdata(self, TMFileName, PlotTMData=1): 
        """ Associate walking trial with treadmill data """
        
        # load general vars
        Subj = TMFileName.split('_')[1]
        # Speed = TMFileName.split('_')[3][:-4]
        # LoadCond = TMFileName.split('_')[2]
        BodyMass = self.getMass(Subj) # load body mass for normalization to %BW
        
        # load vGRFs from datafile
        TMData = loadmat(os.path.join(os.getcwd(), 'PilotData', Subj, TMFileName))
        GRF_R = TMData[TMFileName[0:-4]]['Force'][0][0]['Force'][0][0]
        GRFv_R = GRF_R[2,:].T / BodyMass 
        GRF_L = TMData[TMFileName[0:-4]]['Force'][0][0]['Force'][0][1]
        GRFv_L = GRF_L[2,:].T / BodyMass
        
        # set time variables
        NFrames =  len(GRFv_L)
        fr = int(TMData[TMFileName[0:-4]]['Force'][0][0]['Frequency'][0][0].squeeze()) # set frame rate
        Duration = NFrames / fr
        Time = np.linspace(0, Duration, num = NFrames) 
        
        # resample to 100 hz
        GRFv_Rn = signal.resample_poly(GRFv_R, 100, fr)
        GRFv_Ln = signal.resample_poly(GRFv_L, 100, fr)
        
        TM = {}
        TM['GRFv_Rn'] = GRFv_Rn
        TM['GRFv_Ln'] = GRFv_Ln
        TM['Time'] = np.arange(0, Duration, 0.01).round(2) #Time
        
        # save TM in SQL db?
        
        return TM

        

        
        
        
#%%
class PilotData:
    """
    PilotData Class is designed to identify and save the Pilot 
    data collected in 2023 to a SQL database for model development
    """
    
    def __init__(self, path):
        
        # save path and subject name
        self.path = path
        PathStr = path.split('/')
        self.name = PathStr[-1]
        self.sqlDB = 'C:/Users/richa/Documents/Packages/Vetta/Database/VettaPilot_Pred.db'
        
        # get names of trials and conditions within main folder
        Trls = [x for x in os.listdir(path) if os.path.isdir(x)]
        TrlCond = {}
        for f in Trls: 
            if '.py' in f:
                continue
            if '.db' in f:
                continue
            if '.mat' in f:
                continue
            F = os.listdir(os.path.join(path, f))
            TrlCond[f] = [x for x in F if '.py' not in x]

        self.speed = Trls
        self.conds = TrlCond
        
        # state sensor ID numbers
        SensorIDs = {}
        SensorIDs['Waist'] = 36
        SensorIDs['LAnk'] = 35
        SensorIDs['RAnk'] = 34
        self.SensorIDs = SensorIDs
        
        
    # Raw Data Loading & Analysis
    def getRawData(self):
        """ get all Raw Data for a given subject and save in SQL db """
        
        Col = ['Subj','Speed','Cond','id', 'time', 'accel', 'gyro', 'mag', 'flag']
        AllRawData = pd.DataFrame(columns=Col)
        print('Loading:', self.name)
        
        for s in self.speed:
            for c in self.conds[s]:
                subPath = os.path.join(self.path, s, c)
                
                # skip if no Raw data files present
                if len([x for x in os.listdir(subPath) if 'Raw' in x]) == 0:
                    continue
                
                # load Raw File
                RawFile = [x for x in os.listdir(subPath) if 'Raw' in x][0]
                f = open(os.path.join(subPath, RawFile))
                J = json.load(f)
                f.close()
                RawData = pd.DataFrame(data=np.empty((len(J), len(Col))), columns=Col)
                speed = s.split('_')[-1]
                cond = c.split('_')[-1]
                print('Speed:', speed, '   Trial:', cond)
                start = time.time()
                
                for i in range(len(J)): # loop through json rows to load
                    D = pd.json_normalize(json.loads(J[i]))
                    RawData.iloc[i] = [self.name, speed, cond,
                                       D.id.squeeze(), D.time.squeeze(), 
                                       np.linalg.norm(D.accel.squeeze()), 
                                       np.linalg.norm(D.gyro.squeeze()), 
                                       np.linalg.norm(D.mag.squeeze()), 
                                       D.flag.squeeze()]
                    
                AllRawData = pd.concat([AllRawData, RawData])
                
                end = time.time()
                print('Load Time:', round(end-start, 2), 's')
                
                sqlTableName = '_'.join([self.name, speed, cond])
                # sqlDB = 'C:/Users/richa/Documents/Packages/Vetta/Database/VettaPilot.db'
                pd_to_sql(RawData, sqlTableName, self.sqlDB)

        self.RawData = AllRawData

        return AllRawData
    
    
    
    def plotRawData(self, SpeedChoice, CondChoice):
        
        D = self.RawData
        
        # specify speed & trial
        SpdInds = [x for x, z in enumerate(D['Speed'].tolist()) if z == SpeedChoice]
        CondInds = [x for x, z in enumerate(D['Cond'].tolist()) if z == CondChoice]

        # get Left side data
        IdInds = [x for x, z in enumerate(D['id'].tolist()) if z == '34']
        Inds = list(set(IdInds) & set(CondInds) & set(SpdInds))
        SubDataL = D.iloc[Inds]
        # sort timeseries
        SubDataL['TimeStr'] = [datetime.datetime.fromtimestamp(x) for x in SubDataL.time]
        SubDataL.sort_values(by='TimeStr', inplace=True)

        # get Right side data
        IdInds = [x for x, z in enumerate(D['id'].tolist()) if z == '35']
        Inds = list(set(IdInds) & set(CondInds) & set(SpdInds))
        SubDataR = D.iloc[Inds]
        # sort timeseries
        SubDataR['TimeStr'] = [datetime.datetime.fromtimestamp(x) for x in SubDataR.time]
        SubDataR.sort_values(by='TimeStr', inplace=True)
        
        # get waist data
        IdInds = [x for x, z in enumerate(D['id'].tolist()) if z == '36']
        Inds = list(set(IdInds) & set(CondInds) & set(SpdInds))
        SubDataW = D.iloc[Inds]
        # sort timeseries
        SubDataW['TimeStr'] = [datetime.datetime.fromtimestamp(x) for x in SubDataW.time]
        SubDataW.sort_values(by='TimeStr', inplace=True)

        # plot
        plt.figure(figsize=(10,10))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
        ax1.plot(SubDataW.TimeStr, SubDataW.accel, label='Right')
        ax2.plot(SubDataL.TimeStr, SubDataL.accel, label='Left')
        ax3.plot(SubDataR.TimeStr, SubDataR.accel, label='Right')
        plt.legend()
        plt.title(self.name)



        
#%% Step Data Loading & Analysis
    
    def getStepData(self):
        # get all Step Data for a given subject
        
        Col = ['Subj','Speed','Cond','id', 'time', 'side', 'peakValue']
        AllStepData = pd.DataFrame(columns=Col)
        
        for s in self.speed:
            for c in self.conds[s]:
                subPath = os.path.join(self.path, s, c)
                StepFile = [x for x in os.listdir(subPath) if 'Step' in x][0]
                f = open(os.path.join(subPath, StepFile))
                J = json.load(f)
                f.close()
                StepData = pd.DataFrame(data=np.empty((len(J), len(Col))), columns=Col)
                
                for i in range(len(J)): # loop through json rows 
                    D = pd.json_normalize(json.loads(J[i]))
                    StepData.iloc[i] = [self.name, s.split('_')[-1], c.split('_')[-1],
                                       D.id.squeeze(), D.time.squeeze(), 
                                       D.side.squeeze(), D.peakValue.squeeze()]
                    
                AllStepData = pd.concat([AllStepData, StepData])
                
        self.StepData = AllStepData
        
        return AllStepData
    

#% Waveform Data Loading & Analysis
    # def getWaveData(self):
    #     # get all Wave Data for a given subject
        
    #     Col = ['Subj','Speed','Cond','id', 'time', 'side', 'peakValue']
    #     AllWaveData = pd.DataFrame(columns=Col)
        
    #     for s in self.speed:
    #         for c in self.conds[s]:
    #             subPath = os.path.join(self.path, s, c)
    #             WaveFile = [x for x in os.listdir(subPath) if 'Wave' in x][0]
    #             f = open(os.path.join(subPath, WaveFile))
    #             J = json.load(f)
    #             f.close()
    #             WaveData = pd.DataFrame(data=np.empty((len(J), len(Col))), columns=Col)
                
    #             for i in range(len(J)): # loop through json rows 
    #                 D = pd.json_normalize(json.loads(J[i]))
    #                 WaveData.iloc[i] = [self.name, s.split('_')[-1], c.split('_')[-1],
    #                                    D.id.squeeze(), D.time.squeeze(), 
    #                                    D.side.squeeze(), D.peakValue.squeeze()]
                    
    #             AllWaveData = pd.concat([AllWaveData, WaveData])
    #     return AllWaveData
    
    
    
#%% SQL functions

def pd_to_sql(input_df: pd.DataFrame,
                table_name: str,
                db_name: str = 'default.db') -> None:

    '''Take a Pandas dataframe `input_df` and upload it to `table_name` SQLITE table

    Args:
        input_df (pd.DataFrame): Dataframe containing data to upload to SQLITE
        table_name (str): Name of the SQLITE table to upload to
        db_name (str, optional): Name of the SQLITE Database in which the table is created. 
                                 Defaults to 'default.db'.
    '''

    # Get columns in the dataframe
    cols = input_df.columns
    cols_string = ','.join(cols)
    val_wildcard_string = ','.join(['?'] * len(cols))

    # Connect to a DB file if it exists, else crete a new file
    con = sqlite3.connect(db_name)
    cur = con.cursor()

    # # check to see if table exists
    # sql_string = f"""IF OBJECT_ID {table_name} IS NOT NULL THEN CREATE TABLE {table_name} ({cols_string})"""
    # cur.execute(sql_string)
    
    # drop table if it exists
    sql_string = f"""DROP TABLE IF EXISTS {table_name} """
    cur.execute(sql_string)

    # Create Table
    # sql_string = f"""CREATE TABLE IF NOT EXISTS {table_name} ({cols_string});"""
    sql_string = f"""CREATE TABLE {table_name} ({cols_string});"""
    cur.execute(sql_string)

    # Upload the dataframe
    rows_to_upload = input_df.to_dict(orient='split')['data']
    sql_string = f"""INSERT OR REPLACE INTO {table_name} ({cols_string}) VALUES ({val_wildcard_string});"""
    cur.executemany(sql_string, rows_to_upload)
  
    # Commit the changes and close the connection
    con.commit()
    con.close()



def sql_to_pd(sql_query_string: str, db_name: str ='default.db') -> pd.DataFrame:
    '''Execute an SQL query and return the results as a pandas dataframe

    Args:
        sql_query_string (str): SQL query string to execute
        db_name (str, optional): Name of the SQLITE Database to execute the query in.
                                 Defaults to 'default.db'.

    Returns:
        pd.DataFrame: Results of the SQL query in a pandas dataframe
    '''    
    # Step 1: Connect to the SQL DB
    con = sqlite3.connect(db_name)

    # Step 2: Execute the SQL query
    cursor = con.execute(sql_query_string)

    # Step 3: Fetch the data and column names
    result_data = cursor.fetchall()
    cols = [description[0] for description in cursor.description]

    # Step 4: Close the connection
    con.close()

    # Step 5: Return as a dataframe
    return pd.DataFrame(result_data, columns=cols)
   
    
#%% Acceleration Analysis

def GetGCAcc(FiltData, UnFiltData, PlotGCs=0):
    # get gait cycles from acceleration signal on ankle sensor
    
    D = FiltData.y.to_list()
    Time = FiltData.index.to_list()
    UF = UnFiltData.y.to_list()
    UFjerk = np.diff(UF)
    
    # peak-finding parameters
    # using vector mag acceleration at 100 Hz
    wid = 5
    p = 0.5
    ht = 1
    # rh = 0.5
    d = 10
    
    # find peaks, widths, and bases
    Pks, PkVals = signal.find_peaks(D, height=ht, prominence=p, width=wid, distance=d)
    # results = signal.peak_widths(D, Pks, rel_height=rh)
    # widths, width_heights, l_ips, r_ips = signal.peak_widths(D, Pks, rel_height=rh)
    
    # determine which peaks are foot offs vs foot strikes
    Strikes = []
    Offs = []
    for i, x in enumerate(Pks):
        if i == 0:
            continue
        if i == len(Pks) - 1:
            break
        DurBefore = x - Pks[i-1]
        DurAfter = Pks[i+1] - x
        if DurBefore > DurAfter:
            Offs.append(x)
        elif DurBefore < DurAfter:
            Strikes.append(x)
    
    # loop through strikes and look for jerk peak slightly after each strike
    SrchWin = 50 # number of frames for search window 
    ht = 0
    p = 0.1
    StrikeJ = []
    for x in Strikes: 
        PksJ, PkInfo = signal.find_peaks(UFjerk[x:x+SrchWin], height=ht, prominence=p)
        if len(PksJ) == 0:
            StrikeJ.append(x)
            continue
        StrikeJ.append(x + PksJ[0])
    
    # plot curve, peaks, starts & ends
    if PlotGCs == 1:
        plt.figure()
        plt.plot(D)
        plt.plot(UF)
        plt.plot(UFjerk)
        plt.plot(Strikes, [D[x] for x in Strikes], 'og')
        plt.plot(Offs, [D[x] for x in Offs], 'or')
        if len(StrikeJ) > 0:
            plt.plot(StrikeJ, [UFjerk[x] for x in StrikeJ], 'ok')
            plt.vlines(StrikeJ, 0, 3, 'k')
    
    
    # Get Gait Cycles (Foot strike to foot strike)
    # loop through rise cycles and interpolate & save
    Starts = []
    Ends = []
    L = len(StrikeJ)
    All = np.zeros([L, 100])
    a = 0
    for i in range(L-1):
        Raw = D[StrikeJ[i]:StrikeJ[i+1]]
        RawX = range(len(Raw))
        X = np.linspace(0, len(Raw), 100)
        
        if len(Raw) < 10: # exclude step if too short
          continue
    
        A = np.interp(X, RawX, Raw)
        
        # exclude invalid steps if: 
        if np.mean(A[10:20]) < 0.5 or np.mean(A[20:30]) < 0.5: # too low during loading peak
          continue
        if np.mean(A[30:40]) < 0.5: # too low during mid stance
          continue
    
        All[a, :] = A # otherwise save step
        Starts.append(StrikeJ[i])
        Ends.append(StrikeJ[i+1])
        a += 1
        del Raw, RawX, X, A
        
    ToDel = [x for x in range(L) if sum(All[x,:]) == 0]
    All = np.delete(All, ToDel, axis=0)
        
    if PlotGCs == 1:
        plt.figure()
        plt.plot(All.T)
    
    Avg = np.mean(100*All, axis=0)
    Std = np.std(100*All, axis=0)
    GC = {} # save data in dict for export
    GC['All'] = All
    GC['Avg'] = Avg
    GC['Std'] = Std
    GC['Start'] = Starts
    GC['End'] = Ends
    GC['Time'] = Time
    GC['StartTimes'] = [Time[x] for x in Starts]
    GC['EndTimes'] = [Time[x] for x in Ends]
    GC['Num_GCs'] = len(Starts)
    
    return GC

def GetGCAccSlow(FiltData, UnFiltData, PlotGCs=0):
    ''' get gait cycles from acceleration signal on ankle sensor, 
    altered for slow walking
    
    Inputs:  
        FiltData - filtered acceleration data 
        UnFiltData - unfiltered acceleration data 
        PlotGCs - optional logit to plot gait cycles (default=No)
        
    Outputs:
        GC - dictionary containing all gait cycles, start & end times, averages 
            and standard deviations, and more metrics of identified and parsed GCs
    
    '''
    
    D = FiltData.y.to_list()
    Time = FiltData.index.to_list()
    UF = UnFiltData.y.to_list()
    UFjerk = np.diff(UF)
    Ddiff = np.diff(D)
    A1 = 0.4
    
    # find flat spots in signal
    w = 20
    Thresh = 0.025
    Inds = []
    for i, x in enumerate(Ddiff):
        if i < 10:
            continue
        Vals = abs(Ddiff[i:i+w])
        Log = Vals < Thresh
        if False not in Log:
            Inds.append(i)
            
    
    # find starts of stance phases
    # where above threshold before and consecutive flats after
    StanceStarts = []
    for i in Inds: 
        if abs(Ddiff[i-1]) > Thresh:
            Vals = abs(Ddiff[i:i+20])
            Log = Vals < Thresh
            if False not in Log:
                StanceStarts.append(i)
    
    # peak-finding parameters
    ht = 0.12
    d = 15
    
    # find peaks right before stance phases
    Strikes = []
    for i in StanceStarts:
        Pks, PkVals = signal.find_peaks(UFjerk[i-20:i+5], height=ht, distance=d) 
        if len(Pks) > 0:
            Strikes.append(i+Pks[0]-20)

    # plot curve, peaks, starts & ends
    if PlotGCs == 1:
        plt.figure()
        plt.plot(D, label='Filt', alpha=A1)
        plt.plot(Ddiff, label='Filt Diff', alpha=A1)
        plt.plot(UF, label='UnFilt', alpha=A1)
        plt.plot(UFjerk, label='UnFilt Diff', alpha=A1)
        X = np.arange(0, len(D))
        plt.plot(X[Inds], Ddiff[Inds], 'og', label='Flats')
        plt.vlines(StanceStarts, -1, 3)
        plt.plot(Strikes, [UFjerk[x] for x in Strikes], 'ok', label='Strikes')
        plt.legend()
        
    
    # Get Gait Cycles (Foot strike to foot strike)
    # loop through rise cycles and interpolate & save
    Starts = []
    Ends = []
    L = len(Strikes)
    All = np.zeros([L, 100])
    a = 0
    for i in range(L-1):
        Raw = D[Strikes[i]:Strikes[i+1]]
        RawX = range(len(Raw))
        X = np.linspace(0, len(Raw), 100)
        
        if len(Raw) < 40: # exclude step if too short
          continue
        if len(Raw) > 250: # exclude step if too short
          continue
    
        A = np.interp(X, RawX, Raw)
        
        # exclude invalid steps if: 
        if np.mean(A[10:20]) < 0.5 or np.mean(A[20:30]) < 0.5: # too low during loading peak
          continue
        if np.mean(A[30:40]) < 0.5: # too low during mid stance
          continue
    
        All[a, :] = A # otherwise save step
        Starts.append(Strikes[i])
        Ends.append(Strikes[i+1])
        a += 1
        del Raw, RawX, X, A
        
    ToDel = [x for x in range(L) if sum(All[x,:]) == 0]
    All = np.delete(All, ToDel, axis=0)
        
    if PlotGCs == 1:
        plt.figure()
        plt.plot(All.T)
    
    
    Avg = np.mean(100*All, axis=0)
    Std = np.std(100*All, axis=0)
    GC = {} # save data in dict for export
    GC['All'] = All
    GC['Avg'] = Avg
    GC['Std'] = Std
    GC['Start'] = Starts
    GC['End'] = Ends
    GC['Time'] = Time
    GC['StartTimes'] = [Time[x] for x in Starts]
    GC['EndTimes'] = [Time[x] for x in Ends]
    GC['Num_GCs'] = len(Starts)
    
    return GC


def PredvGRF(W, L_GC, model, PlotPred=0):
    " Use ankle gait cycle metrics to parse waist acc and predict vGRF"

    # get waist data during each gait cycle
    All = np.zeros([L_GC['Num_GCs'], 100])
    Time = W.index.to_list()
    a = 0
    StartInds = []
    EndInds = []
    
    for i in range(L_GC['Num_GCs']):

        # get waist indicies from leg sensor tiem values
        a = np.argmin(abs(np.subtract(L_GC['StartTimes'][i], Time)))
        b = np.argmin(abs(np.subtract(L_GC['EndTimes'][i], Time)))
        Raw = list(W.y)[a:b]
        StartInds.append(a)
        EndInds.append(b)
        # Raw = W[L_GC['Start'][i]:L_GC['End'][i]]
        
        RawX = range(len(Raw))
        X = np.linspace(0, len(Raw), 100)
        if len(Raw) < 50: # exclude step if too short
            continue
        
        A = np.interp(X, RawX, Raw) # interpolate steps to 100 data points
        
        # exclude invalid steps if: 
        if np.mean(A[0:10]) < 5: # too low during loading peak
            continue
        if np.mean(A[10:20]) < 5 or np.mean(A[20:30]) < 5: # too low during loading peak
            continue
        if max(A) > 20: # exclude steps with too high of max acceleration 
            continue
        if max(A) < 10: # exclude steps with too low of max acceleration 
            continue
        
        
        All[i, :] = A # otherwise save step
        a += 1
        del Raw, RawX, X, A
        
    
    if a < L_GC['Num_GCs']:
      All = All[0:a, :]
    Avg = np.mean(100*All, axis=0)
    Std = np.std(100*All, axis=0)

    W_LGC = {} # save data in dict for export
    W_LGC['All'] = All
    W_LGC['Avg'] = Avg
    W_LGC['Std'] = Std
    W_LGC['StartInds'] = StartInds
    W_LGC['EndInds'] = EndInds
    W_LGC['StartTimes'] = [Time[x] for x in StartInds]
    W_LGC['EndTimes'] = [Time[x] for x in EndInds]
    W_LGC['Num_GCs'] = L_GC['Num_GCs']
    
    # estimate vGRF from waist accelerations
    Preds = np.zeros([W_LGC['Num_GCs'], 100])
    Peaks = np.zeros([W_LGC['Num_GCs'], 1])
    
    for i, x in enumerate(W_LGC['All']):
        X = np.array(x).reshape(-1,100)             # create array and reshape to match MLP inputs
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
        Preds[i, :] = model.run([label_name], {input_name: X})[0].squeeze() # predict vGRF from waist 
        Peaks[i] = np.max(Preds[i, :])
        
    W_LGC['Preds'] = Preds                          # save predicted vGRF waveforms
    W_LGC['PredAvg'] = np.mean(Preds, axis=0)
    W_LGC['PredStd'] = np.std(Preds, axis=0)
    W_LGC['Peaks'] = Peaks.squeeze()                # save peak vGRFs
    W_LGC['PeakAvg'] = np.mean(Peaks, axis=0)
    W_LGC['PeakStd'] = np.std(Peaks, axis=0)
    
    if PlotPred == 1:
        a1 = 0.25
        plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        # plot inputs and predictions
        ax1.plot(All.T, alpha=a1)
        ax2.plot(Preds.T, alpha=a1)
            
        ax1.set_title('Waist Inputs')
        ax2.set_title('vGRF Predictions')
        plt.show()
        
    return W_LGC


    
def GetGCs(D):
    # get gait cycles from vGRF signal
      
    # set peak-finding parameters
    wid = 10
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


# def StompAnalysis():
    
def CompileSteps(TimeData, AccStepTimes, AccData, TMStepTimes, TMData):
    """ loop through all valid steps and save into dataframe """
    
    cols = ['val' + str(x) for x in range(100)]
    AccDF = pd.DataFrame(columns=cols)
    TMDF = pd.DataFrame(columns=cols)
    
    if len(AccStepTimes) != len(TMStepTimes): 
        raise Exception('Step times are not the same length')
    
    for i in range(len(AccStepTimes) - 1):
        
        # exclude steps that are too short or long
        if TMStepTimes[i+1] - TMStepTimes[i] > 1.5:
            continue
        if TMStepTimes[i+1] - TMStepTimes[i] < 0.8:
            continue
        if AccStepTimes[i+1] - AccStepTimes[i] > 1.4:
            continue
        if AccStepTimes[i+1] - AccStepTimes[i] < 0.8:
            continue
        
        # exclude steps that have too few/many vGRF peaks
        a = TimeData.index(TMStepTimes[i])
        b = TimeData.index(TMStepTimes[i+1])
        p = 0.2
        ht = 0.9
        pks = signal.find_peaks(TMData[a:b], prominence=p, height=ht)
        if len(pks) > 2:
            # raise StopIteration
            continue
        
        # exclude steps that are too low near midstance
        # if np.mean(LvGRFsync[a:b])
        
        # save Acc Data
        a = TimeData.index(AccStepTimes[i])
        b = TimeData.index(AccStepTimes[i+1])
        Raw = list(AccData)[a:b]
        RawX = range(len(Raw))
        X = np.linspace(0, len(Raw), 100)
        if len(Raw) < 50: # exclude step if too short
            continue
        A = np.interp(X, RawX, Raw).tolist() # interpolate steps to 100 data points
        AccDF.loc[len(AccDF),:] = A # save into dataframe
        
        # save TM Data
        a = TimeData.index(TMStepTimes[i])
        b = TimeData.index(TMStepTimes[i+1])
        Raw = list(TMData)[a:b]
        RawX = range(len(Raw))
        X = np.linspace(0, len(Raw), 100)
        if len(Raw) < 50: # exclude step if too short
            continue
        A = np.interp(X, RawX, Raw).tolist() # interpolate steps to 100 data points
        TMDF.loc[len(TMDF),:] = A # save into dataframe
        
    return AccDF, TMDF


#%%
# old GC ID fxn for Acc signal


# def GetGCAcc(Data, PlotGCs=0):
#     # get gait cycles from acceleration signal on ankle sensor
    
#     D = Data.y.to_list()
#     Time = Data.index.to_list()
    
#     # peak-finding parameters
#     # using vector mag acceleration at 100 Hz
#     wid = 15
#     p = 0.8
#     ht = 0.95
#     rh = 0.95
#     d = 50
    
#     # find peaks, widths, and bases
#     Pks, PkVals = signal.find_peaks(D, height=ht, prominence=p, width=wid, distance=d)
#     # results = signal.peak_widths(D, Pks, rel_height=rh)
#     widths, width_heights, l_ips, r_ips = signal.peak_widths(D, Pks, rel_height=rh)
    
#     # plot curve, peaks, starts & ends
#     if PlotGCs == 1:
#         plt.plot(D)
#         plt.plot(Pks, [D[x] for x in Pks], 'om')
#         # plt.hlines(*results[1:], 'k')
#         plt.plot(l_ips, width_heights, 'og')
#         plt.plot(r_ips, width_heights, 'or')
    
#     Rising = [round(x) for x in l_ips]
#     Falling = [round(x) for x in r_ips]
#     Starts = []
#     Ends = []
    
#     # Get Stance Phases (heel strike to toe off)
#     # loop through rise cycles and interpolate & save
#     L = min(len(Rising), len(Falling))
#     All = np.zeros([L, 100])
#     a = 0
#     for i in range(L):
#         Raw = D[Rising[i]:Falling[i]]
#         RawX = range(len(Raw))
#         X = np.linspace(0, len(Raw), 100)
        
#         if len(Raw) < 10: # exclude step if too short
#           continue
    
#         A = np.interp(X, RawX, Raw)
        
#         # exclude invalid steps if: 
#         if np.mean(A[10:20]) < 0.5 or np.mean(A[20:30]) < 0.5: # too low during loading peak
#           continue
#         if np.mean(A[30:40]) < 0.5 or np.mean(A[45:55]) < 0.5: # too low during mid stance
#           continue
#         if np.mean(A[65:75]) < 0.5 or np.mean(A[75:85]) < 0.5: # too low during propulsive peak
#           continue
    
#         All[i, :] = A # otherwise save step
#         Starts.append(Rising[i])
#         Ends.append(Falling[i])
#         a += 1
#         del Raw, RawX, X, A
    
#     if a < L:
#       All = All[0:a, :]
#     Avg = np.mean(100*All, axis=0)
#     Std = np.std(100*All, axis=0)
#     Stance = {} # save data in dict for export
#     Stance['All'] = All
#     Stance['Avg'] = Avg
#     Stance['Std'] = Std
#     Stance['Start'] = Starts
#     Stance['End'] = Ends
#     Stance['Time'] = Time
#     Stance['StartTimes'] = [Time[x] for x in Starts]
#     Stance['EndTimes'] = [Time[x] for x in Ends]
#     Stance['Num_GCs'] = len(Starts)
    
#     return Stance
 
    
    