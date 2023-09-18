#%% Show Vetta Data 

import panel as pn
import hvplot.pandas
import numpy as np
import pandas as pd
import VettaUtils as VU
import datetime

from bokeh.plotting import figure

# get SQL database
# sqlDB = r'C:\Users\richa\Documents\Packages\Vetta\Database\VettaPilot.db'
sqlDB = r'C:\Users\richa\Documents\Packages\Vetta\Database\VettaPilot_Pred.db'

D = VU.AllData()
sqlTableNames = D.getTableNames()
userList = []
speedList = []
condList = []

for x in sqlTableNames: 
    S = x.split('_')
    userList.append(S[0])
    speedList.append(S[1])
    condList.append(S[2])
    
Users = np.unique(userList).tolist()
Speeds = np.unique(speedList).tolist()
Conds = np.unique(condList).tolist()

# create dict of speeds and conditions for each user?


#%% Sensor Inputs Page

selectInfo = pn.Column(
    '## Select Raw Data', 
     )

# column widget to select data
user = pn.widgets.Select(name="User", options=Users, value=Users[1])
speed = pn.widgets.Select(name="Speed", options=Speeds, value=Speeds[0])
cond = pn.widgets.Select(name="Condition", options=Conds, value=Conds[0])

def loadSensor(user, speed, cond, sensorID):
    # download sql data based on selections
    table_name = '_'.join([user, speed, cond])
    sqlStr = f"SELECT * FROM {table_name}"
    df = VU.sql_to_pd(sqlStr, sqlDB)
    DF = df[df.id == sensorID]
    # transform time from unix standard to real time
    DF['TimeStr'] = [datetime.datetime.fromtimestamp(x) for x in DF.time]
    DF.sort_values(by='TimeStr', inplace=True)
    Start = DF.TimeStr.iloc[0]
    Elapsed = []
    for x in DF['TimeStr']:
        ts = x - Start
        Elapsed.append(ts.seconds + ts.microseconds / 1000000)
        
    # filter data ?
    
    return pd.DataFrame(dict(y=DF['accel'].to_list()), index=Elapsed)

def loadPred(user, speed, cond, side):
    # download predictions from sql data based on selections
    tableName = '_'.join([user, speed, cond]) + '_' + side + '_GC'
    sqlStr = f"SELECT * FROM {tableName}"
    df = VU.sql_to_pd(sqlStr, sqlDB).T.reset_index(drop=True)
    
    C = ['Step' + str(x) for x in range(len(df.columns))]
    df.columns = C
    
    # remove zero predictions
    ToDrop = []
    for i, x in enumerate(df.sum(axis=0)):
        if x == 0:
            ToDrop.append(C[i])
    df = df.drop(columns=ToDrop)
    
    # remove erroneous predictions
    ToDrop = []
    r, c = np.shape(df)
    for i in range(c):
        if max(df.iloc[60:,i]) > 0.3:
            ToDrop.append(df.columns[i])
    df = df.drop(columns=ToDrop)

    return df



#%% Process Sensor Data

# create data processing info, button, and 
proc = pn.Column(pn.layout.Divider(), 
                 '## Process Data', 
                 'press this button to process the selected data'
                 )

processButton = pn.widgets.Button(name='Process Sensor Data', 
                                  button_type='primary', 
                                  button_style='solid', 
                                  )

L_Steps = 0
R_Steps = 0

# data processing function
def ProcessSensorData(user, speed, cond):
    
    # L = loadSensor(user, speed, cond, '34')
    # R = loadSensor(user, speed, cond, '35')
    # W = loadSensor(user, speed, cond, '36')
    
    # L_GC = VU.GetGCAccFoot(L)
    # R_GC = VU.GetGCAccFoot(R)
    
    # L_steps = L_GC['Num_GCs']
    # R_steps = R_GC['Num_GCs']
    
    processButton.button_style='outline'
    
    L_Steps =  40
    R_Steps = 40

    return L_Steps, R_Steps


processButton.on_click(ProcessSensorData(user, speed, cond))

procText = pn.Column('L Steps:  ' + str(L_Steps), 
                     'R Steps:  ' + str(R_Steps))



#%% Build Template

# MAIN PAGE 1
dfi_waist = hvplot.bind(loadSensor, user, speed, cond, '36').interactive()
dfi_lAnk = hvplot.bind(loadSensor, user, speed, cond, '34').interactive()
dfi_rAnk = hvplot.bind(loadSensor, user, speed, cond, '35').interactive()

plot_opts1 = dict(responsive=True, min_height=200, min_width=800)

# Instantiate the template with widgets displayed in the sidebar
template = pn.template.GoldenTemplate(
    title='Vetta Data',
    sidebar=[selectInfo, user, speed, cond, proc, processButton, procText],
    row_height=100,
    sidebar_width=25,
    )


# Populate the main area with plots, to demonstrate the grid-like API
# add sensor Inputs
template.main.append(
    pn.Column(
        pn.Card(dfi_waist.hvplot(**plot_opts1).output(), title='Waist'), 
        pn.Card(dfi_lAnk.hvplot(**plot_opts1).output(), title='Left Ankle'), 
        pn.Card(dfi_rAnk.hvplot(**plot_opts1).output(), title='Right Ankle'), 
        name='Sensor Inputs'
        )
    )



# MAIN PAGE 2
# dfi_L = hvplot.bind(loadPred, user, speed, cond, 'L').interactive()
# dfi_R = hvplot.bind(loadPred, user, speed, cond, 'R').interactive()
dfi_L = hvplot.bind(loadPred, user, speed, cond, 'L').interactive()
dfi_R = hvplot.bind(loadPred, user, speed, cond, 'R').interactive()

# PredL = loadPred(user.value, speed.value, cond.value, 'L')
# a1 = 0.4
# p = figure(width=600, height=300)
# p.line(PredL.iloc[0,:])


plot_optsL = dict(responsive=True, min_height=250, min_width=500, 
                  xlim=(0,100), legend=None)
plot_optsR = dict(responsive=True, min_height=250, min_width=500)

# extracting waveforms
template.main.append(
    pn.Column(
        pn.Card(dfi_L.hvplot(**plot_optsL).output(), title='Left'), 
        pn.Card(dfi_R.hvplot(**plot_optsR).output(), title='Right'),
        # pn.Card(dfi_L.hvplot().output(), title='Left'), 
        # pn.Card(dfi_R.hvplot().output(), title='Right'),
        name='vGRFs'
        )
    )


template.header_color = '#151943'
template.servable();