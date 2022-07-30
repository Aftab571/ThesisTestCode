from sys import settrace
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import gif

from json.tool import main
from memory_profiler import profile

import pandas as pd
from torch.nn import ReLU
from neo4j import GraphDatabase

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

import numpy as np
from statistics import mean
import plotly.express as px
from sklearn import preprocessing
import coloredlogs, logging
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader, NeighborLoader
from torch_geometric.nn import Linear, SAGEConv, Sequential, to_hetero, MetaPath2Vec
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv,GATv2Conv, Linear, SuperGATConv
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HGTConv, Linear, HeteroLinear
from torch_geometric.nn import GATConv, Linear, to_hetero
import torch.nn.functional as F
from tqdm import tqdm
import time
import wandb
from IPython.display import Image
import os
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel

pd.options.mode.chained_assignment = None  # default='warn'

torch.cuda.empty_cache()

mylogs = logging.getLogger(__name__)


class Connection:
    
    def fetch_data(self,query, params={}):
        with self.driver.session() as session:
            result = session.run(query, params)
            #return result
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def __init__(self):
        self.driver = GraphDatabase.driver("bolt://127.0.0.1:17687", auth=("neo4j", "123456"))

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), out_channels)
        self.lin1 = Linear(-1, out_channels)
        # convolution layers considers k-hops so more layers means more hops
        #self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=True)
        #self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index, return_attention_weights=True):
        x = F.dropout(x, p=0.6, training=True)
        x, alpha= self.conv1(x, edge_index,return_attention_weights=True) 
        x = x + self.lin1(x)
        x = F.relu(x)
        return F.log_softmax(x, dim=1),alpha



def create_train_test_mask(df):
    X = df.iloc[:,df.columns != 'label']
    Y = df['label']
    mask =[]
    #print(X)
    skf = StratifiedKFold(n_splits=10,random_state=1234, shuffle=True)
    for train_index, test_index in skf.split(X, Y):
        #print("Train_complete:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        #X_train, X_test, y_train, y_test = train_test_split(X,df['label'].values.tolist(), test_size=0.2,random_state=1,stratify=Y)
        train_mask = torch.zeros(df.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(df.shape[0], dtype=torch.bool)

        print("y_train: ",Counter(y_train).values()) 
        print("y_test: ",Counter(y_test).values()) 
    

        train_mask[X_train.index] = True
        test_mask[X_test.index] = True

        conf_df = pd.DataFrame()
        conf_df['admmision_id']= X_test['admmision_id']
        conf_df['actual']= y_test

        obj={
            'train_mask_set': train_mask,
            'test_mask_set' :  test_mask,
            'admission': conf_df
        }
        mask.append(obj)
    #print(mask)

    return mask


def create_train_val_test_mask(df):
    X = df.iloc[:,df.columns != 'label']
    Y = df['label']
   
  
 
    X_train_complete, X_test, y_train, y_test = train_test_split(X,df['label'].values.tolist(), test_size=0.2,random_state=1,stratify=Y)
    X_train, X_val, y_trainval, y_testval = train_test_split(X_train_complete,y_train, test_size=0.2,random_state=1,stratify=y_train)
    
    print("y_train: ",Counter(y_train).values()) 
    print("y_test: ",Counter(y_test).values()) 
    print("y_trainval: ",Counter(y_trainval).values()) 
    print("y_testval: ",Counter(y_testval).values()) 


    train_mask = torch.zeros(df.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(df.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(df.shape[0], dtype=torch.bool)

    train_mask[X_train.index] = True
    test_mask[X_test.index] = True
    val_mask[X_val.index] = True


    conf_df = pd.DataFrame()
    conf_df['admmision_id']= X_test['admmision_id']
    conf_df['actual']= y_test



    return train_mask,test_mask,val_mask,conf_df



def map_edge_list(lst1, lst2):
    final_lst=[]
    set1= set(lst1+lst2)

    i=0
    lst1_new={}
    for val in set1:
        lst1_new[val]=i
        i=i+1

    for x in range(len(lst1)):
        start= lst1_new[lst1[x]]
        end= lst1_new[lst2[x]]
        final_lst.append([start,end])
    
    return final_lst


def getData():
    conn = Connection()
    diagnosis_query="""MATCH (n:D_ICD_Diagnoses) where (n.long_title contains 'sepsis') or (n.long_title contains 'septicemia')  RETURN collect(n.icd9_code) as icd_arr""" 
    df_icd9 = conn.fetch_data(diagnosis_query)
    icd9_arr= df_icd9['icd_arr'][0]
    
    #hadm_query= """MATCH (n:Admissions)-[r:DIAGNOSED]-(m:D_ICD_Diagnoses) where m.icd9_code in """+str(icd9_arr)+""" RETURN n.hadm_id as hadm"""

    #hadm_query ="""MATCH (n:Note_Events) where n.category='Discharge summary' and ((toLower(n.text) contains toLower('sepsis')) or ((toLower(n.text) contains toLower('septic shock')) or ((toLower(n.text) contains toLower('severe sepsis')))))  RETURN collect(distinct n.hadm_id) as cols"""
    hadm_query= """MATCH (n:Note_Events) where n.category='Discharge summary' and (toLower(n.text) contains 'sepsis' or toLower(n.text) contains 'septic') RETURN collect(distinct n.hadm_id) as cols"""
    df_hadm = conn.fetch_data(hadm_query)
    hadm_arr=  df_hadm['cols'][0] #df_hadm['hadm'].tolist()  #

    hadm_arr = hadm_arr[0:10]

    print(hadm_arr)


    adm_pat_query = """MATCH (n:Admissions) where n.hadm_id in """+str(hadm_arr)+""" RETURN n.subject_id as patients, n.hospital_expire_flag as expire, n.hadm_id as hadm_id"""

    df_pat = conn.fetch_data(adm_pat_query)
    df_grp = df_pat.groupby(['expire'])


    # temp_lst =[]
    # for x in [0,1]:
    #     if x<=0:
    #         grp = df_grp.get_group(x)
    #         temp_lst.extend(grp['hadm_id'].values.tolist()[0:100])
    #     else:
    #         grp = df_grp.get_group(x)
    #         temp_lst.extend(grp['hadm_id'].values.tolist()[0:100])
    # for x in [0,1]:
    #     grp = df_grp.get_group(x)
    #     temp_lst.extend(grp['hadm_id'].values.tolist()[0:1882])
        
    #hadm_arr = temp_lst


    print(len(hadm_arr))
    pat_arr= df_pat['patients'].tolist()

    pat_dat_query = """ MATCH (n:Patients) where n.subject_id in """+str(hadm_arr)+""" RETURN n.gender as gender, n.dob as birth, n.subject_id as patient_id"""

    #df_pat_data = conn.fetch_data(pat_dat_query)

    df_diagnosis_query = """MATCH (n:Admissions)-[r:DIAGNOSED]->(m:D_ICD_Diagnoses) where n.hadm_id in """+str(hadm_arr)+""" RETURN n.hadm_id as hadm_id, n.hospital_expire_flag as expire, m.long_title as title"""

    df_diagnosis = conn.fetch_data(df_diagnosis_query)

    adm_query= """MATCH (n:Admissions) where n.hadm_id in """+str(hadm_arr)+""" RETURN n.subject_id as patients, n.hospital_expire_flag as label, n.marital_status as marital, n.ethnicity as ethnicity, n.religion as religion, n.hadm_id as hadm_id"""

    df_admission = conn.fetch_data(adm_query)
    

    # weights : and m.itemid in [50983,51221,50971,51249,51006,51265,50902,51301,50882,51250,50931,50912,51222,51279,51277,50868,51248,50960,50970,51237,51274,50893,51275,50804,50820,50821,50813,50818,50802]

    #and m.itemid in [50813,50931,50912,50868,50983,51237,51006,50885,50960,50902,50971,50820,50825]
    # high variance : [50813,50868,50885]
    #low variance: [50960,50971,50983]
    lab_query= """MATCH (x:Patients)-[xr:ADMITTED]-(n:Admissions)-[r:HAS_LAB_EVENTS]->(m:D_Lab_Items) where n.hadm_id in"""+str(hadm_arr)+""" and x.subject_id in """+str(pat_arr)+""" and duration.inSeconds(datetime(n.admittime), datetime(r.charttime)).hours<= 24 RETURN ID(n) as start, ID(m) as end, r.value as value, r.valueUOM as units, date(r.charttime) as lab_time, n.hospital_expire_flag as label, n.marital_status as marital, n.ethnicity as ethnicity, n.religion as religion, m.fluid as fluid, m.itemid as itemid, m.category as category, m.label as lab_label,m.label as lab_name, n.hadm_id as adm_id,n.hadm_id as hadm_id, x.gender as gender, date(x.dob) as birth, date(n.admittime) as admit, duration.inSeconds(datetime(n.admittime), datetime(r.charttime)).hours as diff""" 

    df_lab = conn.fetch_data(lab_query)
    
    

    drug_query= """MATCH (n:Admissions)-[r:PRESCRIBED]->(m:DRUGS) where n.hadm_id in """+str(hadm_arr)+""" RETURN  ID(n) as start, ID(m) as end, r.STARTDATE as drug_start_date, r.ENDDATE as drug_end_date, r.dosage_val as dosage_val, r.dosage_unit as dosage_unit, r.generic_name as generic_name, m.name as drug_name, duration.inSeconds(datetime(r.STARTDATE), datetime(r.ENDDATE)).hours as drug_duration, n.hadm_id as hadm_id"""

    df_drug =  conn.fetch_data(drug_query)

    return df_lab, df_drug, df_admission, df_diagnosis
    

def reverse_edge_index(data):

    cols = data.columns.tolist()

    rev_cols = reversed(cols)

    df1=data.reindex(columns=rev_cols)

    return df1.values.tolist()

def map_edge_list(lst1):
    final_lst=[]
    set1= set(lst1)

    i=0
    lst1_new={}
    for val in set1:
        lst1_new[val]=i
        i=i+1
    
    return lst1_new

def group_df(df_result):
    df_grp= df_result.sort_values(['value'],ascending=True).groupby(['start','end'])

    unique_start =  df_result['start'].unique()
    unique_end=  df_result['end'].unique()
    final_cols=[]
    final_cols.append('start')
    final_cols.append('end')
    final_cols.append('adm_id')
    final_cols.append('expire')
    df_final = pd.DataFrame(columns=final_cols)
    final_lst=[]
    for i,start in enumerate(unique_start) :
        for end in unique_end:
            try:
                obj= {}
                plt_result = df_grp.get_group((start,end))
                plt_result["value"]= plt_result["value"].astype(float)
                obj['value']= plt_result[["value"]].mean()['value']
                obj['lab_label'] = plt_result['lab_label'].tolist()[0]
                obj['label']= plt_result['label'].max() #.tolist()[0]
                obj['marital']= plt_result['marital'].tolist()[0]
                obj['ethnicity']= plt_result['ethnicity'].tolist()[0]
                obj['religion']= plt_result['religion'].tolist()[0]
                obj['category']= plt_result['category'].tolist()[0]
                obj['age']= plt_result['age'].tolist()[0]
                obj['gender']= plt_result['gender'].tolist()[0]
                obj['end']= end
                obj['start']= start
                final_lst.append(obj)
            

            except Exception as e:
                print(str(e))
                continue
    
            
        
    df_final = pd.DataFrame(final_lst)
    return df_final

def grp_labs(df):
    df['value'] = pd.to_numeric(df['value'], errors='coerce',downcast='float')
    df_grp = df.groupby(['adm_id','itemid']).agg({'start':'first','end':'first','value':'mean','units':'first','lab_time':'first','label':'first','marital':'first','ethnicity':'first','religion':'first','fluid':'first','itemid':'first','category':'first','lab_label':'first','adm_id':'first','gender':'first','birth':'first','admit':'first'})
    #df_grp = df.groupby(['adm_id','itemid'], as_index=False).sum()

    # unq_adm = df['adm_id'].unique()
    # unq_itm = df['itemid'].unique()

    # for i, adm in enumerate(unq_adm):
    #     for itm in unq_itm:
    #         try:
    #             df_result = df_grp.get_group((adm,itm))
    #             print(df_result['value'].mean())
    #          except Exception as e:
    #             print(str(e))
    #             continue

    return df_grp
    



def group_df_drugs(df_result):
    df_grp= df_result.sort_values(['dosage_val'],ascending=True).groupby(['start','end'])

    unique_start =  df_result['start'].unique()
    unique_end=  df_result['end'].unique()
    final_cols=[]
    final_cols.append('start')
    final_cols.append('end')
  
    df_final = pd.DataFrame(columns=final_cols)
    final_lst=[]
    print(unique_start)
    for i,start in enumerate(unique_start) :
        for end in unique_end:
            try:
                obj= {}
                #print(df_grp.get_group((start,end)))
                plt_result = df_grp.get_group((start,end))
                plt_result["dosage_val"]= plt_result["dosage_val"].astype(float)
                obj['dosage_val']= plt_result[["dosage_val"]].mean()['dosage_val']
                obj['drug_name'] = plt_result['drug_name'].tolist()[0]
                obj['end']= end
                obj['start']= start
                final_lst.append(obj)
            

            except Exception as e:
                print(str(e))
                continue
        
    df_final = pd.DataFrame(final_lst)
    return df_final

def folder(x,y):
    #timenow = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    folderpath = os.path.join(os.getcwd(), "mimicImages", str(x),str(y))
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        print('Created:', folderpath)
    return folderpath+"/"

def plotTimeCharts(df_conf):
    conn = Connection()
    df_normal = pd.read_csv("Lab_Normal_Range.csv")
    df_grp_typ= df_conf.groupby(['type'])
    unique_grp = df_conf['type'].unique()
    for x in unique_grp:
        print(x)
        plt_grp = df_grp_typ.get_group(x)
        hadm_arr = plt_grp['admmision_id'].unique().tolist()
        query = """Match (p:Patients)-[r1:ADMITTED]-(n:Admissions)-[r:HAS_LAB_EVENTS]-(m:D_Lab_Items) where n.hadm_id in 
        """+str(hadm_arr)+""" and duration.inSeconds(datetime(n.admittime), datetime(r.charttime)).hours<= 24*30
        return n.hadm_id as adm_id, m.itemid as itemid, m.label as label,m.category as category,m.fluid as fluid, 
        r.value as value, r.valueUOM as uom, r.charttime as time, n.admittime as admitTime, 
        duration.inSeconds(datetime(n.admittime), datetime(r.charttime)).hours/24 as diff, 
        p.subject_id as pat_id, p.gender as gender, n.hospital_expire_flag as expire"""

        #print(query)
        df = conn.fetch_data(query)
        #df.to_csv('timeplot.csv')
        df_grp= df.sort_values(['diff'],ascending=True).groupby(['itemid'])
        df_grp
        unique_adm_id =  df['adm_id'].unique()
        unique_lab=  df['itemid'].unique()
        #for i,pat in enumerate(unique_adm_id) :
        path = folder('time',x)

        for lab in unique_lab:
            try:
                fig = go.Figure()
                plt_result = df_grp.get_group((lab))
                unique_adm_id =  plt_result['adm_id'].unique()
                df_grp_adm= plt_result.groupby(['adm_id'])
                for i,pat in enumerate(unique_adm_id): 
                    #print(plt_result)
                    plt_result = df_grp_adm.get_group((pat))
                    #print(plt_result)
                    fig_df = plt_result[['adm_id','value','diff','expire']]
                    fig_df['value'] = pd.to_numeric(fig_df['value'], errors='coerce')
                    fig_df['adm_id'] = 'Pat - ' + fig_df['adm_id'].astype(str)
                    fig_df = fig_df.sort_values(['diff'],ascending=True)
                    #fig_df.set_index('diff', inplace=True)
                    if plt_result['label'].iloc[0].lower()=='creatinine':
                        text = plt_result['label'].iloc[0].lower()+plt_result['gender'].iloc[0].lower()
                    else:
                        text = plt_result['label'].iloc[0].lower()
                    df_normal_val = df_normal[df_normal['Lab'].str.lower()==text]
                    #print(fig_df['diff'])
                    #print(plt_result['label'].iloc[0].lower()+plt_result['gender'].iloc[0].lower())
                    if x=='TP':
                        color_val = 'blue'
                        tit_text ='Dead'
                    elif(x=='TN'):
                        color_val = 'red'
                        tit_text ='Dead'
                    else:
                        #print('expire',plt_result['expire'].iloc[0])
                        if plt_result['expire'].iloc[0] >0:
                            color_val = 'red'
                            tit_text ='Dead'
                        else:
                            color_val = 'blue'
                            tit_text ='Alive'
                    #print(plt_result['adm_id'].iloc[0])
                    fig.add_trace(go.Scatter(x=fig_df['diff'].values, y=fig_df['value'].values,name= fig_df['adm_id'].iloc[0],
                        line = dict(color=color_val, width=1), mode='lines+markers'
                            ))
                    fig.update_layout(title_text=str(plt_result['label'].iloc[0]), title_x=0.5,legend_title_text="Patient Admissions")
                    fig.update_xaxes(title_text='Days')
                    fig.update_yaxes(title_text='Lab Value ('+ str(plt_result['uom'].iloc[0])+")")

                #fig.add_figure(px.line(fig_df,x='diff',title=plt_result['label'].iloc[0]))
                #fig = px.box(fig_df, x="diff", y="value", title= plt_result['label'].iloc[0], color="expire")
                

            except Exception as e:
                print("Labs does not exists:",str(e))
                continue

            if(df_normal_val.shape[0]>0):
                fig.add_hrect(y0=df_normal_val['min'].iloc[0],y1=df_normal_val['max'].iloc[0], line_width=0, fillcolor="green", 
                            opacity=0.2,
                        annotation_text="Normal Range in "+str(plt_result['uom'].iloc[0]), 
                        annotation_position="bottom right",
                        annotation_font_size=20,
                        annotation_font_color="blue")
            img_bytes = fig.to_image(format="png", width=1000, height=350, scale=2)
            lab_name = ''.join(filter(str.isalnum, plt_result['label'].iloc[0]))
            fig.write_image(path+lab_name+".png",width=1000, height=350, scale=2)
            #fig.show()
                #break
            #break


#and m.itemid in [50813,50931,50912,50868,50983,51237,51006,50885,50960,50902,50971,50820,50825]
def make_charts(df_conf):
    conn = Connection()
    df_normal = pd.read_csv("Lab_Normal_Range.csv")
    df_grp_typ= df_conf.groupby(['type'])
    unique_grp = df_conf['type'].unique()
    for x in unique_grp:
        print(x)
        plt_grp = df_grp_typ.get_group(x)
        hadm_arr = plt_grp['admmision_id'].unique().tolist()
        query = """Match (p:Patients)-[r1:ADMITTED]-(n:Admissions)-[r:HAS_LAB_EVENTS]-(m:D_Lab_Items) where n.hadm_id in 
    """+str(hadm_arr)+"""  and duration.inSeconds(datetime(n.admittime), datetime(r.charttime)).hours<= 24*30
    return n.hadm_id as adm_id, m.itemid as itemid, m.label as label,m.category as category,m.fluid as fluid, 
    r.value as value, r.valueUOM as uom, r.charttime as time, n.admittime as admitTime, 
    duration.inSeconds(datetime(n.admittime), datetime(r.charttime)).hours as diff, 
    p.subject_id as pat_id, p.gender as gender, n.hospital_expire_flag as expire"""
        
        df = conn.fetch_data(query)
    
        df_grp= df.sort_values(['diff'],ascending=True).groupby(['itemid'])

        unique_lab=  df['itemid'].unique()
        path = folder('labVariation',x)
        # if not os.path.exists("/mimicImages/"+str(x)):
        #     os.mkdir(os.path.abspath(os.getcwd())+"\\mimicImages\\"+x)

        for lab in unique_lab:
            try:
                plt_result = df_grp.get_group(lab)
                plt_result['value'] = pd.to_numeric(plt_result['value'], errors='coerce')
                plt_result['adm_id'] = 'Pat - ' + plt_result['adm_id'].astype(str)
                if plt_result['label'].iloc[0].lower()=='creatinine':
                    text = plt_result['label'].iloc[0].lower()+plt_result['gender'].iloc[0].lower()
                else:
                    text = plt_result['label'].iloc[0].lower()
                df_normal_val = df_normal[df_normal['Lab'].str.lower()==text]
                if x=='TP':
                        color_val = 'blue'
                        tit_text ='Dead'
                elif(x=='TN'):
                    color_val = 'red'
                    tit_text ='Dead'
                else:
                    #print('expire',plt_result['expire'].iloc[0])
                    if plt_result['expire'].iloc[0] >0:
                        color_val = 'red'
                        tit_text ='Dead'
                    else:
                        color_val = 'blue'
                        tit_text ='Alive'
                #plt_result['value'] = plt_result['value'].astype(float)
                #print(plt_result['value'])
                fig = px.box(plt_result, x="adm_id", y="value", title= plt_result['label'].iloc[0], color='expire')
                if(df_normal_val.shape[0]>0):
                    fig.add_hrect(y0=df_normal_val['min'].iloc[0],y1=df_normal_val['max'].iloc[0], line_width=0, fillcolor="green", opacity=0.2,
                        annotation_text="Normal Range in "+str(plt_result['uom'].iloc[0]), 
                        annotation_position="bottom right",
                        annotation_font_size=20,
                        annotation_font_color="blue"
                        )
                #fig.show()
                img_bytes = fig.to_image(format="png", width=1000, height=350, scale=2)
                lab_name = ''.join(filter(str.isalnum, plt_result['label'].iloc[0]))
                fig.write_image(path+lab_name+".png",width=1000, height=350, scale=2)
                #Image(img_bytes)
            except Exception as e:
                print("Labs does not exists:",str(e))
                continue
            #break

@gif.frame
def make_heatMap(df,epoch):
    #print('df in heatmap:',df)
    temp_df = df[['hadm_id','lab_name','itemid','value','weights','diff']]
    #print(temp_df)

    df_grp = temp_df.groupby(['hadm_id','itemid'])
    unq_hadm= temp_df['hadm_id'].unique()
    #print(len(unq_hadm))
    unq_items= temp_df['itemid'].unique()
    heat_arr=[]
    for x in unq_hadm:
        obj= {}
        obj = {
                'hadm_id': 'Pat - '+str(x)
            }
        for y in unq_items:
            try:
                plt_result = df_grp.get_group((x,y))
                lab_text = plt_result['lab_name'].tolist()[0]
                #print(plt_result[["weights"]].max())
                obj[lab_text]= plt_result[["weights"]].max()['weights']

            except Exception as e:
                #print('novalues',str(e))
                continue
        heat_arr.append(obj)
                
    df_heat= pd.DataFrame(heat_arr)
    df_heat.set_index('hadm_id',inplace=True)
    #print(df_heat)

    scaler = MinMaxScaler() 
    df_scaled = scaler.fit_transform(df_heat.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=df_heat.columns)
    fig = px.imshow(df_scaled)
    #print(type(fig))
    #fig.show()
    path = folder('matrix','heatMapAtEpoch')
    fig.write_image(path+str(epoch)+".png",width=1000, height=350, scale=2)
    return fig

def create_heatMap_animation(maps):
    print(len(maps))
    path = folder('matrix','heatmap')
    gif.save(maps, path+"GATfImp.gif", 
         duration=15, unit="s", 
         between="startend")

def get_bert_based_similarity(text, model, tokenizer):
    """
    computes the embeddings of each sentence and its similarity with its corresponding pair
    Args:
        sentence_pairs(dict): dictionary of lists with the similarity type as key and a list of two sentences as value
        model: the language model
        tokenizer: the tokenizer to consider for the computation
    
    Returns:
        similarities(dict): dictionary with similarity type as key and the similarity measure as value
    """
    
    inputs_1 = tokenizer(text,padding=True, truncation=True, max_length = 500, return_tensors='pt')
    sent_1_embed = np.mean(model(**inputs_1).last_hidden_state[0].detach().numpy(), axis=0)
    return sent_1_embed

#@profile
def main():
    model = None
    prev_mask = None
    global dataset

    heatmaps=[]
 
    #transform = T.ToUndirected(merge=True)

    st_time_nodes = time.time()

    df_labs, df_drugs, df_admission, df_diagnosis = getData()
    print(df_admission)
    print(df_drugs)
    print(df_diagnosis)

    # enable to average the edges
    #df_labs = grp_labs(df_labs)



    end_time = time.time()

    print("Time for fetching data: ",end_time-st_time_nodes)

    df_labs = df_labs[df_labs['value'].notna()]

    df_labs['age'] = df_labs.apply(lambda e: (e['admit'] - e['birth']).days/365, axis=1)

     
    df_labs = df_labs[df_labs.age < 100 ]
    print(df_labs.shape)
    df_labs = df_labs[df_labs.age > 18 ]
    print(df_labs.shape)


    #print(df_admission)
    #print(df_labs)
    dict_start = map_edge_list(df_admission['hadm_id'].values.tolist())
   
    #vals = df_labs['adm_id'].values.tolist()

    df_admission['admmision_id'] = df_admission['hadm_id']
    #print(dict_start)
    #for x in dict_start:
    #df_labs= df_labs.replace({"adm_id": dict_start})
    #df_admission = df_admission.replace({"hadm_id": dict_start})

    

   
 
    
    label_encoder = preprocessing.LabelEncoder() 

    df_admission['ethnicity']= label_encoder.fit_transform(df_admission['ethnicity']) 
    df_admission['religion']= label_encoder.fit_transform(df_admission['religion']) 
    df_admission['marital']= label_encoder.fit_transform(df_admission['marital']) 

    df_labs['ethnicity']= label_encoder.fit_transform(df_labs['ethnicity']) 
    df_labs['lab_label'] = label_encoder.fit_transform(df_labs['lab_label'])
    df_labs['category'] = label_encoder.fit_transform(df_labs['category'])
    df_labs['marital']= label_encoder.fit_transform(df_labs['marital'])
    df_labs['gender']= label_encoder.fit_transform(df_labs['gender'])

    df_drugs['drug_name']  = label_encoder.fit_transform(df_drugs['drug_name'])

    

    df_labs['weight']= np.where(df_labs['label']<1, 1, 1)
    #print(df_labs)
    mask_list = create_train_test_mask(df_admission) #create_train_val_test_mask(df_admission)  #create_train_test_mask(df_admission) 
    df_labs["adm_id"]= df_labs["adm_id"].map(dict_start)
    df_drugs["hadm_id"]= df_drugs["hadm_id"].map(dict_start)
    df_diagnosis["hadm_id"]= df_diagnosis["hadm_id"].map(dict_start)
    df_admission["hadm_id"] = df_admission["hadm_id"].map(dict_start)

    df_admission.index = df_admission['hadm_id']
    df_admission.sort_index()

   
    df_labs= df_labs[pd.to_numeric(df_labs['value'], errors='coerce').notnull()]
    df_labs['value'] = df_labs['value'].astype(float)
    df_labs = df_labs.reset_index(drop=True)
    df_labs['index_col'] = df_labs.index
    df_admission['index_col'] = df_admission.index
    df_drugs['index_col'] = df_drugs.index
    df_diagnosis['index_col'] = df_diagnosis.index
    #print(df_admission)
    #print(df_labs)
    #'sum','mean','min','max','mul'
    #df_labs['weight']= df_labs.apply(lambda row: 100 if row.label>0 else 1, axis=1)
    
    #df_labs.loc[1 if df_labs['label']<1 else 3] = 1
    
    #df_labs = df_labs.sort_values(['label'])
    #print(df_labs)
    for aggr in ['sum']:
        test_acc_arr=[]
        train_acc_arr=[]
        for i in mask_list:
            #print(i)
            data = HeteroData()
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(device)
            
            data['Admission'].x = torch.tensor(df_admission[['ethnicity','marital','religion']].values, dtype = torch.float).to(device)
            data['Admission'].y =  torch.tensor(df_admission['label'].values, dtype = torch.long).to(device)
            data['Admission'].train_mask = i['train_mask_set'].to(device)
            data['Admission'].test_mask =  i['test_mask_set'].to(device)
            #data['Admission'].val_mask =  mask_list[2].to(device)
            data['Labs'].x = torch.tensor(df_labs[['category','lab_label','age','gender','value']].values, dtype = torch.float).to(device)
            data['Admission', 'has_labs', 'Labs'].edge_index = torch.tensor(df_labs[['adm_id','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            #data['Admission', 'has_labs', 'Labs'].edge_attr  = torch.tensor(df_labs[['value']].values, dtype=torch.long).t().contiguous().to(device)
            
    
        
            data.num_node_features = 3
            data.num_classes = len(df_labs['label'].unique())
            data = T.ToUndirected()(data.to(device))
            #data = T.RandomNodeSplit()
            data = T.NormalizeFeatures()(data.to(device))
            dataset = data.to(device)

            #print(data)
            data = dataset.to(device)
            if dataset:
            #['sum','mean','min','max','mul']
                #print(data.metadata())
                if model is not None:
                    #model = model
                    #model = to_hetero(model, data.metadata(), aggr=aggr).to(device)
                    model = GAT(hidden_channels=64, out_channels=data.num_classes).to(device)
                    model = to_hetero(model, data.metadata(), aggr=aggr).to(device)
                else:
                    model = GAT(hidden_channels=64, out_channels=data.num_classes).to(device)
                    model = to_hetero(model, data.metadata(), aggr=aggr).to(device)
                
                    
                
            
                print("model parameters is :", model.parameters())
                wandb.init(project="test-project", entity="master-thesis-luffy07")
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
                

                model.train()
                for epoch in range(20):
                    optimizer.zero_grad()
                    out,w = model(data.x_dict, data.edge_index_dict,True)
                    mask = data['Admission'].train_mask
                    loss = F.nll_loss(out['Admission'][mask], data['Admission'].y[mask])
                    wandb.log({"loss": loss})
                    #wandb.watch(model)
                    if epoch%1==0:
                        print(loss)
                    loss.backward()
                    optimizer.step()
                    if epoch%100==0:
                        df_links_0 = pd.DataFrame(w['Admission'][0][0].cpu().numpy(), columns=['src'])
                        df_links_1 = pd.DataFrame(w['Admission'][0][1].cpu().numpy(), columns=['dest'])
                        df_weights = pd.DataFrame(w['Admission'][1].cpu().detach().numpy(), columns=['weights'])
                        df_edge_links = pd.concat([df_links_0,df_links_1,df_weights],axis=1)
                        #print(df_edge_links.sort_values(by='src', ascending=False))
                        df3 = pd.merge(df_edge_links, df_labs[['itemid','hadm_id','index_col','lab_name','value','lab_time','diff']], left_on='src', right_on='index_col', how='left')
                        #print(df3)
                        #df3.to_csv('heatmap.csv')

                        #heatmaps.append(make_heatMap(df3,epoch))

                model.eval()
                torch.save(model, 'HGTmodel.pth')
                #create_heatMap_animation(heatmaps)
                #print(out['Admission'])
                pred = out['Admission'].argmax(dim=1)
                #print('edges',w['Admission'][0])
                #print('edges weights',w['Admission'][1])
                
                mask = data['Admission'].train_mask
                

                correct = (pred[mask] == data['Admission'].y[mask]).sum()
                train_acc = int(correct) / int(mask.sum())
                print(f'Training Accuracy: {train_acc:.4f}')
                train_acc_arr.append(train_acc)

                mask = data['Admission'].test_mask
                #print("test mask: ",mask)
                correct = (pred[mask] == data['Admission'].y[mask]).sum()
                test_acc = int(correct) / int(data['Admission'].test_mask.sum())
                print(f'Test Accuracy: {test_acc:.4f}')

                wandb.log({'test_acc':test_acc})  #, 'valid_acc':val_acc
                conf_df = i['admission']
                test_acc_arr.append(test_acc)
                #conf_df = conf_df.reset_index()
            
                #conf_df['actual']= data['Admission'].cpu().y[mask]
                conf_df['pred']= pred[mask].cpu()
                
                
            
                for i,row in conf_df.iterrows():
                    if row['actual'] == row['pred'] and row['actual']==0:
                        conf_df.at[i,'type']='TP'
                    elif row['actual'] == row['pred'] and row['actual']==1:
                        conf_df.at[i,'type']='TN'
                    elif row['actual'] != row['pred'] and row['actual']==1:
                        conf_df.at[i,'type']='FP'
                    else:
                        conf_df.at[i,'type']='FN'
                
                conf_df.to_csv('conf.csv')
            

                cf_matrix = confusion_matrix(data['Admission'].cpu().y[mask], pred[mask].cpu())

                #print(data['Admission'])
                #print("actual",data['Admission'].y)
                #print("pred",pred)

                print(cf_matrix)
                ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt= '.3g')

                ax.set_title('Confusion Matrix\n\n')
                ax.set_xlabel('\nPredicted Values')
                ax.set_ylabel('Actual Values ')
                ax.xaxis.set_ticklabels(['Survived','Died'])
                ax.yaxis.set_ticklabels(['Survived','Died'])
                fig = ax.get_figure()
                path = folder('matrix','confusion')
                fig.savefig(path+aggr+"confusion.png") 
                df_grp = df3.groupby(['itemid']).agg({'weights': 'sum','lab_name':'first','diff':'first'})
                
                #df_grp.set_index('diff',inplace=True)
                df_grp = df_grp.sort_values(['weights'],ascending=False)
                df_grp = df_grp[df_grp['weights']>=0.05]
                df_grp.to_csv('weights.csv')
                fig = px.bar(df_grp, x='lab_name', y='weights')
                path = folder('matrix','weightDistribution')
                fig.write_image(path+'weightPlot'+".png",width=1000, height=350, scale=2)
                #fig.show()
                #make_charts(conf_df)
                #plotTimeCharts(conf_df)

                #plt.savefig('confusion_matrix.png', dpi=400)
                #plt.show()
        print(test_acc_arr)
        print(train_acc_arr)
        print("Average of the Train Accuracy =", round((sum(train_acc_arr) / len(train_acc_arr)), 2))
        print("Average of the Test Accuracy =", round((sum(test_acc_arr) / len(test_acc_arr)), 2))
def seed_everything(seed=1234):                                                  
       #random.seed(seed)                                                            
       torch.manual_seed(seed)                                                      
       torch.cuda.manual_seed_all(seed)                                             
       np.random.seed(seed)                                                         
       os.environ['PYTHONHASHSEED'] = str(seed)                                     
       torch.backends.cudnn.deterministic = True                                    
       torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    #settrace()

    seed_everything()
    main()