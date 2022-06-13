'''
 (c) Copyright 2022
 All rights reserved
 Programs written by Yasser Abduallah
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA

 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.

 @author: Yasser Abduallah
'''

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    import tensorflow as tf    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')
import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta
from sklearn.metrics import brier_score_loss
from tensorflow.keras.regularizers import l1, l2
import os 
import time 
import math 

import shutil
from sklearn.metrics import confusion_matrix
import sys
from tensorflow import keras 
import keras.losses
import random
import glob 
from sklearn import metrics
import pickle

format_logging = True 
verbose = False 
g_verbose = False
t_window = ''
d_type = ''
log_handler = None
log_file = None
start_feature = 4
n_features = 15
series_len=10


tf_version = tf.__version__
print('Tensorflow bakcend version:',tf_version )
if tf.test.gpu_device_name() != '/device:GPU:0':
  print('WARNING: GPU device not found.')
else:
    print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices ) > 0:        
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
c_date = datetime.now()

def create_log_file(alg, t_window, d_type, dir_name='logs'):
    os.makedirs(dir_name, exist_ok=True)
    global log_handler
    global log_file
    try:
        log_file = dir_name + '/SEP_run_'  + str(c_date.month) + '-' + str(c_date.day) + '-' + str(c_date.year) +  '.log'
    except Exception as e:
        print('creating default logging file..')
        log_file = dir_name + '/SEP_run_'  + str(c_date.month) + '-' + str(c_date.day) + '-' + str(c_date.year) +  '.log'
    log_handler = open(log_file,'a')
    sys.stdout = Logger(log_handler)  
    log('')

class Logger(object):
    def __init__(self,logger):
        self.terminal = sys.stdout
        self.log = logger

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass  

def truncate_float(number, digits=4) -> float:
    try :
        if math.isnan(number):
            return 0.0
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper
    except Exception as e:
        return number

def set_verbose(b):
    global g_verbose
    g_verbose = b
    
def log(*message,verbose=verbose,format_logging=True, end=' '):
    log_str = []
    global g_verbose
    if verbose or g_verbose:
        if format_logging:
            print('[' + str(datetime.now().replace(microsecond=0))  +'] ', end='')
        for m in message:
            print(m,end=end)
        print('')
    log_handler.flush()
    
def parse_time(time):
    time = str(time).strip()
    time = time.replace('A','0').replace('90','09').replace('91','01').replace('U','0').replace('//','00')
    if '.' in time :
        time = time[:time.index('.')]
    
    time = time.replace('T',' ').replace('Z',':00')
    return datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
  
def boolean(b):
    if b == None:
        return False 
    b = str(b).strip().lower()
    if b in ['y','yes','ye','1','t','tr','tru','true']:
        return True 
    return False

def clean_entry(s):
    if '\n' in  str(s):
        return '"' + str(s).replace('\n',' ') + '"'
    try:
        if math.isnan(float(s)):
            return '0'
    except:
        return str(s)
    if str(s).strip() == '':
        return '0'
    return str(s)

def remove_new_line(s):
    if '\n' in  str(s):
        return '"' + str(s).replace('\n',' ').replace('"','') + '"'
    return str(s)

def get_data_with_filter(datafile, column=None, value=None,cols_list=None):
    # log('Loading data from file:' , datafile)

    if cols_list == None:
        return pd.read_csv(datafile)
    data =  pd.read_csv(datafile,usecols=cols_list)
    data = data.reindex(columns=cols_list)
    log('data.cols from list:', data.columns) 
    return data


def load_data(datafile, 
              series_len=series_len, 
              start_feature=4, 
              n_features=n_features,
              mask_value=0,
              stride=1,
              stride2=1, 
              column=None, 
              value=None,
              operator='eq' , 
              add_ar=False, 
              remove_positive=False,
              cols_list=None):
    log('Loading data from data file:', datafile)
    if not os.path.exists(datafile):
        log('Error Data file does not exist:', datafile)
        print('\nError: Data file does not exist:', datafile)
        exit()
    df = get_data_with_filter(datafile, column=column, value=value,cols_list=cols_list)
    if  column is not None and value is not None:
        log('Filtering data the data with filters: ' , column , value)
        if operator == 'eq':
            df = df.loc[df[column] == value].reset_index()
        else:
            df = df.loc[df[column].str.contains('|'.join(value)) ].reset_index()
    if remove_positive:
        log('Removing Positive labels for Rare Event Autoencoder, size before:', len(df))
        df = df.loc[df['Flare'] != 'P'].reset_index()
        log('Size after removing Positive:', len(df))
    return load_datasets(df, series_len, start_feature, n_features, mask_value, stride, stride2, column=None, value=None, add_ar=add_ar)

def load_datasets(df, 
                  series_len, 
                  start_feature, 
                  n_features, 
                  mask_value, 
                  stride, 
                  stride2, 
                  column=None, 
                  value=None, 
                  add_ar=False):
    if 'index' in df.columns:
        df = df.drop('index',axis=1)
    if column is not None and value is not None:
        df = df.loc[df[column] == value].reset_index()
    no_features=['TOTUSJZ','USFLUX','AREA_ACR','MEANALP']
    for f in no_features:
        df = df.drop(f,axis=1)
    df_values0 = df.values
    df_values = df_values0

    X = []
    y = []
    tmp = []
    for k in range(start_feature, start_feature + n_features):
        tmp.append(mask_value)
    n_neg = 0
    n_pos = 0
    for idx in range(0, len(df_values)):
        if np.mod(idx, stride2) != 0:
#             log('here')
            continue
        each_series_data = []
        row = df_values[idx]
        label = row[0]
        row_ar = row[2]
        if label == 'padding':
            continue
        has_zero_record = False


        if has_zero_record is False:
            cur_harp_num = int(row[3])
            each_series_data.append(row[start_feature:start_feature + n_features].tolist())
            itr_idx = idx - stride
            while itr_idx >= 0 and len(each_series_data) < series_len:
                prev_row = df_values[itr_idx]
                prev_harp_num = int(prev_row[3])
                if prev_harp_num != cur_harp_num:
                    break
                has_zero_record_tmp = False
                if len(each_series_data) < series_len and has_zero_record_tmp is True:
                    each_series_data.insert(0, tmp)

                if len(each_series_data) < series_len and has_zero_record_tmp is False:
                    each_series_data.insert(0, prev_row[start_feature:start_feature + n_features].tolist())
                itr_idx -= stride

            while len(each_series_data) > 0 and len(each_series_data) < series_len:
                each_series_data.insert(0, tmp)

            if (label == 'N' or label == 'P') and len(each_series_data) > 0:
                d1 = np.array(each_series_data).reshape(series_len, n_features).tolist()
                
                if add_ar:
                    d1.append(row_ar)
                    X.append([np.array(each_series_data).reshape(series_len, n_features).tolist(),row_ar])
                else:
                    X.append(np.array(each_series_data).reshape(series_len, n_features).tolist())
                if label == 'N':
                    y.append(0)
                    n_neg += 1
                elif label == 'P':
                    y.append(1)
                    n_pos += 1
    X_arr = np.array(X)
    y_arr = np.array(y)
    nb = {'Negative': n_neg ,'Positive': n_pos}
    return X_arr, y_arr, nb, df.columns[start_feature: start_feature + n_features]



 
def get_confusion_matrix_table(y_true,y_pred):
    if len(y_true) != len(y_pred):
        return 'Invalid samples, sizes are not equal'
    p=0                             
    n=0                             
    pt=0                            
    pf=0                            
    nt=0                            
    nf=0                            
    for i in range(0, len(y_true)): 
        if y_true[i] == 1:
            p += 1
        else:
            n +=1    
        if y_true[i] == y_pred[i]:
            if y_pred[i] == 1:
                pt +=1
            else:
                nt +=1
        else:
            if y_true[i] == 1 and y_pred[i] == 0:
                nf +=1
            if y_true[i] == 0 and y_pred[i] == 1:
                pf +=1
    TP=pt 
    TN = nt 
    FP = pf 
    FN = nf
    table_str = '\n***************************************** Start of Confusion Matrix **************************************'
    table_str = table_str + '\n\t\t\tActual\n\t\t\tP\tN' + '\nPredicted\tP\t' + str(pt) + '\t' + str( pf) + '\n\t\tN\t' + str(nf) + '\t' + str(nt) 
    table_str = table_str + '\n\nTotal\t\t\t' + str(p)+ '\t' + str(n) + '\n'
    table_str = table_str +  '\n***************************************** End of Confusion Matrix ****************************************\n'
    return table_str, TP, TN, FP, FN

def calc_confusion_matrix(y_true, 
                          y_pred, 
                          time_window,
                          e_type,
                          series_len=10,
                          log_to_file=True, 
                          cm_file='data/SEP_cm.csv',
                          predictions_file='result/SEP_predictions.csv',
                          epochs=10,
                          n_splits=None,
                          test_year=None,
                          ignore_low_tss=True,
                          probs_array=None,
                          probs_calibrated = None,
                          is_one_d=False,
                          sampling_type='',
                          n_features=15):
    
    cnn_type='BiLSTM'
    alg='BiLSTM'
    log('y_true distinct:', list(set(y_true)))
    log('y_pred distinct:', list(set(y_pred)))
    if len(list(set(y_pred))) == 1:
        #avoid any division by zero when it's in test mode.
        p0 = list(set(y_pred))[0]
        v = 1
        if p0 == 1:
            v = 0
        for s in range(10):
            y_pred[int(random.uniform(0, len(y_pred)))] = v
    
    pred_true_file = cm_file.replace('.csv','_true_pred_data.csv')
    cm = confusion_matrix(y_true, y_pred)

    cm_table_str, TP, TN, FP, FN = get_confusion_matrix_table(y_true,y_pred)
    log('CM:', cm, 'len(CM):', len(cm))
    log(cm_table_str)
    P = TP + FN 
    N = TN + FP
    T = N + P 
    accuracy = 0
    balanced_accuracy = 0 
    precision = 0 
    recall = 0 
    TSS = 0 
    TSS1 = 1
    HSS = 1 
    ApSS = 1
    TPR= 0
    FPR = 0
    GMEAN= 0
    BS = -1000
    BSS = -1000
    BSC = -1000
    BSSC = -1000
    
    try :
        if probs_array is not None:
            if is_one_d:
                b = probs_array[:]
            else:
                b = probs_array[:, 1]
             
            log('Calculating the BS')
            def getVal(i):
                if math.isnan(i):
                    i = 0                
                i = abs(i)
                if float(i) > 1:
                    i = 1
                if float(i) < 0:
                    i = 0
                return i
            b = [getVal(i) for i in b]
            BS = brier_score_loss(y_true, b)
            BS= truncate_float(BS)
            a = y_true
            m = np.array(a).mean()
            a_m = a - m
            a_square = np.square(a).sum()
            a_n = a_square / len(a)
             
            BSS = 1 - (BS/a_n)
            BSS = truncate_float(BSS)
        if probs_calibrated is not None:
            if is_one_d:
                b = probs_calibrated[:]
            else:
                b = probs_calibrated[:, 1]
             
            log('Calculating the BS')
            def getVal(i):
                if math.isnan(i):
                    i = 0
                i = abs(i)
                if float(i) > 1:
                    i = 1
                if float(i) < 0:
                    i = 0
                return i            
            b = [getVal(i) for i in b]
            BSC = brier_score_loss(y_true, b)
            BSC = truncate_float(BSC)
            a = y_true
            m = np.array(a).mean()
            a_m = a - m
            a_square = np.square(a).sum()
            a_n = a_square / len(a)
             
            BSSC = 1 - (BSC/a_n)
            BSSC = truncate_float(BSSC)                
                
        accuracy = (TP+TN) / (TP+FP+TN+FN)
        balanced_accuracy =  ( (TP/(TP + FN)) + (TN/(TN + FP)) ) /2
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        GMEAN = math.sqrt(TPR * FPR)        
        precision = TP / (TP + FP)
        recall = TP / (TP  + FN) 
        TSS = (TP/(TP+FN)) - (FP/(FP+TN))
        TSS1 = ((TP * TN) - (FP*FN))/(P*N)
        HSS = (2 * (TP * TN - FP * FN))/((TP + FN)*(FN+TN) + (TP + FP)*(FP+TN) )
      
        ApSS = (TP -  FP) / (TP + FN)

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        auc_val = truncate_float(metrics.auc(fpr, tpr))
        wauc = 'N/A'
        wauc = truncate_float(weighted_auc(np.array(tpr), np.array(fpr),alg))
        log('roc_curve thresholds:', thresholds)
        log('Epochs', epochs)
        log('SN',series_len)
        if n_splits is not None:
            log('NPS',n_splits)
        if test_year is not None:
            log("TY",test_year)
        log('T:', T) 
        log('N:', N) 
        log('P:', P)
        log('TP:',(TP)) 
        log('TN:',(TN)) 
        log('FP:', (FP)) 
        log('FN:', (FN)) 
        log('ACC:',truncate_float(accuracy)) 
        log('BAC:', truncate_float(balanced_accuracy))
        log('Pre:', truncate_float(precision)) 
        log('Rec:', truncate_float(recall)) 
        log('TSS:', truncate_float(TSS))
        log('TPR:', truncate_float(TPR))
        log('FPR:', truncate_float(FPR))
        log('GMEAN:', truncate_float(GMEAN))
        log('BS:', BS)
        log('BSS:', BSS)
        log('BSC:', BSC)
        log('BSSC:', BSSC)
        log('FPRA:', 'S'.join([str(truncate_float(s)) for s in fpr]))
        log('TPRA:', 'S'.join([str(truncate_float(s)) for s in tpr]))
        log('AUC:', auc_val)
        log('WAUC:', wauc)
        log('sampling_type:', sampling_type)
        log('n_features:', n_features)
        log('ApSS:', truncate_float(ApSS))
        p_dic = [{'Algorithm':cnn_type,
                  'eType':str(e_type), 
                  'SN':str(series_len),
                  'NF':n_features,
                  'TW':time_window,  
                  'Epochs':str(epochs),
                  'ST': str(sampling_type),
                  'TP': ( TP ),'TN': ( TN ),'FP': (  FP ),'FN': (  FN ),
                  'ACC': truncate_float( accuracy ),
                  'BACC': truncate_float(  balanced_accuracy),
                  'Pre': truncate_float(  precision ),
                  'Rec': truncate_float(  recall ),
                  'TSS': truncate_float(  TSS),
                  'HSS': truncate_float(HSS),
                  'ApSS': truncate_float(ApSS),
                  'T':T,
                  'P':P,
                  'N':N,
                  'FPR':truncate_float(FPR),
                  'TPR':truncate_float(TPR),
                  'GMEAN':truncate_float(GMEAN),
                  'BS':truncate_float(BS),
                  'BSS':BSS,
                  'BSC': BSC,
                  'BSSC':BSSC,
                  'FPRA':'S'.join([str(s) for s in fpr]),
                  'TPRA':'S'.join([str(s) for s in tpr]),
                  'AUC': truncate_float(auc_val),
                  'WAUC': wauc,
                  }]
    
        cols = ['Algorithm','eType','TW', 'T','P','N','TP','TN','FP','FN','ACC','BACC','Pre','Rec',
                'TSS','HSS','AUC','WAUC','BSS','BSSC']
                
        cols_print = cols[:]
        p_df = pd.DataFrame(p_dic,index=None, columns=cols)
        res_value = p_df.to_csv(sep='\t', index=False)
        
        p_df_print = pd.DataFrame(p_dic,index=None, columns=cols_print)
        res_value_print = p_df_print.to_csv(sep='\t', index=False)    
        log('Full Confusion Matrix Data Frame\n',res_value_print)
        
        log(cm_table_str)
        if log_to_file :
            os.makedirs(os.path.abspath(cm_file),exist_ok=True)
            log('Saving the performance metrics to files:',  cm_file,verbose=True)
            h = open(cm_file,'w')
            h.write(str(','.join(cols)).strip() + '\n')
            cm_to_write = p_df.to_csv(sep=',', index=False, header=None)
            h.write(str(cm_to_write).strip() + '\n')
            h.flush()
            h.close()
    except Exception as e:
        log('Unable to calculate metrics:',e)
        cols = ['Algorithm','eType','TW','T','P','N','TP','TN','FP','FN','ACC','BACC','Pre','Rec',
                'TSS','HSS','AUC','WAUC',
                'BSS','BSSC']        
        return {'TSS':-10000}, cols
    return p_dic[0], cols

def get_existing_model_tss(model_name, e_type, time_window,dir_name='models' ):
    glob_srch = dir_name+ os.sep + str(model_name) + '_model_' + str(e_type) + '_' + str(time_window) +'hr_tss_*'
    log('glob searching for:', glob_srch)
    files = []
    files_search = glob.glob(glob_srch)
    if int(tf_version[0]) > 1 :
        for f in files_search:
            if not str(f).endswith('.h5'):
                files.append(f)    
    if len(files) == 0:
        log('No model found for:', model_name, e_type, time_window)
        return ( -100.0, 'no_file_found')

    if len(files) > 1:
        log('Warning: more than one model found for:', model_name, e_type, time_window,'will load the first one in the list')
    tss = float(files[0][files[0].index('_tss_'):].replace('_'+str(time_window)+'_','').replace('.h5','').replace('_tss_',''))
    return (tss, files[0])

def delete_file(file_name):
    if os.path.exists(file_name):
        if os.path.isdir(file_name):
            shutil.rmtree(file_name,ignore_errors=True)
        else:
#             os.remove(file_name)
            shutil.rmtree(file_name,ignore_errors=True)
def delete_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name,ignore_errors=True)
    
def get_n_features_thresh( time_window):
    n_features = 18
    thresh = 0.55
    type='lstm'
    if type == 'gru':
        if time_window == 12:
            n_features = 16
            thresh = 0.45
        elif time_window == 24:
            n_features = 12
            thresh = 0.4
        elif time_window == 36:
            n_features = 9
            thresh = 0.45
        elif time_window == 48:
            n_features = 14
            thresh = 0.45
        elif time_window == 60:
            n_features = 5
            thresh = 0.5
    elif type == 'lstm':
        if time_window == 12:
            n_features = 15
            thresh = 0.4
        elif time_window == 24:
            n_features = 12
            thresh = 0.45
        elif time_window == 36:
            n_features = 8
            thresh = 0.45
        elif time_window == 48:
            n_features = 15
            thresh = 0.45
        elif time_window == 60:
            n_features = 6
            thresh = 0.5
        
    return n_features, thresh


def select_random_k(l, k):
    random.shuffle(l)
    result = []
    for i in range(0,  k):
        result.append(l[i])
    return result


def weighted_auc(tpr, fpr, alg='BiLSTM'):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights =        [       0.0002,   0.5]
        
    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    
    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)
    
    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        # print(mask)
        lower_b = 0
        if len(fpr[mask]) > 0: #for algorithms that are not perfect!!
            lower_b = fpr[mask][-1]
        x_padding = np.linspace(lower_b, 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric
        
    return (competition_metric / normalization)*0.9

def save_prediction_results(e_type, time_window, y_true,y_pred, y_calibrated_prop):
    predictions_file= 'results' + os.sep + 'SEP_prediction_results_'+ str(e_type) +'_' + str(time_window) +'.csv'
    log('Saving result to file:',predictions_file,verbose=True)
    h =open(predictions_file,'w')
    h.write('Label,Prediction,CalibratedProbability\n')
    for i in range(len(y_true)):
        h.write(str(y_true[i]) + ',' + str(y_pred[i])  + ',' + str(y_calibrated_prop[i])+ '\n')
    h.flush()
    h.close()

def print_summary_to_file(s):
    global log_file
    with open(log_file,'a') as f:
        print(s, file=f)
        
create_log_file('BiLSTM', '', '',dir_name='logs')
log('********************************  Executing Python program:', sys.argv[0].split(os.sep)[-1], '  ********************************',verbose=True)  
    