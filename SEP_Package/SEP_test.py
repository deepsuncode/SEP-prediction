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

from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')
import numpy as np
from tensorflow import keras

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression as IR

from SEP_utils import *
from SEP_model import SEPModel 

time_windows = [12,24,36,48,60,72]
model_to_save = None
cm_target_dir='results'

#You may update the following line to:
#models_directory='default_models'
# to always predict using the default models.
models_directory=None

def test(e_type, start_hour, end_hour):
    for k in range(start_hour,end_hour,12):
        print('Running classification test type:', e_type,' training for h =', k, 'hour ahead')
        time_window = k 
        testing_data_file = 'data/events_' + str(e_type).replace('_S','').lower() + '_testing_' + str(time_window) + '.csv' 
        if not os.path.exists(testing_data_file):
            log('Error: testing data file does not exist:', testing_data_file)
            print('\nError: testing data file does not exist:', testing_data_file)
            exit()
                 
        model = SEPModel()
        log('Loading the model and its weights.', verbose=verbose)
        model.load_model(e_type=e_type,time_window=time_window, w_dir=models_directory)
        model.summary()
        n_features = model.input_shape[1]
        series_len = model.input_shape[0]
        x_test, y_test, nb_test, columns = load_data(datafile = testing_data_file, 
                                                     series_len=series_len, 
                                                     n_features=n_features)
        predictions_atten_proba = model.predict(x_test,verbose=0)
        predictions_atten_classes=(predictions_atten_proba> 0.5).astype("int32")         
        predictions = np.array(predictions_atten_classes).reshape(len(predictions_atten_classes)).tolist()
        
        log('Prediction and calibration..', verbose=verbose)
        predictions_proba = predictions_atten_proba
        calibration_curve(y_test, 
                          predictions_proba, 
                          n_bins=10, 
                          normalize=True)
        ir = IR()
        predictions_proba = predictions_proba.reshape(predictions_proba.shape[0])
        log('predictions_proba shape:', predictions_proba.shape, verbose=verbose)
        log('y_test shape: ', y_test.shape, verbose=verbose)
        ir.fit(predictions_proba,y_test)
        cal_pred = ir.predict(predictions_proba)
        calibration_curve(y_test, cal_pred,
                          n_bins=10, 
                          normalize=True)
        
        save_prediction_results(e_type, time_window, y_test, predictions, cal_pred)
        result, cols = calc_confusion_matrix(y_test,
                                             predictions, 
                                             time_window, e_type,epochs=0,test_year='',
                                             cm_file=cm_target_dir + os.sep  + 'SEP_performance_metrics_BiLSTM_'+ str(e_type) +'_' + str(time_window)  +'.csv',
                                             probs_array=predictions_proba[:],
                                             probs_calibrated = cal_pred, 
                                             is_one_d=True)


if __name__ == '__main__':
    starting_hour = 12
    ending_hour = 73
    e_type = 'FC_S'
    if len(sys.argv) < 2:
        print('Using default parameters: classification type is:', e_type, ' time windows to train:', time_windows)
    
    if len(sys.argv) >= 2  :
        e_type = sys.argv[1].strip().upper()
        if not e_type in ['FC_S', 'F_S']:
            print('Error: invalid classification type:', e_type,', must be one of:', ', '.join(['FC_S', 'F_S']))
            exit()
    if len(sys.argv) >= 3:
        starting_hour = int(float(sys.argv[2]))
        ending_hour = starting_hour + 1
    

    time
    if not starting_hour in time_windows:
        print('Invalid training hour:', starting_hour,'\nHours must be one of: ', time_windows)
        exit() 
    print('Starting hour:', starting_hour, 'ending hour:', ending_hour-1)
    test(e_type, starting_hour, ending_hour)