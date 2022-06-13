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
from tensorflow import keras
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression as IR
from sklearn.utils import class_weight
from SEP_utils import *
from SEP_model import SEPModel 

time_windows = [12,24,36,48,60,72]
model_to_save = None


#Training parameters. 
#The number of iterations is set to 5 for testing and verification puropose, 
# for full training , it should be set to at least 100, but it will take a very long time to finish.
iterations = 5

total_epochs =  iterations



cm_target_dir='results'
prev_tss = -200000.0
current_tss = -20000.0
tss_threshold = 0.99

verbose=False

def train(e_type, start_hour, end_hour):
    dir_name='temp'
    delete_dir(dir_name)
    for k in range(start_hour,end_hour,12):
        prev_tss = -200000.0
        time_window = k       
        print('\n\nRunning classification type:', e_type,' training for h =', k, 'hour ahead')
        training_data_file = 'data/events_' + str(e_type).replace('_S','').lower() + '_training_' + str(time_window) + '.csv'
        testing_data_file = 'data/events_' + str(e_type).replace('_S','').lower() + '_testing_' + str(time_window) + '.csv'         
        if not os.path.exists(training_data_file):
            log('Error: required training data file does not exist:', training_data_file)
            print('\nError: required training data file does not exist:', training_data_file)
            exit()

        if not os.path.exists(testing_data_file):
            log('Error: required testing data file does not exist:', testing_data_file)
            print('\nError: required testing data file does not exist:', testing_data_file)
            exit()
                        
        x_train, y_train, nb_train, columns = load_data(datafile= training_data_file)
        
        
        x_test, y_test, nb_test, columns = load_data(datafile= testing_data_file)
        x_train_orig, y_train_orig = x_train[:], y_train[:] 
        log('nb_train:', nb_train)
        log('x_train.shape:', x_train.shape)
        log('y_train.shape:', y_train.shape)
        log('x_test.shape:', x_test.shape)
        log('y_test.shape:', y_test.shape)
        log('nb_test:', nb_test)
        x_train_ex = np.append(x_train, x_test)
        s1 = len(x_train_ex)//(x_train.shape[1]*x_train.shape[2])
        s2 = x_train.shape[1]
        s3 = x_train.shape[2]
        x_train = x_train_ex.reshape((s1,s2,s3))
        # x_train= x_train_ex
        y_train = np.append(y_train,y_test)
        log('x_train.shape:', x_train.shape)
        log('y_train.shape:', y_train.shape)
        log('x_test.shape:', x_test.shape)
        log('y_test.shape:', y_test.shape)   
    
        
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_values = {0: class_weights[0], 1: class_weights[1]}
        cnn_type = 'BiLSTM'
        input_shape = (x_train.shape[1], x_train.shape[2])

        models = []
        c_iter = 0
        tss = -20000
        epoch = 1
        for i in range(0,iterations):
            c_iter = (i+1)
            print('Epoch', str(epoch) +'/' + str(total_epochs))
    
            x_train = x_train_orig
            y_train = y_train_orig
            train_shape_orig = x_train.shape 
            nsamples, nx, ny = x_train.shape
            x_train = x_train.reshape((nsamples,nx*ny))
            
            x_train = x_train.reshape(x_train.shape[0], train_shape_orig[1], train_shape_orig[2])
            
            model = SEPModel()
            model.build_base_model(input_shape )
            model.models()
            model.compile()
            model.summary()
            
            history = model.fit(x_train, y_train)
            model.save_weights(e_type,time_window,w_dir=None)
            models.append(model)
            epoch +=1
            predictions_atten_proba = model.predict(x_test)
            predictions_atten_classes=(predictions_atten_proba> 0.5).astype("int32")             
            predictions = np.array(predictions_atten_classes).reshape(len(predictions_atten_classes)).tolist()
            
            predictions_proba = predictions_atten_proba
            fop, mpv = calibration_curve(y_test, predictions_proba, n_bins=10, normalize=True)
            ir = IR()
            predictions_proba = predictions_proba.reshape(predictions_proba.shape[0])
            log('predictions_proba shape:', predictions_proba.shape)
            log('y_test shape: ', y_test.shape)
            ir.fit(predictions_proba,y_test)
            cal_pred = ir.predict(predictions_proba)
            fop, mpv = calibration_curve(y_test, cal_pred, n_bins=10, normalize=True)

            result, cols = calc_confusion_matrix(y_test,predictions, 
                                                 time_window, e_type, cnn_type,epochs=total_epochs,test_year='',log_to_file=False,
                                                 cm_file=cm_target_dir + os.sep + 'SEP_cm_BiLSTM_'+ str(e_type) +'_' + str(time_window)  +'.csv',
                                                 probs_array=predictions_proba[:],
                                                 probs_calibrated = cal_pred, is_one_d=True)

            if float(result['TSS']) >= tss_threshold:
                log('Breaking iterations at #:', (i+1))
                model_to_save = model
                tss = float(result['TSS']) 
                model.save_weights(e_type,time_window,w_dir=None)
                break
            current_tss = float(result['TSS'])
            if current_tss > prev_tss:
                model_to_save = model
                tss = current_tss
                        
            
            file_ext = '.h5'
            if int(tf_version[0]) > 1 :
                file_ext = ''
            alg ='bilstm'
            model_name = dir_name + '/' + alg +'_model_' + str(e_type) + '_' + str(time_window) + 'hr_tss_' + str(tss) +  str(file_ext)
            log('Saving the final model to:', model_name , ' with TSS values: prev_tss = ', prev_tss, 'current_tss:', current_tss, 'at iteration #:', c_iter,verbose=verbose)
            model_saved_tss, prev_file_name = get_existing_model_tss(alg,e_type, time_window, dir_name=dir_name)
            log('prev_file_name for tss:', prev_file_name,verbose=verbose)
            if tss > model_saved_tss:
                log('Saving the model because it has greater than tss previous value, tss:', tss, 'model_saved_tss:', model_saved_tss,verbose=verbose)
                delete_file(prev_file_name)
                model_to_save.save(model_name)
                model.save_weights(e_type,time_window,w_dir=None)
            else:
                log('The model has less tss than or equal previous value, will not save it, tss:', tss, 'model_saved_tss:', model_saved_tss,verbose=verbose)
            prev_tss  = float(result['TSS']) 
    delete_dir(dir_name)

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
    

    if not starting_hour in time_windows:
        print('Invalid training hour:', starting_hour,'\nHours must be one of: ', time_windows)
        exit() 
    print('Starting hour:', starting_hour, 'ending hour:', ending_hour-1)
    train(e_type, starting_hour, ending_hour)