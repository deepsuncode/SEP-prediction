This ReadMe explains the requirements and getting started to run the solar energetic particles (SEP) prediction tool using our BiLSTM network.

Prerequisites:

Python, Tensorflow, and Cuda:
The initial work and implementation of the BiLSTM network was done using Python version 3.9.7, Tensorflow 2.6.0 and GPU Cuda version cuda_11.4.r11.4.
Therefore, in order to run the default out-of-the-box models to make some predictions, you should use the exact version of Python and Tensorflow. 
Other versions are not tested, but they should work if you have the environment set properly to run deep learning jobs.

You may use the following link if still available in the Python download site to install Python version 3.9.7 for your operating system:
https://www.python.org/downloads/release/python-397/

Python Packages:
The following python packages and modules are required to run our BiLSTM network:
tensorboard==2.8.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow-estimator==2.8.0
tensorflow-gpu==2.6.0
numpy==1.19.5
pandas==1.4.1
keras==2.6.0
scikit-learn==1.0.1

To install the required packages, you may use Python package manager "pip" as follows:
1.	Copy the above packages into a text file,  i.e., "requirements.txt"
2.	Execute the command:
pip install -r requirements.txt
Note: There is a requirements file already created for you to use that includes all packages with their versions. 
       The files are located in the root directory of this SEP_Package.
Note: Python packages and libraries are sensitive to versions. Please make sure you are using the correct packages and libraries versions as specified above.

Cuda Installation Package:
You may download and install Cuda v 11.0 from https://developer.nvidia.com/cuda-11.0-download-archive

Package Structure
After downloading the files from github repository: https://github.com/deepsuncode/SEP-prediction the SEP_Package includes the following folders and files:
 
 ReadMe.txt                    - this ReadMe file.
 requirements.txt              - includes Python required packages for Python version 3.9.7.
 models                        - directory for newly trained models. 
 default_models                - includes default trained model used during the initial work of our BiLSTM network.
 logs                          - includes the logging information.
 data                          - includes a list of SEP data sets that can be used for training and testing/prediction.
 results                       - will include the prediction result file(s).
 SEP_test.py                   - Python program to test/predict a trained model.
 SEP_train.py                  - Python program to train a model and save it to the "models" directory.
 Other files are included as utilities files for training and testing.
 
Running a Test/Prediction Task:
To run a test/prediction, you should use the existing data sets from the "data" directory. 
 	SEP_test.py is used to run the test/prediction. 

Type: python SEP_test.py :
	Without any option will test all available time windows: 12, 24, 36, 48, 60, and 72 for the FC_S classification problem and save the prediction results to the "results" directory.

Type: python SEP_test.py FC_S 24 :
	provide a classification type and a time window, for example 24, and save the prediction results.
	Available classification types: FC_S and F_S
	Available time windows are: 12, 24, 36, 48, 60, and 72

Type: python SEP_test.py F_S :
		With only one option will test all available time windows: 12, 24, 36, 48, 60, and 72 for the F_S classification problem and save the prediction results to the "results" directory.
		Available classification types: FC_S and F_S
		
Type: python SEP_test.py F_S 36 :
	provide a classification type and a time window, for example F_S and 36, and save the prediction results.
	Available classification types: FC_S and F_S
	Available time windows are: 12, 24, 36, 48, 60, and 72

Note that the testing task will use the models from the "default_models" directory if you did not run a training task for the specified classification and time window.

Running a Training Task:
	SEP_train.py is used to run the training. 
	Examples to run a training job:
	python SEP_train.py	
	without any options to run a training job with default parameters to train all available hours 12 to 72 and save them to the "models" directory.

	python SEP_train.py F_S 72 
	provide a classification type and a time window, for example 72, and save the trained model to the "models" directory.
	Available classification types: FC_S and F_S
	Available time windows are: 12, 24, 36, 48, 60, and 72

