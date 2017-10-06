"""
Runs an FTPdetectinfo file through the confirmation filters and outputs a new FTPdetectinfo file with the meteor tracklets.
"""

import glob

from ftpinfo_io import *

#Params
ftpfilter = '*FTP*info*txt'
batch_size = 32 #Batch size for LSTM
rfmn = 'rf.pkl' #File name for pickled Random Forest
lstmmn = 'lstm.h5' #File name for pickled Random Forest
prob_thresh = 0.5

#Get all desired files
files = glob.glob(ftpfilter)

#Loop through all files
for f in files:
    #Read the data
    data = read_detect_info(f)
    
    #Read the random forest model
    rf, [pr_rf, rec_rf], [randst, test_size, ncv], scaler, [npv, spv, skv,pdv] = read_rf(rfmn)

    #Read the LSTM model
    lstm, mean, std, maxn = read_lstm(lstmmn)

    #Standardize data for RF
    rfdf = standardize(data, scaler)

    #Standardize data for LSTM
    lstmdf = standardize(data, mean=mean, std=std, maxn=maxn)

    #Predict meteor probabilities
    #Note: for RF, sklearn outputs probabilities for both classes. In our case, class 1 (meteor) is the second class, so take just that one
    rf_prob = np.expand_dims(classify(rfdf, rf)[:, 1], 1)
    lstm_prob = classify(lstmdf, lstm, batch_size)

    probs = np.hstack([rf_prob, lstm_prob])

    #Average probs from each model
    prob = np.mean(probs, axis=1)

    #Finally, write new FTPdetectinfo files for just meteors
    meteors = np.where(prob >= prob_thresh)[0]

    write_detect_info(np.array(data)[meteors], f[:-4] + '_cleaned.txt', f)
