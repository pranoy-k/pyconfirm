"""
Various tools for interfacing meteor confirmation models with CAMS I/O files.
"""

import numpy as np
import pandas as pd
import glob
import pickle

from keras.models import load_model
from sklearn.metrics import r2_score
from statsmodels import robust
from scipy.stats import skew, kurtosis

# fields = mat['a'][0].dtype.names
FIELDS = [
    'FFname',
    'CALname',
    'FFfolder',
    'CALfolder',
    'cameranumber',
    'detectnumber',
    'nsegments',
    'framerateHz',
    'hnr',
    'mle',
    'bin',
    'pixelsperfrm',
    'houghrho',
    'houghphi',
    'frameno',
    'colcent',
    'rowcent',
    'radeg',
    'decdeg',
    'azdeg',
    'eldeg',
    'inten'
]

def read_detect_info(fn, n_col=3, nhed=10, na=10, nb=8):
    """
    Reads an FTP_detectinfo file from the CAMS software pipeline.
    """
    with open(fn, 'r') as f:
        #Read number of meteor tracks
        n = np.int(f.readline().split()[n_col])

        #Read header
        for i in np.arange(nhed):
            if i == 3:
                ff_folder = f.readline().split()[3]
            elif i == 4:
                cal_folder = f.readline().split()[3]
            else:
                f.readline()

        #Loop over tracks
        ds = []
        for i in np.arange(n):
            f.readline()

            #Make dictionary for this track
            d = {}

            d[FIELDS[0]] = f.readline().split()[0]
            d[FIELDS[1]] = f.readline().split()[0]
            d[FIELDS[2]] = ff_folder
            d[FIELDS[3]] = cal_folder

            line = f.readline().split()

            for j in np.arange(na):
                if j < 3:
                    d[FIELDS[j + 4]] = np.int(line[j])
                else:
                    d[FIELDS[j + 4]] = np.float(line[j])

            #Initiliaze time-series arrays with nsegments
            for j in np.arange(nb):
                d[FIELDS[j + 4 + na]] = np.zeros(d['nsegments']).tolist()

            for j in np.arange(d['nsegments']):
                line = f.readline().split()
                
                for k in np.arange(nb):
                    d[FIELDS[k + 4 + na]][j] = np.float(line[k])

            #Add track to list
            ds.append(d)

    return ds

def write_detect_info(data, outfn, infn=None, ftpfilter='FTP*info*txt', nhed=11):
    """
    Takes an FTPdetectinfo dataset as a python list and writes out the CAMS format.
    """
    #First we need to read an input file to get the header
    #If not specified, just pick the first one we find
    if infn is None:
        infn = glob.glob(ftpfilter)[0]

    firstline = 'Meteor Count = {0:0>6d}\n'.format(len(data))
        
    with open(infn, 'r') as infile:
        head = [next(infile) for x in xrange(nhed)]

    head[0] = firstline

    #Now get output file ready and write the header
    with open(outfn, 'w') as outfile:
        for line in head:
            outfile.write(line)

        #Now loop through each tracklet in data and write it to the file in series
        for tracklet in data:
            write_tracklet(tracklet, outfile)

def write_tracklet(tracklet, f, 
                   meta_line=['{0:0>4d} ', '{0:0>4d} ', '{0:0>4d} ', '{0:0>7.2f} ', '{0:0>5.1f} ', 
                              '{0:0>5.1f} ', '{0:0>5.1f} ', '{0:0>5.1f} ', '{0:0>6.1f} ', '{0:0>6.1f}\n'],
                   data_line=['{0:0>6.1f} ', '{0:0>7.2f} ', '{0:0>7.2f} ', '{0:0>6.2f} ', '{0:0>6.2f} ',
                              '{0:0>6.2f} ', '{0:0>6.2f} ', '{0:0>6.0f}\n']):
    """
    Writes a tracklet to a file in FTPdetectinfo format.

    Parameters:
    ------------
    tracklet: dict, contains the full tracklet information as created by read_detect_info
    f: file object already opened for writing
    """
    #Write separator line
    f.write('-------------------------------------------------------\n')
    
    #Write the FF and Cal files
    f.write(tracklet['FFname'] + '\n')
    f.write(tracklet['CALname'] + '\n')

    #Write metadata line
    for i, col in enumerate(meta_line):
        f.write(col.format(tracklet[FIELDS[4 + i]]))

    #Loop through segments in tracklet and write
    for i in np.arange(tracklet['nsegments']):
        seg = get_segment(tracklet, i)
        
        for j, col in enumerate(data_line):
            f.write(col.format(seg[j]))

def get_segment(tracklet, i):
    """
    Utility function that gets the ith segment in a tracklet.
    """
    seg = []

    for field in FIELDS[14:]:
        seg.append(tracklet[field][i])

    return seg

def distill_detect_info(data):
    """
    Takes a detectinfo dataset makes a dataframe with summary statistics of light curve and trajectory
    """
    #Get number of rows
    n = len(data)
    
    #Create a new dataframe to hold this distilled data
    df = pd.DataFrame()
    
    #Add columns for each feature
    #Slope of XY trajectory
    df['Slope_XY'] = np.zeros(n)
    #RMS scatter about linear fit to traj
    df['RMS_XY'] = np.zeros(n)
    #R^2 of linear fit to traj
    df['R2_XY'] = np.zeros(n)
    #Distance traveled in pix
    df['dXY'] = np.zeros(n)
    #Avg mag
    df['Mean_inten'] = np.zeros(n)
    #Std of mag
    df['Std_inten'] = np.zeros(n)
    #MAD of mag
    df['MAD_inten'] = np.zeros(n)
    #Skew of mag
    df['Skew_inten'] = np.zeros(n)
    #Kurtosis of mag
    df['Kurtosis_inten'] = np.zeros(n)
    #Time elapsed
    df['dt'] = np.zeros(n)
    #Period as calculated by FFT
    df['Period'] = np.zeros(n)

    #Loop through data frame rows
    for j, tracklet in enumerate(data):
        row = tracklet
        
        #First let us fit a line to the XY path
        #The trajectory should be basically straight, so this should be okay
        x = row['rowcent']
        y = row['colcent']
        fit = np.polyfit(x, y, 1)
    
        #Calculate residuals
        ypred = np.polyval(fit, x)
        res = y - ypred
    
        #RMS of the residuals
        rms = np.std(res)
    
        #R^2
        r2 = r2_score(y, ypred)
    
        #Slope of the fit
        slope = fit[0]
        
        #Also calculate the Euclidean distance traversed in pixels
        dxy = np.linalg.norm(np.array([x[0], y[0]]) - np.array([x[-1], y[-1]]))
    
        #Now let's summarize the light curve
        i = row['inten']
        #Get the elapsed time from the frame rate * nobs
        dt = row['nsegments'] / row['framerateHz'] / 2. #Use 2*framerate to account for the interleaving reading
        
        #Average mag
        m = np.mean(i)
        #Std mag
        s = np.std(i)
        #MAD
        mad = robust.mad(i)
        #Skew
        sk = skew(i)
        #Kurtosis
        kurt = kurtosis(i)
        
        #Finally, calculate an FFT of the light curve and take the strongest period from that
        #Calculate the Fourier transform of the data
        fft = np.fft.fft(i)

        #Calculate the power 
        p = np.square(np.abs(fft))

        #Normalize the power
        p /= np.abs(fft[0])

        #Calculate the frequencies of the FFT
        freq = np.fft.fftfreq(row['nsegments'], 1 / row['framerateHz'] / 2.)
  
        #The peak frequency is then the maxi'th entry in freq
        #Find where the peak power is
        
        try:
            maxp = np.max(p[1 : np.int(np.ceil(row['nsegments']/2.) + 1)])
        except TypeError, ValueError:
            print(row['nsegments'])
            print(p)
            print(p[1 : np.int(np.ceil(row['nsegments'] / 2.) + 1)])
            return p

        #Find what index in the array this corresponds to
        maxi = np.where(p == maxp)[0][0]

        #The peak frequency is then the maxi'th entry in freq
        maxf = freq[maxi]

        per = 1. / maxf

        #Add everything to the DF
        df.loc[j, 'Slope_XY'] = slope
        df.loc[j, 'RMS_XY'] = rms
        df.loc[j, 'R2_XY'] = r2
        df.loc[j, 'dXY'] = dxy
        df.loc[j, 'dt'] = dt
        df.loc[j, 'Mean_inten'] = m
        df.loc[j, 'Std_inten'] = s
        df.loc[j, 'MAD_inten'] = mad
        df.loc[j, 'Skew_inten'] = sk
        df.loc[j, 'Kurtosis_inten'] = kurt
        df.loc[j, 'Period'] = per
       
    return df

def extract_time_series(data, zeropad=True, max_length=402):
    """
    Extracts the time-series trajectory and photometric data from a series of FTP_detect objects stored in a JSON DF.
    """
    xs = []
    ys = []
    ts = []
    ints = []

    #Loop through tracklets
    for j, tracklet in enumerate(data):
        row = tracklet
        
        #Get the X, Y, and intensity
        #For positions, start everything at 0
        x = np.array(row['rowcent']) - row['rowcent'][0]
        y = np.array(row['colcent']) - row['colcent'][0]
        i = row['inten']
        
        #Figure out the times (normalized to t=0)
        #Get the time step
        dt = 1. / row['framerateHz'] / 2. #Use 2*framerate to account for the interleaving reading

        t0 = row['frameno'][0]

        t = [(frmno - t0) * dt for frmno in row['frameno']]

        #Pad with zeros if desired
        if zeropad:
            x = np.pad(x, [0, max_length - np.size(x)], 'constant', constant_values=0.)
            y = np.pad(y, [0, max_length - np.size(y)], 'constant', constant_values=0.)
            i = np.pad(i, [0, max_length - np.size(i)], 'constant', constant_values=0.)
            t = np.pad(t, [0, max_length - np.size(t)], 'constant', constant_values=0.)

        xs.append(x)
        ys.append(y)
        ts.append(t)
        ints.append(i)
        

    return xs, ys, ts, ints

def read_rf(mfn):
    """
    Reads a pickled Random Forest meteor classifier model.

    Returns:
    --------
    rf: a scikit-learn GridSearchCV object with estimator=RandomForestClassifier
    pr_rf, rec_rf: Precision and recall of the classifier on test set
    randst: int, the RGN seed
    test_size: float 0-1, the fraction of the dataset used as the test set
    ncv: Number of folds used for the CV grid search
    scaler: a scikit-learn Scaler object used to standardize the training data for this model
    npv, spv, skv, pdv: the versions of numpy, scipy, sklearn, and pandas used when training this model
    """
    with open(mfn, 'rb') as f:
        [rf, [pr_rf, rec_rf], [randst, test_size, ncv], scaler,
                     [npv, spv, skv, pdv]] = pickle.load(f)

    return rf, [pr_rf, rec_rf], [randst, test_size, ncv], scaler, [npv, spv, skv, pdv]

def get_maxn_from_model(model):
    """
    Gets the input length for the time-series for this model.
    """
    return model.get_input_shape_at(0)[1]

def read_lstm(mfn, sfn='scaler.pkl'):
    """
    Reads the model and scaling parameters for the LSTM for meteor classification.
    """
    model = load_model(mfn)

    #Read scaler
    with open(sfn, 'rb') as f:
        mean, std = pickle.load(f)

    #Get time-series length
    maxn = get_maxn_from_model(model)

    return model, mean, std, maxn

def standardize(indata, scaler=None, mean=None, std=None, maxn=32):
    """
    Standardizes meteor tracklet data, either using a sklearn scaler or the provided mean and std.
    """
    if scaler is not None:
        df = distill_detect_info(indata)

        #Take the logs of the features with intensity units
        df['logMean_inten'] = np.log10(df['Mean_inten'])

        #Note that the Std and MAD of some objects are 0 which are bad for logarithms. So before we take the log, let's replace these with the minimum.
        df.loc[df['MAD_inten'] == 0, 'MAD_inten'] = np.min(df.loc[df['MAD_inten'] > 0, 'MAD_inten'])
        df.loc[df['Std_inten'] == 0, 'Std_inten'] = np.min(df.loc[df['Std_inten'] > 0, 'Std_inten'])
        
        df['logMAD_inten'] = np.log10(df['MAD_inten'])
        #I think the Std and MAD are essentially the same thing, so I am not sure if keeping Std is adding much info
        df['logStd_inten'] = np.log10(df['Std_inten'])

        #Drop unnecessary features
        df = df.drop(['Mean_inten', 'MAD_inten', 'Std_inten', 'Slope_XY'], 1)

        #Standardize all the things
        df = scaler.transform(df)
    else:    
        ts = extract_time_series(indata)
        x, y, t, i = ts

        df = np.dstack([x, y, t, i])
        df = df[:, :maxn, :]

        df = (df - mean) / std

    return df

def classify(data, model, batch_size=None):
    """
    Feeds FTP detect info into a classifier model.
    """
    if batch_size is not None:
        labels = model.predict_proba(data, batch_size)
    else:
        labels = model.predict_proba(data)

    return labels
