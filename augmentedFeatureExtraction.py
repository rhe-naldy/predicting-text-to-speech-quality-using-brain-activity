import scipy.io as sio
import scipy
from scipy import signal
import numpy as npy
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from joblib import Parallel, delayed
import multiprocessing

num = 10

for k in range(1, 31):
    std = num / 100
    noise = npy.random.normal(0, std, [44, 64, 7936])
    
    for x in range(8, 29):
        source_folder = 'Research-Materials/EEG/'
        target_folder = 'Research-Materials/Augmented-EEG-Spectrogram/'
        
        data = sio.loadmat(source_folder + 'sub' + str(x) + '_noCSD.mat')
        
        indexes = [i[0][0][0] for i in data["EEG"][0][0][25][0] if i[0][0][0] in npy.arange(5, 49)]
        records = data["EEG"][0][0][15].transpose(2,0,1)[1:-1]
        
        records = records + noise
        
        def generate_spectrogram(sample):
            sample_spectrogram = []
            for c in sample:
                f, t, Sxx = scipy.signal.spectrogram(c, fs = 256, window = ('boxcar'), noverlap = 128)
                sample_spectrogram.append(normalize_ch(Sxx[:45]))
        
            return npy.array(sample_spectrogram)
        
        def normalize_ch(ch_spec):
            scaler = StandardScaler()
            return scaler.fit_transform(ch_spec)
        
        def save_record(sample, track, x):
            pickle.dump(sample, open(target_folder + "augmented_" + str(x) + "_" + str(track) + "_" + str(k) + ".pkl", "wb"))
            pass
        
        def prep(sample, index):
            sample_spectrogram = generate_spectrogram(sample)
            save_record(sample_spectrogram, index, x)
            
        preprocess_eeg = Parallel(n_jobs = multiprocessing.cpu_count())(delayed(prep)(sample, index) for sample, index in zip(records, indexes))

    num += 1