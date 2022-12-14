# machine-learning-predicting-text-to-speech-quality-using-brain-activity

[Predicting Text-To-Speech Quality using Brain Activity](https://ieeexplore.ieee.org/document/9975857) is a group research paper made for Machine Learning course in the fourth semester, which was published by [The 2022 International Conference on Internet of Things and Intelligence System (IoTaIS 2022)](http://iotais.org/) and posted to the IEEE Xplore digital library. Below is the abstract to the paper.

# ABSTRACT
With the importance of audio quality in developing a text-to-speech systems, it is important to conduct audio quality evaluation. Various methods have been developed in order to conduct audio quality evaluation, which are done either subjectively or objectively. Subjective methods require a large amount of time and resources, while objective methods lack human influence factors, which are important to a user’s perception of the experience offered by the audio quality. These human influence factors manifest inside an individual’s brain in forms such as electroencephalograph (EEG). In this study, we performed audio quality predictions using EEG data, which resulted in the proposed model yielding lower error distribution compared to other methods.

### Code Explanation
1. [featureExtraction](https://github.com/rhe-naldy/machine-learning-predicting-text-to-speech-quality-using-brain-activity/blob/main/featureExtraction.py)

In this file, we performed fast Fourier transform (FFT) in order to extract spectrogram from each EEG sample with a 1-second window and a 0.5 second overlap, followed by normalization with the highest frequency limited to 45 Hz.

2. [augmentedFeatureExtraction](https://github.com/rhe-naldy/machine-learning-predicting-text-to-speech-quality-using-brain-activity/blob/main/augmentedFeatureExtraction.py)

In this file, we augmented the raw EEG signal 30 times using Gaussian noise. Then, we performed fast Fourier transform (FFT) in order to extract spectrogram from each EEG sample with a 1-second window and a 0.5 second overlap, followed by normalization with the highest frequency limited to 45 Hz.

3. [peakPicking.py](https://github.com/rhe-naldy/machine-learning-predicting-text-to-speech-quality-using-brain-activity/blob/main/peakPicking.py)

In this file, we performed peak picking technique with a minimal horizontal distance of 18 in samples between neighboring peaks, followed by signal resampling and standardization of data.

4. [nonAugmentedCV.py](https://github.com/rhe-naldy/machine-learning-predicting-text-to-speech-quality-using-brain-activity/blob/main/nonAugmentedCV.py)

In this file, we constructed three machine learning models, consisting of SVR, MLP, and Decision Tree Regressor. We trained and tested the models independently using nested cross-validation by dividing the dataset into 4 different sets of data. Then, we tested every possible combination of train-test data with the train data using 3 sets of non augmented data and the remaining one set of non augmented data as the test data.

5. [augmentedCV.py](https://github.com/rhe-naldy/machine-learning-predicting-text-to-speech-quality-using-brain-activity/blob/main/augmentedCV.py)

In this file, we constructed three machine learning models, consisting of SVR, MLP, and Decision Tree Regressor. We trained and tested the models independently using nested cross-validation by dividing the dataset into 4 different sets of data. Then, we tested every possible combination of train-test data with the train data using 3 sets of augmented data and the remaining one set of non augmented data as the test data.
