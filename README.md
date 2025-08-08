# 🔊 Speaker Verification System with MFCCs and Cosine Distance

A speaker verification system developed in a Google Colab notebook using Python. This project determines whether an audio sample belongs to a pre-enrolled "genuine" speaker or an "impostor." It implements a complete audio processing and machine learning pipeline from scratch, based on spectro-temporal feature extraction.

# 🚀 Open the Project in Google Colab

The easiest way to explore and run this project is through Google Colab. Click the badge below to open the notebook:
https://drive.google.com/drive/folders/1Ec72USQx-ec_2H2RIY5xFbwJt4xKq5tg?usp=sharing

To run this notebook successfully, you must have the audio dataset available in your Google Drive and connect the notebook to it.

- Dataset: You need the audio dataset used for this project. Download it here in previous link
- Upload to Google Drive: Unzip the dataset and upload the folder to your Google Drive. It is recommended to place it in a specific path, for example: My Drive/Colab Notebooks/SpeakerVerification/data.
- Connect to Drive in Colab: When you open the notebook, the first few code cells will be for mounting your Google Drive. Run these cells and follow the prompts to authorize the connection.
- Important: Make sure the file paths used throughout the notebook match the location where you stored the dataset in your Drive.
  
# 🎯 Project Objective

The primary goal is to build a voice biometric model capable of:

- Enrollment: Learning the unique vocal characteristics of a target speaker from several voice samples to create a reference "voiceprint" vector.
- Verification: Comparing a new audio sample against the reference voiceprint to decide if the voice belongs to the genuine speaker.
  
# 🔬 Implemented Methodology

The system represents each audio file as a single, 156-dimension feature vector. The workflow is illustrated below:

<img width="640" height="790" alt="image" src="https://github.com/user-attachments/assets/280cb54f-04a5-4b1a-971f-3f6dac320020" />

The process is divided into the following key steps:

- Audio Pre-processing: Each input .wav file is converted to mono, the signal is resampled to 16 kHz, a standard frequency for speech analysis, amplitude is normalized to ensure volume variations do not affect the results.
- Voice Activity Detection (VAD): The audio is segmented into 2-second chunks. The energy of each chunk is calculated, and the 3 most energetic segments are selected. This focuses the analysis on parts with clear speech, discarding silence or background noise.
- Feature Extraction: Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from each energetic segment. 26 MFCCs are calculated for each short analysis window. To capture the temporal dynamics of speech, the first and second derivatives (delta and delta-delta) are also computed, resulting in a 78-feature matrix per window (26 MFCCs + 26 deltas + 26 delta-deltas).
- Feature Vector Generation (156-D): Statistical functions are applied to the 78-feature matrix to summarize each segment into a single vector. The mean and standard deviation are calculated for each of the 78 features. This produces a final 156-dimension vector (78 means + 78 std devs) that uniquely represents a segment of speech.
- Final Representative Vector: The 156-D vectors from the 3 most energetic segments are averaged to obtain a single vector representing the entire audio file.
- Normalization and Comparison: All vectors are normalized using StandardScaler to remove scale-related biases.
- The similarity between two vectors is measured using cosine distance, which is robust to variations in recording volume as it measures orientation rather than magnitude.
- Classification (Speaker vs. Impostor): An optimal decision threshold is established by finding the intersection point of the distance distributions for genuine and impostor audios (minimizing classification errors). If a new audio's cosine distance to the reference vector is less than the threshold, it is classified as the Genuine Speaker. Otherwise, it's classified as an Impostor.

# 🛠️ Tech Stack
- Language: Python
- Core Libraries:
  - librosa: For audio processing (loading, resampling, MFCC extraction).
  - scikit-learn: For data normalization (StandardScaler).
  - numpy: For numerical computations and vector manipulation.
  - scipy: For finding the optimal threshold (e.g., using Brent's method).
  - matplotlib & seaborn: For visualizing the distance distributions.
    
# 👥 Author

Juan Pablo González Blandón

# 📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.
