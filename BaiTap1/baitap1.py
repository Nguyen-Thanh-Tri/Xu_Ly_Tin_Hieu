
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def plot(path,i):
    file_path = path
    y, sr = librosa.load(file_path)
    win_len = int(5e-3*sr) #5ms
    hop_len = int(2e-3*sr) #2ms
    D = librosa.stft(y, n_fft=1024, hop_length=hop_len,win_length=win_len)
    D_log = librosa.amplitude_to_db(np.abs(D))
    plt.subplot(2,3,i)
    librosa.display.specshow(D_log, sr=sr,hop_length=hop_len,x_axis="time", y_axis='linear')
    plt.colorbar(format="%+2.f")
    

folder_path = "BaiTap1\signals"           
for speaker in os.listdir(folder_path):
    speaker_folder = os.path.join(folder_path, speaker)
    if os.path.isdir(speaker_folder):
        print(f"Currently selected: {speaker_folder}")
        fig, axs = plt.subplots(2,3,figsize=(10, 10),layout="constrained")
        i = 0
        for audio_file in os.listdir(speaker_folder):
            i+=1
            if audio_file.endswith('.wav'):
                audio_file_path = os.path.join(speaker_folder, audio_file)
                spectrogram = plot(audio_file_path,i)
                plt.title(f"{audio_file}")
        plt.show()