import os
import librosa
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

data_dir = ['/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/01MDA',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/02FVA',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/03MAB',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/04MHB',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/05MVB',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/06FTB',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/07FTC',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/08MLD',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/09MPD',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/10MSD',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/11MVD',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/12FTD',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/14FHH',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/15MMH',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/16FTH',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/17MTH',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/18MNK',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/19MXK',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/20MVK',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/21MTL',
            '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmKiemThu-16k/22MHL',]
train_dir = ['/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/23MTL',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/24FTL',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/25MLM',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/27MCM',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/28MVN',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/29MHN',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/30FTN',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/32MTP',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/33MHP',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/34MQP',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/35MMQ',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/36MAQ',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/37MDS',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/38MDS',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/39MTS',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/40MHS',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/41MVS',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/42FQT',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/43MNT',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/44MTT',
             '/Users/thanhhuong/Desktop/XLTH/BT2/NguyenAmHuanLuyen-16k/45MDV']
N_FFTs = [512, 1024, 2048]
vowels = ['a', 'e', 'i', 'o', 'u']
chiso = 0.1
a_vectors = []
e_vectors = []
i_vectors = []
o_vectors = []
u_vectors = []
a_vector = []
e_vector = []
i_vector = []
o_vector = []
u_vector = []

def get_frames(signal, samplerate):
    frame_size = int(20e-3 * samplerate)
    frame_shift = int(20e-3/2 * samplerate)
    # Function to split the signal into frames
    nsamples = len(signal)
    count = 1+ (nsamples - frame_size) // frame_shift
    frames = np.zeros((count, frame_size))
    for i in range(count):
        start = i * frame_shift
        frames[i, :] = signal[start: start + frame_size]
    return frames
# lay nguyen am
def segment_vowel(signal, samplerate):
    energy = []
    frames = get_frames(signal, samplerate)
    row, col =frames.shape
    for j in range(row):
        frame_energy = np.sum(frames[j]**2)
        energy.append(frame_energy)
    threshold = chiso * np.max(energy)
    vowel_segments = []
    vowel_segments = np.array([frames[i] for i, value in enumerate(energy) if value > threshold])
    return vowel_segments
#ham lam minj frames
def hamming_window(length):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(length) / (length - 1))
#lay cung on dinh
def get_stable_region(vowel_segments, samplerate):
    vowel_segment = vowel_segments.flatten()
    segment_len = len(vowel_segment)
    size = segment_len // 3
    vowel_segment = vowel_segment[:3*size]
    region = vowel_segment[size:2 * size]
    stable_region = get_frames(region, samplerate)
    M = stable_region.shape[0]
    for j in range(M):
         wind = hamming_window(len(stable_region[0, :]))
         stable_region[j, :] = stable_region[j, :] * wind
    return stable_region, M
#trich fft
def get_fft(signal, N_FFT, samplerate):
    # Tim nguyen am
    vowel_segments = segment_vowel(signal, samplerate)
    
    # Tim vung on dinh
    stable_region, M = get_stable_region(vowel_segments, samplerate)
    
    #vecto dac trung
    char_vector = np.zeros((N_FFT//2+1,M))
    for j in range(M):
        fft_signal = fft(stable_region[j], n=N_FFT)
        char_vector[:,j] = np.abs(fft_signal)[:N_FFT // 2 + 1]
        
    feature_vector = np.mean(char_vector, axis=1)   
    return feature_vector
#so khop 
def predict_vowels(fft, avg_vector, name, vowels):
    min_dist = float("inf")
    pred_label = 'unknown'
    
    for label, train_vector in zip(vowels, avg_vector):
        dist = euclidean_dist(fft, train_vector)
        if dist < min_dist:
            min_dist = dist
            pred_label = label
    return name, pred_label
def euclidean_dist(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

  # Pad the shorter vector with zeros
    if len(vec1) < len(vec2):
        vec1 = np.pad(vec1, (0, len(vec2) - len(vec1)))
    elif len(vec2) < len(vec1):
        vec2 = np.pad(vec2, (0, len(vec1) - len(vec2)))
    return np.sqrt(np.sum((vec1 - vec2)**2))

def ghi(N_FFT, path):
    for subfolder_path in path:
        for vowel in vowels:
            filepath = os.path.join(subfolder_path, vowel + '.wav')               
            #lay tin hieu, toc do lay mau        
            signal, samplerate = librosa.load(filepath)
            char_vector = get_fft(signal, N_FFT, samplerate)                       
            if vowel == 'a':
                a_vectors.append(char_vector)
            elif vowel == 'e':
                e_vectors.append(char_vector)
            elif vowel == 'i':
                i_vectors.append(char_vector)
            elif vowel == 'o':
                o_vectors.append(char_vector)
            elif vowel == 'u':
                u_vectors.append(char_vector)                               
                           
    return a_vectors, e_vectors, i_vectors, o_vectors, u_vectors
def avarage(a_vectors, e_vectors, i_vectors, o_vectors, u_vectors):
    max_len = max(len(a) for a in a_vectors + e_vectors + i_vectors + o_vectors + u_vectors)
    # Tính trung bình cộng
    a_vectors = [np.pad(a, (0, max_len - len(a))) for a in a_vectors]
    e_vectors = [np.pad(a, (0, max_len - len(a))) for a in e_vectors]
    i_vectors = [np.pad(a, (0, max_len - len(a))) for a in i_vectors]
    o_vectors = [np.pad(a, (0, max_len - len(a))) for a in o_vectors]
    u_vectors = [np.pad(a, (0, max_len - len(a))) for a in u_vectors]

    avg_a_vector = np.mean(a_vectors, axis=0)
    avg_e_vector = np.mean(e_vectors, axis=0)
    avg_i_vector = np.mean(i_vectors, axis=0)
    avg_o_vector = np.mean(o_vectors, axis=0)
    avg_u_vector = np.mean(u_vectors, axis=0)
    return avg_a_vector, avg_e_vector, avg_i_vector, avg_o_vector, avg_u_vector

fig, axs = plt.subplots(len(N_FFTs), 1, figsize=(10, 5 * len(N_FFTs)))
print(5*len(N_FFTs))
for i, N_FFT in enumerate(N_FFTs):
    a_vectors, e_vectors, i_vectors, o_vectors, u_vectors = ghi(N_FFT, data_dir)
    avg_a_vector, avg_e_vector, avg_i_vector, avg_o_vector, avg_u_vector = avarage(
        a_vectors, e_vectors, i_vectors, o_vectors, u_vectors
    )

    # Plot the five feature vectors for each N_FFT
    axs[i].plot(avg_a_vector, label='a')
    axs[i].plot(avg_e_vector, label='e')
    axs[i].plot(avg_i_vector, label='i')
    axs[i].plot(avg_o_vector, label='o')
    axs[i].plot(avg_u_vector, label='u')

    axs[i].set_xlabel('Frequency Bins')
    axs[i].set_ylabel('Amplitude')
    axs[i].set_title(f'Average Feature Vectors for N_FFT = {N_FFT}')
    axs[i].legend()
plt.tight_layout()
plt.show()


for N_FFT in N_FFTs:
    correct_counts = {vowel: 0 for vowel in vowels}
    total_counts = {vowel: 0 for vowel in vowels}
    matrix = np.zeros((5,5))
    a_vectors, e_vectors, i_vectors, o_vectors, u_vectors = ghi(N_FFT, data_dir)
    train = avarage(a_vectors, e_vectors, i_vectors, o_vectors, u_vectors)
            
    for folder_path in train_dir:               
        for vowel1 in vowels:
            filepath1 = os.path.join(folder_path, vowel1 + '.wav')                            
                        # Check if the file is a valid audio file
                    
            signal, samplerate = librosa.load(filepath1)
            char_vector = get_fft(signal, N_FFT, samplerate)
            name, label = predict_vowels(char_vector, train, vowel1, vowels)
            total_counts[name] += 1
            for count1, nam in enumerate(vowels):
                if nam == name:
                    for count, vol in enumerate(vowels):
                        if vol == label:
                            matrix[count1][count]+=1
                           
            if name == label:
                correct_counts[label] += 1
    print(matrix)           
    total=0
    for vowel in vowels:
            total += correct_counts[vowel]
            accuracy = "{:.3}".format(correct_counts[vowel] / total_counts[vowel]*100)+"%" 
            print(accuracy,end='  /  ')
    totall = "{:.3}".format(total / 105 * 100)+"%"
    print(f'\nFFT = {N_FFT} : {totall}')
    print(f'{total}/105')

