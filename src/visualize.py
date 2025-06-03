import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_audio(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set_title('Waveform')

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    ax[1].set_title('Spectrogram')

    librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax[2])
    ax[2].set_title('MFCC')

    plt.tight_layout()
    plt.savefig(f"{os.path.splitext(file_path)[0]}_viz.png")
