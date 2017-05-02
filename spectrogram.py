import librosa
import matplotlib.pyplot as plt
import numpy as np
from librosa.display import specshow
import os
import json
import pandas as pd

def scrape_audio_filenames(root):
    """Return list of absolute file paths within a directory
    INPUT: root (string): root directory to crawl through
    OUTPUT: filenames (list of strings): files within directory
    """
    raw_filenames = []
    # list of filenames in directory, including invisible and metadata files
    for root, dirs, files in os.walk(root, topdown=False):
        for name in files:
            filenames.append(os.path.join(root, name))

    audio_filenames = []
    # sort through and populate with list of audio files only

    for filename in filenames:
        if "/." not in filename:
            if filename[-5:] != ".json" and filename[-4:] != ".csv":
                audio_filenames.append(filename)
    return audio_filenames

def audio_to_spectrogram(root, audio_filenames):
    """
    """

    for audio_filename in audio_filenames[:100]:
        csv_filename = ".".join(audio_filename.split(".")[:-1]) + ".csv"
        metadata = pd.read_csv(csv_filename)

        # breaks up spectrogram into individual sound components if necessary
        # (e.g. several dog barks into individual barks)
        for row in metadata.itertuples():
            offset = row[1]
            duration = row[2] - row[1]
            label = row[4]

            spectrogram_filename = (root + "/spectrograms/" +
                                        label + "/" + str(i) + ".png")
            try:
                y, sr = librosa.load(audio_filename,
                                        offset=offset,
                                        duration=duration)
                D = librosa.core.stft(y)

                librosa.display.specshow(librosa.amplitude_to_db
                                                        (D, ref=np.max))
                plt.savefig(spectrogram_filename)
                i += 1
            except:
                continue


if __name__ == '__main__':
    root = '~' # this is where you saved the UrbanSound data
    audio_filenames = scrape_audio_filenames(root)
    audio_to_spectrogram(root, audio_filenames)
