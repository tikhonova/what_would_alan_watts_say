''' Remove music and noise leaving voice only via https://librosa.org/librosa_gallery/auto_examples/plot_vocal_separation.html '''
# script
from __future__ import print_function

import librosa.display
import numpy as np
import soundfile as sf
import os

file_path = "E:/AlanWattsMaterialSorted/split_audio5/"
dest_path = "E:/AlanWatts/dataset/split_audio2/"


for file in os.listdir(file_path):
    if file not in os.listdir(dest_path):
        # Load the audio file
        y, sr = librosa.load(os.path.join(file_path, file))

        # Compute the spectrogram magnitude and phase
        S_full, phase = librosa.magphase(librosa.stft(y))

        # We'll compare frames using cosine similarity, and aggregate similar frames
        # by taking their (per-frequency) median value.
        #
        # To avoid being biased by local continuity, we constrain similar frames to be
        # separated by at least 2 seconds.
        #
        # This suppresses sparse/non-repetetitive deviations from the average spectrum,
        # and works well to discard vocal elements.

        S_filter = librosa.decompose.nn_filter(S_full,
                                               aggregate=np.median,
                                               metric='cosine',
                                               width=int(librosa.time_to_frames(2, sr=sr)))

        # The output of the filter shouldn't be greater than the input
        # if we assume signals are additive.  Taking the pointwise minimium
        # with the input spectrum forces this.
        S_filter = np.minimum(S_full, S_filter)

        # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
        # Note: the margins need not be equal for foreground and background separation
        margin_i, margin_v = 2, 10
        power = 2

        mask_i = librosa.util.softmask(S_filter,
                                       margin_i * (S_full - S_filter),
                                       power=power)

        mask_v = librosa.util.softmask(S_full - S_filter,
                                       margin_v * S_filter,
                                       power=power)

        # Once we have the masks, simply multiply them with the input spectrum
        # to separate the components

        S_foreground = mask_v * S_full
        S_background = mask_i * S_full

        # Convert the foreground spectrogram back to the time domain
        new_y = librosa.istft(S_foreground * phase)

        sf.write(os.path.join(dest_path, file), new_y, sr)
