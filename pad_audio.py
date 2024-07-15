import argparse

import numpy as np
import soundfile as sf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str, default="padded.wav")
    args = parser.parse_args()

    data, samplerate = sf.read(args.input)
    print(f"samplerate: {samplerate}")
    print(f"data.shape: {data.shape}")
    padding = [(int(samplerate * 0.5), int(samplerate * 0.5))]
    for _ in range(len(data.shape) - 1):
        padding += [(0, 0)]
    print(f"padding: {padding}")
    padded_data = np.pad(data, padding, mode="constant")
    # padded_data = data[:, 1]
    print(f"padded_data.shape: {padded_data.shape}")
    sf.write(args.output, padded_data, samplerate)