import math

from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

def energy(frames):
    e=0
    energyVec = []
    for i in range(frames.shape[0]):
        for j in range(frames.shape[1]):
            e += frames[i][j]**2
        energyVec.append(e)
        e=0
    return energyVec
def zeroCrossing(frames):
    ZerocrossingVec = np.zeros(shape=(frames.shape[0]))
    sizeVertical = frames.shape[0]
    sizeHorizontal = frames.shape[1]

    for i in range(sizeVertical):
        for j in range(sizeHorizontal - 1):
            if frames[i, j] * frames[i, j + 1] < 0:
                ZerocrossingVec[i] += 1
    return ZerocrossingVec
    # x = np.arange(0, frames.shape[1])
    # plt.plot(x, Histogram_final)
    # plt.show()

def framing_and_windowing(path,frame_size,overlap_size,window_type):

    samplerate, data = wavfile.read(path)  # load audio file
    n = data.size
    T = 1 / samplerate  # sample duration (ms)
    timeVec = np.arange(n) * T  # signal duration in time vector

     # plotting the original signal

    plt.figure(figsize=(12,12))
    plt.subplot(4,1,1)
    plt.title("original")
    plt.plot(timeVec,data)

    # framing with overlapping
    FrameSizeSampels = round(frame_size * samplerate)
    OverlapSizeSamples = round(overlap_size * samplerate)
    shift = round((frame_size-overlap_size)*samplerate)
    nframes = math.ceil((len(data)-FrameSizeSampels) / shift)
    frames =np.zeros((nframes,FrameSizeSampels))
    for i in range(0,nframes):
        start = i * shift#OverlapSizeSamples
        end = start + FrameSizeSampels
        if FrameSizeSampels > len(data[start:end]):
            arr = data[start:].copy()
            arr.resize(FrameSizeSampels)  #padding
            frames[i] = arr
        else:
            frames[i] = data[start:end]
    vector = frames.flatten()
    if window_type == "rectangular":
        energyVec = energy(frames)
        zerocrossingVec = zeroCrossing(frames)
        # print(frames)
        print(len(zerocrossingVec))
        # print(len(data))
        # print(len(vector))
        timeVec = np.arange(len(vector)) / samplerate
        plt.subplot(4, 1, 2)
        plt.title("rectangular")
        plt.plot(timeVec, vector)
        timeVec = np.arange(len(energyVec)) / samplerate
        plt.subplot(4, 1, 3)
        plt.title("Energy")
        plt.plot(timeVec, energyVec)
        timeVec = np.arange(len(zerocrossingVec)) / samplerate
        plt.subplot(4, 1, 4)
        plt.title("Zero Crossing")
        plt.plot(timeVec, zerocrossingVec)
        plt.show()

    elif window_type == "hamming":
        for i in range(0, nframes):
            window_hamming = np.hamming(FrameSizeSampels)
            frames[i] = frames[i] * window_hamming
        # print(np.array(frames))
        energyVec = energy(frames)
        zerocrossingVec = zeroCrossing(frames)
        # print(energyVec)
        print(len(energyVec))
        vector = frames.flatten()
        timeVec = np.arange(len(vector)) / samplerate
        plt.subplot(4, 1, 2)
        plt.title("hamming")
        plt.plot(timeVec, vector)
        timeVec = np.arange(len(energyVec)) / samplerate
        plt.subplot(4, 1, 3)
        plt.title("Energy")
        plt.plot(timeVec, energyVec)
        timeVec = np.arange(len(zerocrossingVec)) / samplerate
        plt.subplot(4, 1, 4)
        plt.title("Zero Crossing")
        plt.plot(timeVec, zerocrossingVec)
        plt.show()

    elif window_type == "hanning":
        for i in range(0, nframes):
            window_hanning = np.hanning(FrameSizeSampels)
            frames[i] = frames[i] * window_hanning

        # print(np.array(frames))
        energyVec = energy(frames)
        zerocrossingVec = zeroCrossing(frames)
        # print(energyVec)
        print(len(energyVec))
        vector = frames.flatten()
        timeVec = np.arange(len(vector)) / samplerate
        plt.subplot(4, 1, 2)
        plt.title("hanning")
        plt.plot(timeVec, vector)
        timeVec = np.arange(len(energyVec)) / samplerate
        plt.subplot(4, 1, 3)
        plt.title("Energy")
        plt.plot(timeVec, energyVec)
        timeVec = np.arange(len(zerocrossingVec)) / samplerate
        plt.subplot(4, 1, 4)
        plt.title("Zero Crossing")
        plt.plot(timeVec, zerocrossingVec)
        plt.show()


if __name__ == '__main__':
    # path = input("Enter File Path   ")
    # Frame_size =float( input("Enter Frame size in seconds "))
    # Overlap_size = float(input("Enter Overlap size in seconds "))
    # Window_type = input("Window type to be used ")
    path = "test.wav"
    Frame_size = 0.02
    Overlap_size = 0.01
    Window_type = "hanning"
    print(framing_and_windowing(path,Frame_size,Overlap_size,Window_type))
