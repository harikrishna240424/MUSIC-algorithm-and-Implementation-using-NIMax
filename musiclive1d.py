import nidaqmx
import numpy as np
import matplotlib.pyplot as plt

from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE
from scipy.signal import butter, filtfilt
from numpy.linalg import eig


def bandpass(data, fs, f_low, f_high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [f_low/nyq, f_high/nyq], btype='band')
    return filtfilt(b, a, data, axis=1)


def acquire_data(fs=100000, N=300000):
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai0")
        task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai1")
        task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai2")
        task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai3")

        task.timing.cfg_samp_clk_timing(
            fs,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=N
        )

        data = np.array(task.read(READ_ALL_AVAILABLE))

    return data, fs


def covariance_matrix(X):
    return (X @ X.conj().T) / X.shape[1]


def steering_vector_1d(theta, f0, d, M, c=1500):
    """
    theta : radians (azimuth)
    f0    : center frequency
    d     : element spacing
    M     : number of sensors
    """
    k = 2 * np.pi * f0 / c
    n = np.arange(M)
    return np.exp(-1j * k * d * n * np.sin(theta))


def music_doa_1d(X, f0, d):
    """
    X : (M, N) data matrix
    """
    M = X.shape[0]

    Rxx = covariance_matrix(X)

    eigvals, eigvecs = eig(Rxx)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    num_sources = 1
    En = eigvecs[:, num_sources:]

    angles = np.linspace(-90, 90, 181)
    P = np.zeros(len(angles))

    for i, ang in enumerate(angles):
        a = steering_vector_1d(
            np.deg2rad(ang),
            f0,
            d,
            M
        )
        P[i] = 1 / np.linalg.norm(En.conj().T @ a)**2

    doa = angles[np.argmax(P)]
    return angles, P, doa


if __name__ == "__main__":

    X, fs = acquire_data()

    fl = 20000
    fh = 40000
    Xf = bandpass(X, fs, fl, fh)

    f0 = (fl + fh) / 2

    # Linear array spacing (meters)
    d = 0.05

    angles, P, doa = music_doa_1d(Xf, f0, d)

    print(f"Estimated DOA (Azimuth): {doa:.2f} deg")

    '''
    plt.figure(figsize=(8, 4))
    plt.plot(angles, 10 * np.log10(P / P.max()))
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("MUSIC Spectrum (dB)")
    plt.title("1D MUSIC DOA")
    plt.grid()
    plt.show()
    '''
