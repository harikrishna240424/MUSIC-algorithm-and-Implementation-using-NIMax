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

def acquire(fs=100000,N= 300000 ):

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

def steervec(theta, f0, d, c=1500):
    k = 2 * np.pi * f0 / c
    n = np.arange(4)
    return np.exp(-1j * k * d * n * np.sin(theta))

def music2d(X, f0, hydropos):
    rxx = covariance_matrix(X)

    eigvals, eigvecs = eig(rxx)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    M = 1
    En = eigvecs[:, M:]

    azimuths = np.linspace(-90, 90, 181)
    elevations = np.linspace(0, 90, 91)

    P = np.zeros((len(elevations), len(azimuths)))

    for i, el in enumerate(elevations):
        for j, az in enumerate(azimuths):
            a = steervec(
                np.deg2rad(el),
                np.deg2rad(az),
                f0,
                hydropos
            )
            P[i, j] = 1 / np.linalg.norm(En.conj().T @ a)**2

    idx_max = np.unravel_index(np.argmax(P), P.shape)
    max_el = elevations[idx_max[0]]
    max_az = azimuths[idx_max[1]]

    return azimuths, elevations, P, max_az, max_el

if __name__ == "__main__":

    X, fs = acquire()

    
    fl = 20000
    fh = 40000
    Xf = bandpass(X, fs, fl, fh)
    f0 = (fl + fh) / 2

    # 2D hydrophone positions in meters considering Square planar array
    hydropos = np.array([
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.0, 0.05, 0.0],
        [0.05, 0.05, 0.0]
    ])


    az, el, P, az_max, el_max = music2d(Xf, f0, hydropos)

    print(f"Estimated Azimuth  : {az_max:.2f} deg")
    print(f"Estimated Elevation: {el_max:.2f} deg")

    ''' #Plot 2D MUSIC spectrum
    plt.figure(figsize=(8, 6))
    plt.imshow(
        P,
        extent=[az[0], az[-1], el[0], el[-1]],
        origin='lower',
        aspect='auto'
    )
    plt.colorbar(label="MUSIC Spectrum")
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Elevation (deg)")
    plt.title("2D MUSIC DOA Spectrum")
    plt.show()'''
