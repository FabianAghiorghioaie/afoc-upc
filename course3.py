#********************************************************************
#AFOC Course 3
#Optical Fiber Simulation
#by Aghiorghioaie Fabian
#********************************************************************

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft, ifft

bitRate = 10e9
bitPeriod = 1/bitRate
bitNumber = 2**7
timeWindow = bitNumber * bitPeriod
numberSamplesPerBit = 2**7
samplePeriod = bitPeriod/numberSamplesPerBit
sampleRate = 1/samplePeriod
lambda_ = 1550e-9
c = 3e8

N = int(bitNumber * numberSamplesPerBit)

t = np.linspace(0, timeWindow - samplePeriod, N)
f = np.fft.fftfreq(N, samplePeriod)
w = 2 * np.pi * f
w_c = 2 * np.pi * (sampleRate / 2)

sigma_t = bitPeriod/4
t_0 = timeWindow/2
a_0 = np.exp(-1/2*((t-t_0)/sigma_t)**2)

plt.figure("Initial Pulse")
plt.plot((t-t_0)*1e12, a_0)
plt.xlim([-2*bitPeriod*1e12, 2*bitPeriod*1e12])
plt.ylabel('normalized amplitude')
plt.xlabel('relative time (ps)')

A_0 = fftshift(fft(fftshift(a_0)))

sigma_f = 1/sigma_t
plt.figure("Initial Spectrum")
plt.plot(fftshift(f)*1e-9, abs(fftshift(A_0)*samplePeriod))
plt.xlim([-sigma_f*1e-9, sigma_f*1e-9])
plt.ylabel('normalized amplitude')
plt.xlabel('relative frequency (GHz)')

alpha_dB = 0.2 # dB/km
alpha = alpha_dB / (10 * 1e3) * np.log(10)

z = np.arange(0, 101, 10) * 1e3 # m
a_a = np.exp(-alpha * z[:, np.newaxis]) * a_0

plt.figure("Attenuation")
plt.plot((t - t_0) * 1e12, a_a.T ** 2)
plt.xlim([-2 * bitPeriod * 1e12, 2 * bitPeriod * 1e12])
plt.ylabel('power (normalized)')
plt.xlabel('relative time (ps)')
plt.legend([str(i / 1e3) + ' km' for i in z])

D = 17e-6 # 17 ps/(nm*km)
Beta_2 = -lambda_ ** 2 / (2 * np.pi * c) * D
phi_Beta_2 = Beta_2 / 2 * z[:, np.newaxis] * (w - w_c) ** 2

A_d = np.ones((len(z), 1)) * A_0 * np.exp(-1j * phi_Beta_2)
a_d = fftshift(ifft(fftshift(A_d, axes=1), axis=1), axes=1)

plt.figure("Dispersion Phase")
plt.plot(fftshift(f)*1e-9, phi_Beta_2.T)
plt.xlim([-sigma_f * 1e-9, sigma_f * 1e-9])
plt.ylabel('phase (rad)')
plt.xlabel('relative frequency (GHz)')
plt.legend([str(i / 1e3) + ' km' for i in z])

plt.figure("Dispersion Power")
plt.plot((t - t_0) * 1e12, np.abs(a_d.T) ** 2)
plt.xlim([-2 * bitPeriod * 1e12, 2 * bitPeriod * 1e12])
plt.ylabel('power (normalized)')
plt.xlabel('relative time (ps)')
plt.legend([str(i / 1e3) + ' km' for i in z])

# Kerr
gamma = 1.3 / 1e3 # 1/(W*m)
power = 100e-3    # 100 mW
p_0 = np.sqrt(power) * a_0
P_0 = np.sqrt(power) * A_0
phi_NL = gamma * z[:, np.newaxis] * np.abs(p_0) ** 2
p_NL = np.exp(-1j * phi_NL) * p_0

plt.figure("Nonlinearity")
plt.subplot(2, 1, 1)
plt.plot((t - t_0) * 1e12, np.abs(p_NL.T) ** 2)
plt.xlim([-2 * bitPeriod * 1e12, 2 * bitPeriod * 1e12])
plt.ylabel('power (W)')
plt.xlabel('relative time (ps)')
plt.legend([str(i / 1e3) + ' km' for i in z])
plt.subplot(2, 1, 2)
plt.plot((t - t_0) * 1e12, np.unwrap(-np.angle(p_NL.T)))
plt.xlim([-2 * bitPeriod * 1e12, 2 * bitPeriod * 1e12])
plt.ylabel('phase (rad)')
plt.xlabel('relative time (ps)')
plt.legend([str(i / 1e3) + ' km' for i in z])


# SSFM
delta_phi = 0.005
h_0_exact = np.floor(delta_phi / gamma / np.max(np.abs(p_0)) ** 2)
h_0 = 40 # Manual step size (meters)
z_h = np.arange(0, 101e3, h_0)

sample_z = (z / h_0).astype(int)

p_L = np.zeros((len(z_h), len(t)), dtype=complex)
p_L[0, :] = p_0
P_L = np.zeros((len(z_h), len(f)), dtype=complex)
P_L[0, :] = P_0

for i in range(1, len(z_h)):
    phi_Beta_2_i = Beta_2 / 2 * h_0 * (w - w_c) ** 2
    H_linear = np.exp(-1j * phi_Beta_2_i) * np.exp(-alpha * h_0)
    P_L[i, :] = P_L[i - 1, :] * H_linear
    p_L[i, :] = fftshift(ifft(fftshift(P_L[i, :])))
    phi_NL_i = gamma * h_0 * np.abs(p_L[i, :]) ** 2
    p_L[i, :] = np.exp(-1j * phi_NL_i) * p_L[i, :]
    P_L[i, :] = fftshift(fft(fftshift(p_L[i, :])))

plt.figure("SSFM Propagation")
plt.plot((t - t_0) * 1e12, np.abs(p_L[sample_z, :].T) ** 2)
plt.xlim([-1 * bitPeriod * 1e12, 1 * bitPeriod * 1e12])
plt.ylabel('optical power (W)')
plt.xlabel('relative time (ps)')
plt.legend([str(i / 1e3) + ' km' for i in z])

h_L = delta_phi / gamma / np.max(np.abs(p_0)) ** 2 * np.exp(2 * alpha * z)

plt.figure("Adaptive Step Size")
plt.plot(z / 1e3, h_L / 1e3)
plt.ylabel('SSMF Step Size (km)')
plt.xlabel('Transmission Distance (km)')

plt.show()
