#********************************************************************
#AFOC Final Project
#Optical Fiber Simulation
#by Aghiorghioaie Fabian
#********************************************************************

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

class Settings:
    bitcount = 16
    bitsamples = 10000
    Vpi = 1                  # V
    freq_carrier = 0
    freq_symbol = 10e9
    threshold = 0
    noise_strength = 0.1
    distance = 10           # Km

    R = 1                    # Amps/Watt

    #//////////////////////////////////////////////////////////////////////

    period_symbol = 1 / freq_symbol
    trise = period_symbol / 12
    samplecount = bitcount * bitsamples
    sampletime = period_symbol / bitsamples
    t = np.arange(0, period_symbol * bitcount, sampletime)
    t_bit = np.arange(0, period_symbol, sampletime)
    phases = [(5 * np.pi) / 4, (7 * np.pi) / 4, (3 * np.pi) / 4, np.pi / 4]

    samples = (np.arange(bitcount) * bitsamples + bitsamples // 2).astype(int)


def rectpuls(t, period_symbol):
    return np.where(np.abs(t) <= period_symbol/2, 1, 0)


def plot_signal(t, signal, xlabel, ylabel, title):
    plt.figure()
    plt.plot(t, signal)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)


class Electrical:
    def __init__(self, I, Q) -> None:
        self.bits_I = [int(bit) for bit in I]
        self.bits_I += [0] * (Settings.bitcount - len(self.bits_I))
        self.bits_Q = [int(bit) for bit in Q]
        self.bits_Q += [0] * (Settings.bitcount - len(self.bits_Q))
        self.V_rect_I = np.zeros(len(Settings.t))
        self.V_rect_Q = np.zeros(len(Settings.t))

        for i in range(Settings.bitcount):
            t_shift = Settings.t - (i + 0.5) * Settings.period_symbol
            self.V_rect_I += (self.bits_I[i] * rectpuls(t_shift, Settings.period_symbol))
            self.V_rect_Q += (self.bits_Q[i] * rectpuls(t_shift, Settings.period_symbol))

        plot_signal(Settings.t, self.V_rect_I, "Time [s]", "Amplitude [V]", "Square Wave (I)")
        self.V_rect_I = np.clip(self.V_rect_I, 0, 1)
        self.V_rect_Q = np.clip(self.V_rect_Q, 0, 1)
        plot_signal(Settings.t, self.V_rect_I, "Time [s]", "Amplitude [V]", "Clipped Square Wave (I)")

        self.V_rect_I = 2 * Settings.Vpi * (Electrical_Filter.filter(self.V_rect_I) - 1)
        self.V_rect_Q = 2 * Settings.Vpi * (Electrical_Filter.filter(self.V_rect_Q) - 1)
        plot_signal(Settings.t, self.V_rect_I, "Time [s]", "Amplitude [V]", "Electronic I")
        plot_signal(Settings.t, self.V_rect_Q, "Time [s]", "Amplitude [V]", "Electronic Q")


class Electrical_Filter:
    @staticmethod
    def filter(signal):
        f = np.fft.fftfreq(len(signal), Settings.sampletime)
        bw = 1.0 / (4/3 * Settings.trise)
        H = np.exp(-(f**2) / (2 * bw**2))
        return np.real(ifft(fft(signal) * H))


class Fiber:
    @staticmethod
    def propagate(signal, distance):
        if distance == 0:
            return signal

        f = np.fft.fftfreq(len(signal), Settings.sampletime)
        beta2 = -21.7e-27
        L_m = distance * 1e3
        H_disp = np.exp(-1j / 2 * beta2 * L_m * (2 * np.pi * f)**2)
        res = ifft(fft(signal) * H_disp)

        att = 10 ** (-0.2 * distance / 20)
        res = res * att

        return res


class Modulator:
    def modulate(Electrical):
        carrier = np.cos(2 * np.pi * Settings.freq_carrier * Settings.t)

        E_Out = 0.5 * (np.cos((np.pi * Electrical.V_rect_I) / (2 * Settings.Vpi)) + \
                        1j * np.cos((np.pi * Electrical.V_rect_Q) / (2 * Settings.Vpi))) * carrier

        factor = 1.45 / np.abs(E_Out).max()
        E_Out = E_Out * factor

        plot_signal(Settings.t, E_Out, "Time [s]", "Amplitude [V]", "ModulatorReal")
        plot_signal(Settings.t, np.imag(E_Out), "Time [s]", "Amplitude [V]", "ModulatorImag")

        plt.figure()
        plt.plot(np.real(E_Out[Settings.samples]), np.imag(E_Out[Settings.samples]), 'x')
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Constelation')
        plt.grid(True)
        return E_Out


class Demodulator:
    def __init__(self, signal) -> None:
        self.signal = signal
        self.lo = np.exp(1j * 2 * np.pi * Settings.freq_carrier * Settings.t)

    def detect(self):
        rx = self.signal * np.conj(self.lo)
        irx = Settings.R * np.real(rx)
        qrx = Settings.R * np.imag(rx)

        plot_signal(Settings.t, irx, "Time [s]", "Amplitude [V]", "irx")
        plot_signal(Settings.t, qrx, "Time [s]", "Amplitude [V]", "qrx")

        bits_I = []
        bits_Q = []
        windowsize = int(len(self.signal) / Settings.bitcount)

        for i in range(0, Settings.bitcount):
            averagei = np.mean(irx[i * windowsize : (i + 1) * windowsize])
            averageq = np.mean(qrx[i * windowsize : (i + 1) * windowsize])

            bit_I = "1" if averagei > Settings.threshold else "0"
            bit_Q = "1" if averageq > Settings.threshold else "0"
            bits_I.append(bit_I)
            bits_Q.append(bit_Q)

        print("Decoded Signal I:", bits_I)
        print("Decoded Signal Q:", bits_Q)
        return bits_I, bits_Q


class Noise:
    def noise(signal):
        rng = np.random.default_rng()
        noise = (rng.normal(0, Settings.noise_strength, size=np.shape(signal)) +
                 1j * rng.normal(0, Settings.noise_strength, size=np.shape(signal)))
        signal = signal + noise

        signal.real = np.clip(signal.real, -1, 1)
        signal.imag = np.clip(signal.imag, -1, 1)

        plot_signal(Settings.t, signal, "Time [s]", "Amplitude [V]", "Noisy Signal (real)")
        plot_signal(Settings.t, np.imag(signal), "Time [s]", "Amplitude [V]", "Noisy Signal (imag)")
        return signal


if __name__ == "__main__":
    I = np.random.randint(2, size=Settings.bitcount)
    Q = np.random.randint(2, size=Settings.bitcount)

    #I = "0101100111001111"         # Manual Assignment

    I_list = list(map(str, I))
    Q_list = list(map(str, Q))

    print("Initial Signal I:", I_list)
    print("Initial Signal Q:", Q_list)

    electrical = Electrical(I, Q)
    optical = Modulator.modulate(electrical)
    optical = Noise.noise(optical)

    optical = Fiber.propagate(optical, Settings.distance)

    plt.figure()
    plt.plot(np.real(optical[Settings.samples]), np.imag(optical[Settings.samples]), 'x')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Constelation After Fiber & Noise')
    plt.grid(True)

    rx = Demodulator(optical)
    rx_I, rx_Q = rx.detect()

    matches_I = sum(i == j for i, j in zip(I_list, rx_I))
    matches_Q = sum(i == j for i, j in zip(Q_list, rx_Q))
    correct_I = (matches_I / len(I_list)) * 100
    correct_Q = (matches_Q / len(Q_list)) * 100

    print(f"Signal I has been transmitted {correct_I}% correctly.")
    print(f"Signal Q has been transmitted {correct_Q}% correctly.")
    plt.show()
