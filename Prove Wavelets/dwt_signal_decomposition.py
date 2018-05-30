import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import pywt.data


ecg = pywt.data.ecg()
df = pd.read_csv('sp.csv')
data1 = np.concatenate((np.arange(1, 400),
                        np.arange(398, 600),
                        np.arange(601, 1024)))
x = np.linspace(0.082, 2.128, num=1024)[::-1]
data2 = np.sin(40 * np.log(x)) * np.sign((np.log(x)))

mode = pywt.Modes.smooth

tmp = df['Close'].tolist()
tmp2 = df['Open'].tolist()
diff = []

for i in range(0, len(tmp)):
    diff.append(tmp[i] - tmp2[i])
data3 = diff

miao = pywt.threshold(data3, 2, 'soft')
#print(miao)
hard = pywt.threshold(data3,2,'hard')
def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []
    for i in range(5):
        (a, d) = pywt.dwt(a, w, mode)   #perform a single level transform
        ca.append(a)        # Coefficiente d'approssimazione
        cd.append(d)        # Coefficiente di dettaglio

    caT = []
    cdT = []

    for row in ca:
        pywt.threshold(row,2,'hard')
        caT.append(row)

    ca = caT

    for row in cd:
        pywt.threshold(row,2,'hard')
        cdT.append(row)

    cd = cdT

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):      # Create a list of coefficients
        coeff_list = [coeff, None] + [None] * i
        print(coeff_list)
        rec_a.append(pywt.waverec(coeff_list, w))      # Perform multilevel reconstruction of signal from a list of coeffs

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))

    print(rec_a)
    #hard = pywt.threshold(rec_a, 2, 'hard')
    print(hard)
    #.threshold(rec_d, 1, 'soft')
    # Plot the series
    fig = plt.figure()
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    # Plot the approximation coeffs
    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))

    #Plot the detail coeffs
    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))

    return


plot_signal_decomp(data1, 'coif5', "DWT: Signal irregularity")
plot_signal_decomp(hard, 'sym5',
                   "DWT: Frequency and phase change - Symmlets5")
plot_signal_decomp(data3, 'sym5', "DWT: Ecg sample - Symmlets5")
plot_signal_decomp(miao,'sym5','DWT')

plt.show()