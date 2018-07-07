# Plot the graph of the signal decomposition with Wavelets
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
    #print(hard)
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

