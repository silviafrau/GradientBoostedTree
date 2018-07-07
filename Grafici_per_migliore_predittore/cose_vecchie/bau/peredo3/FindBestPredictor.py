import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import ParameterGrid


class Prediction:

    def __init__(self, dictionary):
        self.name = dictionary['dataset_name']
        self.mean = dictionary['mean']
        self.params = dictionary['out_params']
        self.par = dictionary['in_params']
        self.no_name = dictionary['no_name']
        self.predictions = dictionary['predictions']
        self.labels = dictionary['labels']
        self.features = dictionary['features']

    def getName(self):
        return self.name

    def getMean(self):
        return self.mean

    def getOutParams(self):
        return self.params

    def getInParams(self):
        return (self)

    def getMddDEnd(self):
        return self.no_name

    def returnPredictions(self):
        return self.predictions

    def getAllParams(self):
        out = "Dataset: " + str(self.name) + "\n"
        out += "Accuracy: " + str(self.mean) + "\n"
        out += "Parametri esterni: " + str(self.params) + "\n"
        out += "Parametri interni: " + str(self.par) + "\n"
        out += "Equity/MDD: " + str(self.no_name) + "\n"
        return out


def calculateMdd(element, open):
    train = element.features
    label = element.labels
    train_size = element.params['train_size']
    feature_size = element.params['feature_size']
    accumulatore = 100000
    dollars = 50
    acc = accumulator(train[train_size + feature_size:], accumulatore, dollars,
                      element.predictions, label[train_size + feature_size:], open[train_size + feature_size:])
    mdd = maxDrawdown(acc)
    end = acc[(len(acc) - 1)] - accumulatore

    return mdd, end


def creatListcsv(MergedList, open, dataset_name):
    tmp2 = []
    df3 = pd.DataFrame()
    mddList = []
    endList = []
    fractionList = []
    train_sizeList = []
    featureSizeList = []
    test_sizeList = []
    minSLList = []
    maxDList = []
    n_estList = []

    for element in MergedList:
        tmp2.append(element.mean)
        mdd, end = calculateMdd(element, open)
        mddList.append(mdd)
        endList.append(end)
        fractionList.append(end / mdd)
        train_sizeList.append(element.params['train_size'])
        test_sizeList.append(element.params['test_size'])
        featureSizeList.append(element.params['feature_size'])
        minSLList.append(element.par['min_samples_leaf'])
        maxDList.append(element.par['max_depth'])
        n_estList.append(element.par['n_estimators'])

    df3['ACCURACY'] = tmp2
    df3['MAXDRAWDOWN'] = mddList
    df3['EQUITY'] = endList
    df3['EQUITY/MDD'] = fractionList
    df3['TRAIN_SIZE'] = train_sizeList
    df3['TEST_SIZE'] = test_sizeList
    df3['FEATURE_SIZE'] = featureSizeList
    df3['MIN_SAMPLES_LEAF'] = minSLList
    df3['MAX_DEPTH'] = maxDList
    df3['N_ESTIMATORS'] = n_estList
    df3['LOSS'] = 'DEVIANCE'

    df3.to_csv(dataset_name + 'listaDelleConfigurazioni.csv', sep=',')


def plotMaxGraph(f_train_preparation,accumulatore,dollars,best_pred,l_train_preparation, best_mean, best_out_params, dataset_name,
                 best_in_params, open):
    train_size = best_out_params['train_size']
    test_size = best_out_params['test_size']
    feature_size = best_out_params['feature_size']
    total_length = best_out_params['total_length']
    print('train',len(l_train_preparation[train_size+feature_size:]))
    print('pred',len(best_pred))
    acc = accumulator(f_train_preparation[train_size + feature_size:],accumulatore,dollars,
                      best_pred,l_train_preparation[train_size+feature_size:], open[train_size+feature_size:])
    mdd = maxDrawdown(acc)
    end = acc[(len(acc) - 1)] - accumulatore
    #boh = dict(Acc=best_mean, mdd=mdd, end=end)
    boh = 'Acc = '+str(round(best_mean,4))+'  Mdd = '+ str(round(mdd,4)) + '  End = ' + str(round(end,4))
    #sup = dict(Market = dataset_name, Train =train_size, Test = test_size, Total_length= total_length, feature_size = feature_size)
    sup = 'Market = '+str(dataset_name)+'  Train = '+str(train_size)+'  Test = '+str(test_size) + \
          '  Total Length = '+str(total_length) + '  Feature = ' + str(feature_size)
    #ax = plt.figure(figsize= (10,7)).add_axes([0,0,1,1])
    plt.figure( figsize = (10,7))
    plt.grid(True)
    plt.title(str(boh)+ '  End/Mdd = '+str(round((end/mdd),4)))
    in_par ='loss = '+str(best_in_params['loss']) + '  max_depth = '+ str(best_in_params['max_depth']) \
            + '  min_sample_leaf = '+ str(best_in_params['min_samples_leaf']) + '  n_estimators = '+ str(best_in_params['n_estimators'])
    plt.suptitle(str(sup)+'\n'+in_par)
    #plt.text(0.5, 0, best_in_params, fontsize=12, horizontalalignment = 'center')
    #plt.suptitle(best_in_params, fontsize = 10, fotweight = 'bold')
    #plt.title(boh, fontsize=9)
    plt.ylabel('Amount(USD)')
    plt.xlabel('Days')
    plt.plot(acc)
    plt.savefig(str(dataset_name)+str(feature_size)+'DaysGraph.png', format='png')
    #plt.show()

def saveParams(prediction):
    with open(str(prediction.name)+str(prediction.params['feature_size'])+'DaysGraph.txt','w+') as f:
        f.write(prediction.getAllParams())



# Split the dataset into 7 days of train and 1 predicted
def preparation(f_train, l_train, step_size, feature_size, offset):
    train_list = []
    test_list = []
    #Provo a togliere - feature size
    for i in range(offset, offset + step_size - feature_size):
        list = []
        list = f_train[i:i + feature_size]
        train_list.append(list)
        test_list.append(l_train[i + feature_size])

    return train_list, test_list


def maxDrawdown(tmp):
    # Calculate Max Drawdown
    maxValue = float(tmp[0])
    drawdown = []

    for i in range(0, len(tmp) - 1, 1):
        if (float(tmp[i + 1])) > maxValue:
            maxValue = float(tmp[i + 1])
        else:
            drawdown.append(abs(maxValue - tmp[i + 1]))

    mdd = max(drawdown)
    return mdd

def generateRow(dataset, start, end, hour):
    subset = dataset.loc[start:end]

    # Rimuovi il commento se vuoi vedere cosa sta facendo
    # print ("\nSubset from ", start, " to ", end, ": ", subset)

    openValue = subset['Open'].mean()
    closeValue = subset['Close'].mean()
    return [
        subset.loc[start, 'Date'],  # Date
        hour,  # Time
        openValue,  # Open
        subset['High'].mean(),  # High
        subset['Low'].mean(),  # Low
        closeValue,  # Close
        subset['Up'].mean(),  # Up
        subset['Down'].mean(),  # Down
        subset['Volume'].mean(),  # Volume
        closeValue - openValue  # Diff
    ]


# Add a target column for labels
def add_target_column(df, file_name):
    # Add a target column at the table
    z = []
    for index, row in df.iterrows():
        if row[-1] >= 0:
            z.append(1)
        else:
            z.append(-1)

    df['sign'] = z
    df.to_csv(file_name, sep=',', index=False)

# Add a diff column if that does not exist
def add_diff_column(df):
    tmp = df['Close'].tolist()
    tmp2 = df['Open'].tolist()
    diff = []
    for i in range(0, len(tmp)):
        diff.append(tmp[i] - tmp2[i])

    return diff


# Add a diff column if that does not exist
def add_diff_column1(df):
    tmp = df['CLOSE'].tolist()
    tmp2 = df['OPEN'].tolist()
    diff = []
    for i in range(0, len(tmp)):
        diff.append(tmp[i] - tmp2[i])

    return diff
# If the sign column does not exists, calculate it
def signGenerator(df):
    sign = []
    for element in df:
        if element >= 0:
            sign.append(1)
        else:
            sign.append(-1)

    return sign

def hourlytrasformator(df):
    vals = add_diff_column(df)

    hourArray = []

    lastHour = ""
    mean = 0
    count = 0
    for index, value in enumerate(df['Time']):
        # Divide la stringa dell'ora in due parti dove ci sono i due punti. Poi prende la prima.
        hour = value.split(':')[0]

        # Se il ciclo e' appena iniziato o l'ora e' ancora la stessa dell'iterazione precedente
        if (lastHour == "" or lastHour == hour):
            # Aggiunge il valore (nel tuo caso vai a prendere close-open) a 'mean'
            mean += vals[index]
            # Incrementa il conto (serve per fare la media dopo)
            count += 1

        # Se l'ora e' diversa rispetto all'iterazione precedente, bisogna aggiungere la media
        elif (lastHour != hour):
            print
            "Saving mean for hour ", lastHour, ": ", str(mean / count)
            # Aggiungi la media dividendo 'mean' per il numero di elementi ( 'count' )
            hourArray.append(str(mean / count))

            # Mean prende il valore della nuova ora
            mean = vals[index]

            # Dato che abbiamo un valore il conto va impostato a 1
            count = 1

        # Qua ti salvi il valore dell'ora a questa iterazione (serve per il prossimo ciclo)
        lastHour = hour

    # Alla fine del ciclo se il conto e' diverso da 0 bisogna aggiungere la media
    # (serve per non skippare l'ultimo valore)
    if (count != 0):
        print
        "Saving mean for hour ", lastHour, ": ", str(mean / count)
        hourArray.append(str(mean / count))

    print("Hours array: ", hourArray)
    #d = {'date':date, 'time':time,'diff':diff_Column}
    #df1 = pd.DataFrame(data=d)
    #print('df1',df1)
    #return df1

def hourlyTimeSeCreate(df):
    #subsetCount = 200
    data = df[:]

    #print("\nSubset of ", len(data), " elements: \n", data.to_string())

    newData = pd.DataFrame(columns=list(data.columns) + ['Diff'])

    startIndex = 0
    lastHour = data.loc[0, 'Time'].split(':')[0]
    count = 0
    for index, row in data.iterrows():
        count +=1
        hour = row['Time'].split(':')[0]
        if (hour != lastHour):
            # Generates row
            row = generateRow(data, startIndex, index - 1, lastHour)
            # Adds row to new DataFrame
            newData.loc[-1] = row
            newData.index += 1
            # Restarts from current index
            startIndex = index
        lastHour = hour
        print(count)

    # Adds last row
    lastRow = generateRow(data, startIndex, len(df), lastHour)
    newData.loc[-1] = lastRow
    newData.index += 1

    # Resets indices
    newData = newData.reset_index(drop=True)

    #print("\nNew Data: \n", newData.to_string())
    newData.to_csv("S&P500_hourly2.csv")

def unique(seq, idfun=None):
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

def Date(df):
    date = df['Date']
    date = unique(date)
    return date

# Accumulator in order to create a series of values up or down
def accumulator (f_train_preparation,accumulator, dollars, predictions, l_train_preparation, open):
    acc = []
    for i in range(0, len(predictions)):
        if predictions[i] == l_train_preparation[i]:
            accumulator += abs((f_train_preparation[i] * dollars))
        else:
            accumulator -= abs((f_train_preparation[i] * dollars))
        acc.append(accumulator)

    '''''''''
    open = np.array(open)
    open.flatten()
    print('Close - Open :',f_train_preparation[:10])
    print('Predizione :', predictions[:10])
    print('Equity : ', acc[:10])
    print('Valore del mercato : ', open[:10])
    df2 = pd.DataFrame()
    df2['Diff'] = f_train_preparation[:200]
    df2['Predizioni'] = predictions[:200]
    df2['Equity'] = acc[:200]
    df2['ValoreMercato'] = open[:200]
    df2.to_csv('Test.csv', sep = ',')
    '''
    return acc

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


def predict(df):
    contatore = 0
    # print(df['Time'])
    print('start')
    # hourlyTimeSeCreate(df)
    print('ok')

    # If df does not hace diff and sign columns create them
    if df.columns[-1] == 'Volume':
        diff = add_diff_column(df)
        sign = signGenerator(diff)
    elif df.columns[-1] == 'CLASS':
        diff = add_diff_column1(df)
        sign = signGenerator(diff)

    open = df.iloc[:, [1]].values.tolist()

    # plot_signal_decomp(diff,'sym5','DWT')
    f_train_preparation = diff  # Save the diff column
    l_train_preparation = sign  # Save the sign column

    # In parameters
    loss = ['deviance']
    # n_estimators = [1000, 100, 500, 1500]
    n_estimators = [50, 100, 500,1000]
    #n_estimators = [100]
    max_depth = [3,6]  # tolgo il 3 che non ho voglia di aspettare
    # min_samples_leaf = [15, 10]
    min_samples_leaf = [10, 15]
    #min_samples_leaf = [15]
    # Parameter grid for internal parameters
    param_grid = dict(loss=loss, n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    ingrid = ParameterGrid(param_grid)

    # Out parameters
    # test_sizes = [150, 200, 100, 50]
    test_sizes = [100,150, 200]
    #test_sizes = [150]
    train_sizes = [200, 400, 500]
    # train_sizes = [200,500]
    #train_sizes = [200]
    total_length = [len(diff) - 1]
    feature_sizes = [2,4,7]
    #feature_sizes = [7]
    # train_sizes = [500]
    # test_sizes = [150]

    # Out parameters grid
    parameter = dict(train_size=train_sizes, test_size=test_sizes, total_length=total_length,
                     feature_size=feature_sizes)
    grid = ParameterGrid(parameter)

    list_of_accuracies = []
    mean = 0
    accumulatore = 100000
    dollars = 50
    no_name = 0
    mdd = 0
    count = (len(test_sizes) * len(train_sizes))
    best_no_name = 0
    best_in_params = []
    best_out_params = []

    Sevendays = []
    Fourdays = []
    Twodays = []

    count1 = (len(loss) * len(n_estimators) * len(min_samples_leaf) * len(max_depth))
    # For each internal parameter
    for par in ingrid:
        count -= 1
        #print('Inpar: ', par)
        # Create a new model with new parameters
        model = GradientBoostingClassifier(
            loss=par['loss'],
            n_estimators=par['n_estimators'],
            max_depth=par['max_depth'],
            min_samples_leaf=par['min_samples_leaf'],
            random_state=0)

        # For each external parameter
        for params in grid:
            list_of_predictions = []
            count -= 1  # To the end print the graph
            #print('par', params)

            test_size = params['test_size']
            offset = test_size
            feature_size = params['feature_size']
            train_size = params['train_size']
            in_sample_size = params['total_length']

            if train_size > test_size:

                # Prepare the feature and the label for the fit ( 0-train size)
                featureTr_list, labelTr_list = preparation(f_train_preparation, l_train_preparation, train_size,
                                                           feature_size,
                                                           0)
                featureTr_list = normalize(featureTr_list)
                model.fit(featureTr_list, labelTr_list)

                # Prepare the feature and the label for the predict (train-size - test-size)
                featureTe_list, labelTe_list = preparation(f_train_preparation, l_train_preparation, test_size,
                                                           feature_size, train_size)  # 100 of predictions
                predictions = model.predict(featureTe_list)

                # Add predictions to a list
                list_of_predictions = np.hstack((list_of_predictions, predictions))

                # Accuracy saved
                accuracy = accuracy_score(labelTe_list, predictions)
                list_of_accuracies.append(accuracy)

                #print('TRAIN SIZE: ', train_size)
                #print('TEST_SIZE: ', test_size)

                # From the offset(test size) - feature_size to the total length - train size and test-size, jumping on test size
                # - feature size
                for i in range(offset - feature_size, in_sample_size - train_size - test_size,
                               test_size - feature_size):
                    # Prepare data to the fitting
                    featureTr_list, labelTr_list = preparation(f_train_preparation, l_train_preparation, train_size,
                                                               feature_size,
                                                               i)
                    featureTr_list = normalize(featureTr_list)
                    model.fit(featureTr_list, labelTr_list)

                    # Prepare data to the prediction
                    featureTe_list, labelTe_list = preparation(f_train_preparation, l_train_preparation, test_size,
                                                               feature_size, i + train_size)  # 100 of predictions
                    featureTe_list = normalize(featureTe_list)
                    predictions = model.predict(featureTe_list)  # Predict the first 100th

                    # Calculate th accuracy
                    accuracy = accuracy_score(labelTe_list, predictions)  # first accuracy
                    list_of_accuracies.append(accuracy)

                    # Concantenate the new predictions
                    list_of_predictions = np.hstack((list_of_predictions, predictions))

                old_best_pred = list_of_predictions
                old_mean = mean
                mean = np.mean(list_of_accuracies)

                old_mdd = mdd

                # Create the growth of trade in order to plot it, the mdd and the end of the series
                acc = accumulator(f_train_preparation[train_size + feature_size:], accumulatore, dollars,
                                  list_of_predictions, l_train_preparation[train_size + feature_size:],
                                  open[train_size + feature_size:])
                mdd = maxDrawdown(acc)
                end = acc[(len(acc) - 1)] - accumulatore

                # old_no_name = no_name
                no_name = end / mdd
                #print('END/MDD: ', no_name)
                # print('OLD END/MDD: ', old_no_name)

                if no_name >= best_no_name:
                    best_no_name = no_name
                    #print('BEST: ', best_no_name)
                    best_pred = list_of_predictions
                    best_out_params = params
                    best_in_params = par
                    best_mean = mean
                #else:
                    #print('BEST: ', best_no_name)

                #print('best mean', best_mean)
                # print('best_mean', best_mean)
                #print('best_out_param', best_out_params)

                # For best params plot the graph
                list_of_accuracies = []

                if count <= 0:
                    #plotMaxGraph(f_train_preparation, accumulatore, dollars,
                                 #best_pred, l_train_preparation, best_mean, best_out_params, dataset_name,
                                 #best_in_params, open)
                    # plt.figure()
                    # autocorrelation_plot(acc)
                    # plt.show()




                    # Salvo in un dizionario tutti i valori/parametri di ogni cosa calcolata a ogni ciclo
                    longdict = dict(dataset_name=dataset_name, mean=mean, out_params=params, in_params=par,
                                    no_name=no_name,
                                    predictions=list_of_predictions, labels=l_train_preparation,
                                    features=f_train_preparation)

                    # Istanzio un oggetto di tipo Prediction per salvare ogni oggetto creato
                    pred = Prediction(longdict)
                    #print(pred.name)
                    if feature_size == 7:
                        # Lo appendo in una lista contenene tutti gli oggetti, per poi utilizzarla fuori e cercare i cosi migliori
                        Sevendays.append(pred)
                        #print('length list', len(Sevendays))

                    elif feature_size == 4:
                        Fourdays.append(pred)
                    else:
                        Twodays.append(pred)

                    # Reinizializzo
                    count = (len(test_sizes) * len(train_sizes))
                    best_out_params = []
                    mean = 0
                    best_pred = 0
                    best_no_name = 0
                contatore += 1
                print(str((contatore/16)* 100)+"%")
                # Prendo i valori migliori in assoluto

    return Sevendays, Fourdays, Twodays


# Read the csv
dataset_name = 'EUROBUND_FULL_NATIVE.csv'
df = pd.read_csv(dataset_name)

open1 = df.iloc[:,[1]]
# Mode for wavelets use
#mode = pywt.Modes.smooth

Sevenlist, Fourlist,Twolist = predict(df)

print(len(Sevenlist))
print(len(Fourlist))
print(len(Twolist))

best_mean = -999999999
best_element = Sevenlist[0]
best2 = []
MergedList = Sevenlist + Fourlist + Twolist

creatListcsv(MergedList, open1, dataset_name)

'''''''''
for element in Sevenlist:
    print('no name: ', element.no_name)
    if element.no_name >= best_mean:
        print('no name: ',element.no_name)
        best_element = element
        best_mean = element.no_name

accumulatore = 100000
dollars = 50
print(best_element.par)
plotMaxGraph(best_element.features, accumulatore, dollars, best_element.predictions, best_element.labels,
             best_element.mean, best_element.params,
             dataset_name, best_element.par, open1)

saveParams(best_element)

best_mean = -9999999999
best2 = Fourlist[0]
for element in Fourlist:
    print('no name: ', element.no_name)
    if element.no_name >= best_mean:
        print('no name: ', element.no_name)
        best2 = element
        best_mean = element.no_name

print()
plotMaxGraph(best2.features, accumulatore, dollars, best2.predictions, best2.labels, best2.mean, best2.params,
             dataset_name, best2.par, open1)
saveParams(best2)

#best_element = []
best_mean = -9999999999
best3 = Twolist[0]
for element in Twolist:
    print('no name: ', element.no_name)
    if element.no_name >= best_mean:
        print('no name: ', element.no_name)
        best3 = element
        best_mean = element.no_name

print(best3.par)
plotMaxGraph(best3.features, accumulatore, dollars, best3.predictions, best3.labels,
             best3.mean, best3.params,
             dataset_name, best3.par, open1)
saveParams(best3)



mergedList = Sevenlist + Fourlist + Twolist
best_first = mergedList[0]
counter = 0
b = 0
print(len(mergedList))
for element in mergedList:
    if element.no_name == best_mean:
        best_first = element
        best_mean = element.no_name
        b = counter
    counter +=1

plotMaxGraph(best_first.features, accumulatore, dollars, best_first.predictions, best_first.labels,
             best_first.mean, best_first.params,
             dataset_name, best_first.par, open1)
saveParams(best_first)
del mergedList[b]
counter = 0
print((len(mergedList)))
best__mean = -999999999
best_second = mergedList[0]

for element in mergedList:
    if element.no_name == best_mean:
        best_second = element
        best_mean = element.no_name
        b = counter
    counter += 1

plotMaxGraph(best_second.features, accumulatore, dollars, best_second.predictions, best_second.labels,
             best_second.mean, best_second.params,
             dataset_name, best_second.par, open1)
saveParams(best_second)

del mergedList[b]
print((len(mergedList)))

best__mean = -999999999
best_second = mergedList[0]
for element in mergedList:
    if element.no_name == best_mean:
        best_second = element
        best_mean = element.no_name

plotMaxGraph(best_second.features, accumulatore, dollars, best_second.predictions, best_second.labels,
             best_second.mean, best_second.params,
             dataset_name, best_second.par, open1)
saveParams(best_second)
'''''
'''''''''''
#print(obj_list)
# Controllo il valore migliore per ogni parametro
z = []
predictions = obj_list[0]
#print(obj_list[0])
array_pred = predictions.predictions
print(len(array_pred))

for element in array_pred:
    z.append(element)

print(z)
train_size = 500
test_size = 100
feat_size = 2
date = df.ix[train_size+feat_size:(-33),[0]]
date = df['DATE'].values.tolist()
date = date[train_size+feat_size:-33]
print('date',len(date))
print('z',len(z))
print(type(date))
print(date)
#index = range(0,len(z))

df1 = pd.DataFrame()
df1['Predictions'] = z
#df['index'] = index
#list = []
#for element in date:
#    list.append[element]

df1['Date'] = date
df1.to_csv('Prova.csv', sep=',')

predict(df)

'''''
