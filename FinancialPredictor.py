import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import ParameterGrid
import pywt


def plotMaxGraph(f_train_preparation,feature_size,accumulatore,dollars,best_pred,l_train_preparation, best_mean, best_out_params):
    train_size = best_out_params['train_size']
    test_size = best_out_params['test_size']
    total_length = best_out_params['total_length']
    print('train',len(l_train_preparation[train_size+feature_size:]))
    print('pred',len(best_pred))
    acc = accumulator(f_train_preparation[train_size + feature_size:],accumulatore,dollars,
                      best_pred,l_train_preparation[train_size+feature_size:])
    mdd = maxDrawdown(acc)
    end = acc[(len(acc) - 1)] - accumulatore
    boh = dict(Acc=best_mean, mdd=mdd, end=end, end_d_mdd=(end / mdd))
    sup = dict(Market = 'SP500', Train =train_size, Test = test_size, Total_length= total_length)
    plt.figure()
    plt.grid(True)
    plt.suptitle(sup, fontsize=14, fontweight='bold')
    plt.title(boh, fontsize=8)
    plt.ylabel('Amount(USD)')
    plt.xlabel('Days')
    plt.plot(acc)
    plt.show()

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
def accumulator (f_train_preparation,accumulator, dollars, predictions, l_train_preparation):
    acc = []
    for i in range(0, len(predictions)):
        if predictions[i] == l_train_preparation[i]:
            accumulator += (f_train_preparation[i] * dollars)
        else:
            accumulator -= (f_train_preparation[i] * dollars)
        acc.append(accumulator)
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


hourly_dataset_name = 'sp.csv'

# Read the csv
dataset_name = 'sp.csv'
df = pd.read_csv(hourly_dataset_name)

# Mode for wavelets use
mode = pywt.Modes.smooth

#print(df['Time'])
print('start')
#hourlyTimeSeCreate(df)
print('ok')

# If df does not hace diff and sign columns create them
if df.columns[-1] == 'Volume':
    diff = add_diff_column(df)
    sign = signGenerator(diff)
elif dataset_name =='sp1.csv':
    diff = df.iloc[1000:,['diff']].tolist()
    sign = df.iloc[1000:,['sign']].tolist()
elif dataset_name == 'sp':
    diff = df.iloc[:,-2]
    sign = df.iloc[:,-1]
else:
    diff = df.iloc[:,-1]
    sign = signGenerator(df)


#plot_signal_decomp(diff,'sym5','DWT')
f_train_preparation = diff # Save the diff column
l_train_preparation = sign # Save the sign column
print('f',f_train_preparation)
print('l',l_train_preparation)
# For days i have tested 7 days and 1 predicted. for minutes 170 is good or 171 a bit less
step_size = 500
offset = 0
feature_size =  7 #170 per orarie #7 giornaliere #23 per orarie
list_of_accuracies = []

# Prepare the data 7 days and 1 lists -> 193 elements in labels (from 8 to 200)
featureTr_list, labelTr_list = preparation(f_train_preparation,l_train_preparation,step_size, feature_size,offset)
featureTr_list = normalize(featureTr_list)

# Model of the classifier
model = GradientBoostingClassifier(
                n_estimators=1000,
                learning_rate=0.001,
                max_depth=6,
                min_samples_leaf=15,
                random_state=0)

# Do the search of parameters in order to optimize the algorithm
loss = ['deviance','exponential']
n_estimators = [1000,100,500,1500]
max_depth = [6,3]
min_samples_leaf = [15,10]
param_grid = dict(loss=loss, n_estimators =n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
grid = GridSearchCV(estimator=model, param_grid=param_grid)

# Test the GridSearch in order to get the best ones
grid_result = grid.fit(featureTr_list, labelTr_list)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print(grid_result.best_score_)
print(grid_result.best_params_)
loss = grid_result.best_params_['loss']
n_estimators = grid_result.best_params_['n_estimators']
max_depth = grid_result.best_params_['max_depth']
min_samples_leaf = grid_result.best_params_['min_samples_leaf']

# Save best parameters to next fit
best_params = dict(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=min_samples_leaf)

# Print results of gridsearchCV
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# Tune the hype
#test_sizes = [10,20,30,50,100,130,160,200]
#train_sizes = [100,150,180,200,250,300,500]
#test_sizes = [150,200,100,50]
#train_sizes = [400,500,200]
total_length = [len(diff)-1]
train_sizes = [500]
test_sizes = [150]

parameter = dict(train_size=train_sizes, test_size = test_sizes, total_length= total_length)
grid = ParameterGrid(parameter)

# Create new model type
model = GradientBoostingClassifier(
                loss=loss,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=0)

mean = 0
accumulatore = 100000
dollars = 50

count = (len(test_sizes)*len(train_sizes)) -1
for params in grid:
    list_of_predictions=[]
    count -=1
    print('par',params)
    test_size = params['test_size']
    offset = test_size
    mdd = 0
    train_size = params['train_size']
    in_sample_size= params['total_length']
    print('Parameters:', params)
    '''''''''
    offset = 100
    test_size = 100
    train_size = 400
    in_sample_size = 4000
    '''''

    if train_size > test_size:
        featureTr_list, labelTr_list = preparation(f_train_preparation, l_train_preparation, train_size, feature_size,
                                                   0)
        featureTr_list = normalize(featureTr_list)
        model.fit(featureTr_list,labelTr_list)
        featureTe_list, labelTe_list = preparation(f_train_preparation, l_train_preparation, test_size,
                                                       feature_size, train_size)  # 100 of predictions
        predictions = model.predict(featureTe_list)
        #print('pred1',len(predictions))
        list_of_predictions = np.hstack((list_of_predictions,predictions))
        #print('totalpred',len(list_of_predictions))

        accuracy = accuracy_score(labelTe_list,predictions)
        list_of_accuracies.append(accuracy)

        #in_lenght = int(in_sample_size*0.8)
        #train_size = int(in_sample_size*0.8)
        #test_size = int(in_sample_size*0.2)

        #count = 0
        #date = df.iloc[:1000,0]
        #date = pd.to_datetime(date).year
        #print('date',date)
        # From the offset to the end of the series fit on the 200th element and predict on 100th successive.
        for i in range(offset-feature_size,in_sample_size - train_size - test_size,test_size-feature_size):
            # fit su 200 (90 % esempio)
            #count +=1
            print(i)
            featureTr_list, labelTr_list = preparation(f_train_preparation, l_train_preparation, train_size, feature_size,
                                                       i)
            featureTr_list = normalize(featureTr_list)
            #print('f',featureTr_list)
            #print('l',labelTr_list)
            #print('lenF',len(featureTr_list))
            #print('lenL',len(labelTr_list))
            model.fit(featureTr_list, labelTr_list)
            # test su 50
            #print(i)
            # mi sposto di 50
            #offset = step_size  # Starts from 200
            featureTe_list, labelTe_list = preparation(f_train_preparation, l_train_preparation, test_size,
                                                       feature_size, i+train_size)  # 100 of predictions

            featureTe_list = normalize(featureTe_list)
            #print(len(featureTe_list))
            predictions = model.predict(featureTe_list)  # Predict the first 100th
            #print('2pred',len(predictions))
            #print('prediction',predictions)
            #print('labelTe_list',labelTe_list)
            accuracy = accuracy_score(labelTe_list, predictions)  # first accuracy
            list_of_accuracies.append(accuracy)
            #print(accuracy)
            list_of_predictions = np.hstack((list_of_predictions,predictions))
            #print(list_of_predictions)
        #print(list_of_accuracies)
        old_best_pred = list_of_predictions
        old_mean = mean
        mean = np.mean(list_of_accuracies)

        old_mdd = mdd
        '''''''''
        
        if mean >= old_mean:
            best_mean = mean
            best_pred = list_of_predictions
            best_out_params = params
        #else:
            #best_mean = old_mean
        '''''
        acc = accumulator(f_train_preparation[train_size + feature_size:], accumulatore, dollars,
                          list_of_predictions, l_train_preparation[train_size + feature_size:])
        mdd = maxDrawdown(acc)


        if mdd >= old_mdd:
            best_mdd = mdd
            best_pred = list_of_predictions
            best_out_params = params


        print('mean',mean)
        #print('best_mean', best_mean)
        print('best_out_param', best_out_params)

        # print('train_now',train_size)
        #print('test_now',test_size)
        #print('length',total_length)
        #print('pred',len(list_of_predictions))
        #print('old',len(best_pred))
        # For best params plot the graph
        list_of_accuracies = []
        #print(len(list_of_predictions))
        #print(len(best_pred))
        #print('train',len(l_train_preparation[train_size+feature_size:]))
        if count <= 0 :
            plotMaxGraph(f_train_preparation,feature_size,accumulatore,dollars,
                         best_pred,l_train_preparation,best_mdd,best_out_params)
        '''''''''''
        acc = accumulator(f_train_preparation[train_size+feature_size:],accumulatore,dollars,list_of_predictions,l_train_preparation[train_size+feature_size:])
        print(len(acc))
        print(acc)
        mdd = maxDrawdown(acc)
        end = acc[(len(acc)-1)] - accumulatore
        boh = dict(Acc= best_mean,mdd=mdd,end=end, end_d_mdd=(end/mdd))
        plt.figure()
        plt.grid(True)
        plt.suptitle('Market:SP500 Train: 85%',fontsize = 14, fontweight='bold')
        plt.title(boh, fontsize=8)
        plt.ylabel('Amount(USD)')
        plt.xlabel('Days')
        plt.plot(acc)
        plt.show()
    '''''

train_size = best_out_params['train_size']
test_size = best_out_params['test_size']
total_length = best_out_params['total_length']

'''''''''
count1 = range(0,count+1)
realyear = []
year = range(2000,2015)
number_of_year = int(len(list_of_accuracies)/4) #4 deriva da 12/3 cioè numero mesi in un anno/numero mesi tra train e test

realyear = [2000,2000,2000,2000,2001,2001,2001,2001,2002,2002,2002,2002,2003,2003,2003,2003,2004,2004,2004,2004,2005,2005,2005,2005,2006,2006,2006,2006,2007]


plt.figure()
plt.xlabel('epoca')
plt.ylabel('accuratezza')
#plt.xticks(realyear)
plt.plot(count1,list_of_accuracies)
plt.show()


mean = np.mean(list_of_accuracies)
print('mean',mean)


plt.figure()
autocorrelation_plot(diff)
plt.show()
'''''''''
'''''''''''
Autocorrelation plots are often used for checking randomness in time series. This is done by computing autocorrelations for data
 values at varying time lags. If time series is random, such autocorrelations should be near zero for any and all time-lag 
 separations. If time series is non-random then one or more of the autocorrelations will be significantly non-zero. 
 The horizontal lines displayed in the plot correspond to 95% and 99% confidence bands. The dashed line is 99% confidence band.
'''''''''''