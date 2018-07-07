import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import csv

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

def calculateCovariance(buy_sell_list, dollars, diff,sign):
    tmp = 100000
    accumulatorWithCo = []
    print(len(diff))
    print(len(buy_sell_list))
    accumulatorWithCo.append(tmp)
    for i in range(0, len(buy_sell_list)):
        '''
        if buy_sell_list[i] != 0:
            if buy_sell_list[i] == sign[i]:
                tmp += abs(dollars * diff[i])
            else:
                tmp -= abs(dollars * diff[i])
        accumulatorWithCo.append(tmp)
        '''

        if buy_sell_list[i] > 0:
            if buy_sell_list[i] == sign[i]:
                tmp += abs(dollars * diff[i])
            else:
                tmp -= abs(dollars * diff[i])
        elif buy_sell_list[i] < 0:

            if buy_sell_list[i] == sign[i]:
                tmp += abs(dollars * diff[i])
            else:
                tmp -= abs(dollars * diff[i])

        '''
        if buy_sell_list[i]>0 :
            tmp += abs(dollars * diff[i])
        elif buy_sell_list[i]<0:
            tmp -= abs(dollars * diff[i])
        '''
        #tmp += buy_sell_list[i] * abs(dollars * diff[i])
        accumulatorWithCo.append(tmp)

    return accumulatorWithCo

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

def results_csv(coverage, average_acc, final_accuracy,threshold, n_predictors, dataset_name, mdd, decision_number, end):
    if pd.read_csv('SP500result.csv') is not None:
        df3 = pd.DataFrame()
        df3['NUMERO_PREDITTORI'] = n_predictors
        print(n_predictors)
        print(df3['NUMERO_PREDITTORI'])
        df3['COVERAGE'] = coverage
        df3['MEDIA_ACCURATEZZE'] = average_acc
        df3['ACCURATEZZA_FINALE'] = final_accuracy
        df3['MEDIE_INFERIORI'] = threshold
        df3['MAXDRAWDOWN'] = mdd
        df3['END'] = end
        df3['END/MDD'] = end/mdd
        df3['THRESHOLD'] = str(decision_number) + '%'
        df3.to_csv(dataset_name+'result.csv', sep=',')
    else:
        append_to_csv(n_predictors,coverage,average_acc,final_accuracy,threshold,mdd,end,end/mdd,decision_number)

def append_to_csv(n_predictors,coverage_percentage,avg_accuracies,final_accuracy,medie_inf,mdd,end,end_mdd,percentuale):
    fields = [n_predictors,coverage_percentage,avg_accuracies,final_accuracy,medie_inf,mdd,end,end_mdd,percentuale]
    with open(r'SP500result.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def test_csv(list_of_obj,buy_sell_list,date, dataset_name):
    df4 = pd.DataFrame()
    df4['DATE'] = date
    i = 0
    for element in list_of_obj:
        df4['PREDIZIONE'+str(i)] = element
        i+=1

    df4['DECISIONE'] = buy_sell_list
    df4.to_csv(dataset_name+'test.csv', sep=',')


def print_csv(data, decision, up_or_down, open, max_value, min_value, close, volume, diff):
    df4 = pd.DataFrame()
    df4['DATA'] = data
    df4['DECISIONE'] = decision
    df4['ANDAMENTO MERCATO'] = up_or_down
    df4['OPEN'] = open
    df4['MAX'] = max_value
    df4['MIN'] = min_value
    df4['CLOSE'] = close
    df4['VOLUME'] = volume
    df4['CLOSE-OPEN'] = diff

    df4.to_csv('values.csv', sep=',')

def final_accuracy(n_hold,n_error,length_pred):
    accuracy = 1.0 - (n_error/(length_pred - n_hold))
    print(n_error)
    return accuracy

def predict(df, dict_param):
    contatore = 0
    print('start')

    if df.columns[-1] == 'CLASS':
        diff = add_diff_column1(df)
        sign = signGenerator(diff)

    open = df.iloc[:, [1]].values.tolist()

    # plot_signal_decomp(diff,'sym5','DWT')
    f_train_preparation = diff  # Save the diff column
    l_train_preparation = sign  # Save the sign column

    # In parameters
    loss = 'deviance'
    n_estimators = dict_param['n_estimators']
    max_depth = dict_param['max_depth']
    min_samples_leaf = dict_param['min_samples_leaf']

    test_sizes = dict_param['test_size']
    train_sizes =  dict_param['train_size']
    total_length = len(diff) - 1
    feature_sizes = dict_param['feature_size']

    list_of_accuracies = []
    mean = 0
    accumulatore = 100000
    dollars = 50
    no_name = 0
    mdd = 0
    best_no_name = 0
    best_in_params = []
    best_out_params = []

    Sevendays = []

    model = GradientBoostingClassifier(
        loss=loss,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=0)


    list_of_predictions = []


    offset = test_size
    in_sample_size = total_length

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

    mean = np.mean(list_of_accuracies)

    # print('best_mean', best_mean)
    #print('best_out_param', best_out_params)

    # For best params plot the graph
    list_of_accuracies = []
    # Salvo in un dizionario tutti i valori/parametri di ogni cosa calcolata a ogni ciclo
    params = dict(train_size=train_size, test_size=test_size, feature_size=feature_size, in_sample_size=in_sample_size)

    par = dict(loss=loss, n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf= min_samples_leaf)

    longdict = dict(dataset_name=dataset_name, mean=mean, out_params=params, in_params=par,
                    no_name=no_name,
                    predictions=list_of_predictions, labels=l_train_preparation,
                    features=f_train_preparation)
    # Istanzio un oggetto di tipo Prediction per salvare ogni oggetto creato
    pred = Prediction(longdict)

    return pred


def plotCovariance(nuovo_acc, name_fig, dataset_name, accuratezza_media,coverage_percentage, n_under_thre, number_elements,final_accuracy):
    mdd = maxDrawdown(nuovo_acc)
    end = nuovo_acc[(len(nuovo_acc)-1)] - 100000
    plt.figure(figsize = (10,7))
    plt.grid(True)
    plt.title('Market = '+ dataset_name +' mdd = '+ str(round(mdd,4))+' end = '
              +str(round(end,4))+ ' end/mdd = ' + str(round(end/mdd,4)) + ' final_accuracy ' + str(final_accuracy) + '\n' + ' avg_accuracies = '
              + str(round(accuratezza_media,2)) + ' coverage = ' + str(round(coverage_percentage,3)) + '% '
              + ' accuracies under 50% = ' + str(n_under_thre) + ' n_predictors = ' + str(number_elements))
    plt.ylabel('Amount(USD)')
    plt.xlabel('Days')
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.draw()
    plt.plot(nuovo_acc)
    plt.savefig(name_fig)


# Leggo e salvo i parametri da csv dei 10 migliori
dataset_name = 'sp500configurazionicorte.csv'
dataset_name2 = 'SP500'
df = pd.read_csv(dataset_name)

number_of_elements = [20]
list_of_pred = []
list_of_parameters = []
list_of_accuracies = []

for idx,rows in df.iterrows():
    if idx < number_of_elements[0] :
        train_size = df['TRAIN_SIZE'][idx]
        test_size = df['TEST_SIZE'][idx]
        feature_size = df['FEATURE_SIZE'][idx]
        min_samples_leaf = df['MIN_SAMPLES_LEAF'][idx]
        max_depth = df['MAX_DEPTH'][idx]
        n_estimators = df['N_ESTIMATORS'][idx]
        par_dict = dict(train_size= train_size, test_size=test_size,feature_size=feature_size,
                        min_samples_leaf=min_samples_leaf, max_depth=max_depth, n_estimators=n_estimators)
        list_of_parameters.append(par_dict)
        list_of_accuracies.append(df['ACCURACY'][idx])
        print(idx)

# Predico di nuovo ricalcolandomi tutto
df1 = pd.read_csv('SP500_Corto.csv')
df1 = df1.dropna()
print(len(df1))
date = df1['DATE']
open = df1['OPEN']
close = df1['CLOSE']
max_value = df1['HIGH']
min_value = df1['LOW']
volume = df1['VOLUME']
diff = add_diff_column1(df1)
sign = signGenerator(diff)

for par_dict in list_of_parameters:
    print(par_dict)
    list_of_pred.append(predict(df1, par_dict))

print(len(list_of_pred))
# Allineo le cose delle 10 cose

lengths = []

for obj in list_of_pred:
    lengths.append(len(obj.predictions))

# Recupero parametri per lo slicing
train_size_max = 0
missing_size_max = 0
for obj in list_of_pred:
    _train_size = _prediction.params["train_size"]
    _missing_size = _prediction.params["in_sample_size"] - len(obj.predictions) - _train_size
    if _train_size > train_size_max:
        train_size_max = _train_size
    if _missing_size > missing_size_max:
        missing_size_max = _missing_size

# salvo solo le predizioni per poterle modificare
list_of_obj = [obj.predictions for obj in list_of_pred]

# Cambio l'inizio di ogni predizione in base alla più piccola
start = 0
for obj in list_of_obj:
    obj = obj[train_size_max:-missing_size_max]
    print(len(obj))
date = date[train_size_max:-missing_size_max]
open = open[train_size_max:-missing_size_max]
sign = sign[train_size_max:-missing_size_max]
diff = diff[train_size_max:-missing_size_max]
close = close[train_size_max:-missing_size_max]
max_value = max_value[train_size_max:-missing_size_max]
min_value = min_value[train_size_max:-missing_size_max]
volume = volume[train_size_max:-missing_size_max]

print('lunghezza',len(list_of_obj))
# eseguo
flag = ''
column_list = []
buy_sell_list = []
decision_number = 0.8*number_of_elements[0]
print(decision_number)
count = 0
accuracy_threshold = 0.50

# lo fa solo per 100 percento di coverage
for i in range(0,min_length):
    for element in list_of_obj:
        column_list.append(element[i])
        print('element',element)

    positive = 0
    negative = 0

    for element in column_list:
        if element == 1:
            positive += 1
        else:
            negative +=1
    print('n',negative)
    print('p',positive)

    print('column list', column_list)

    tmp = list(set(column_list))
    print(tmp)
    print(len(tmp))

    if positive > negative and positive >= decision_number:
        flag = 'BUY'
        count += 1
        print('p')
    elif positive < negative and negative >= decision_number:
        flag = 'SELL'
        count += 1
        print('n')
    else:
        flag = 'HOLD'
    buy_sell_list.append(flag)
    column_list = []

coverage_percentage = (count)/len(buy_sell_list) * 100

average_accuracies = np.mean(list_of_accuracies)
n_accuracy_under = 0
for element in list_of_accuracies:
    if element < accuracy_threshold:
        n_accuracy_under +=1

print(buy_sell_list)
print('coverage', coverage_percentage)
print('average accuracies', average_accuracies)
print('n accuracy under the threshold',n_accuracy_under)

# Calcolo parametri
n_hold = 0
dollars = 50
accumulatore1 = []
for element in buy_sell_list:
    if element == 'BUY':
        accumulatore1.append(1)
    elif element == 'SELL':
        accumulatore1.append(-1)
    else:
        n_hold += 1
        accumulatore1.append(0)

n_right = 0
for i in range(0,len(accumulatore1)):
    if accumulatore1[i] == 1 or accumulatore1[i] == -1:
        if accumulatore1[i] == sign[i]:
            n_right +=1

n_error = (len(accumulatore1) - n_hold) - n_right

print(accumulatore1)
accumulatore = calculateCovariance(accumulatore1,dollars,diff, sign)
print(accumulatore)
mdd = maxDrawdown(accumulatore)
end = accumulatore[(len(accumulatore)-1)] - 100000

final_accuracy = final_accuracy(n_hold,n_error,len(buy_sell_list))
# salvo in un file csv i risultati di coverage, accuratezza media, quanti son minori del 50  percento in acc
results_csv(coverage_percentage,average_accuracies,final_accuracy, n_accuracy_under,number_of_elements,
            dataset_name2, mdd,decision_number, end)
print('cove',coverage_percentage)
# salvo in un file anche le liste di predizioni con relativa modalità: buy hold o sell
#test_csv(list_of_obj,buy_sell_list,date,dataset_name2)

# stampo il grafico con l hold
plotCovariance(accumulatore,'SP5500',dataset_name2, average_accuracies, coverage_percentage, n_accuracy_under, number_of_elements, final_accuracy)

print_csv(date,buy_sell_list,sign,open,max_value,min_value,close,volume,diff)



