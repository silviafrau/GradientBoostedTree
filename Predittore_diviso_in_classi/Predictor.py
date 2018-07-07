import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import ParameterGrid
from models.Prediction import Prediction
from commons import *



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




def predict(df):
    contatore = 0
    # print(df['Time'])
    print('start')
    # hourlyTimeSeCreate(df)
    print('ok')

    features, labels = get_features_and_labels(df)
    open = get_market_value(df)     # Valore del mercato giornaliero

    # plot_signal_decomp(diff,'sym5','DWT')
    f_train_preparation = features  # Save the diff column
    l_train_preparation = labels  # Save the sign column

    # In parameters
    loss = ['deviance']
    # n_estimators = [1000, 100, 500, 1500]
    n_estimators = [50, 100, 500,1000]
    #n_estimators = [100]
    max_depth = [3,6]  # tolgo il 3 che non ho voglia di aspettare
    #max_depth = [3]
    #max_depth = [3]
    # min_samples_leaf = [15, 10]
    min_samples_leaf = [10, 15]
    #min_samples_leaf = [15]

    in_grid = get_in_parameters(loss,n_estimators,max_depth,min_samples_leaf)

    # Out parameters
    # test_sizes = [150, 200, 100, 50]
    #test_sizes = [100,150, 200]
    #test_sizes = [150]
    test_sizes = [500]
    #train_sizes = [200, 400, 500]
    #train_sizes = [450,200]
    train_sizes = [2000]
    total_length = [len(diff) - 1]
    feature_sizes = [2,4,7]
    #feature_sizes = [7]
    # train_sizes = [500]
    # test_sizes = [150]

    out_grid = get_out_parameters(train_sizes,test_sizes,total_length,feature_sizes)

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
    counterStrike = 432
    count1 = (len(loss) * len(n_estimators) * len(min_samples_leaf) * len(max_depth))
    # For each internal parameter
    for par in in_grid:
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
        for params in out_grid:
            list_of_predictions = []
            count -= 1  # To the end print the graph
            #print('par', params)

            test_size = params['test_size']
            offset = test_size
            feature_size = params['feature_size']
            train_size = params['train_size']
            in_sample_size = params['total_length']
            counterStrike -= 1
            print(counterStrike)
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
                        print('real',train_size)
                    elif feature_size == 4:
                        Fourdays.append(pred)
                        print('real',train_size)
                    else:
                        Twodays.append(pred)
                        print('real', train_size)

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
dataset_name = 'SP500_FULL_NATIVE.csv'
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
for element in MergedList:
    print('valori in Merge',element.params['train_size'])

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
