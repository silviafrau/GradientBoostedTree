import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import csv
import sys
import time
import os

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
        self.dates = dictionary['dates']

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

# Add a diff column if that does not exist
def get_diff_column(df):
    close_values = df['CLOSE'].tolist()
    open_values = df['OPEN'].tolist()
    diff_values = []
    for index in range(0, len(close_values)):
        diff_values.append(close_values[index] - open_values[index])

    return diff_values

# If the sign column does not exists, calculate it
def get_sign_column_from_diff(df):
    df_sign = []
    for item in df:
        if item >= 0:
            df_sign.append(1)
        else:
            df_sign.append(-1)

    return df_sign

def calculate_covariance(buy_sell_list, initial_capital, dollars, diff, sign):
    covariance = []
    for i in range(0, len(buy_sell_list)):
        if buy_sell_list[i] == sign[i]:
            initial_capital += dollars * abs(diff[i])
        elif buy_sell_list[i] == 0:
            pass
        else:
            initial_capital -= dollars * abs(diff[i])
        covariance.append(initial_capital)
    return covariance

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

def results_to_csv(coverage, average_acc, final_accuracy, threshold, n_predictors, dataset_name, mdd, decision_number, end):
    df3 = pd.DataFrame()
    df3['NUMERO_PREDITTORI'] =[n_predictors]
    df3['COVERAGE'] = coverage
    df3['MEDIA_ACCURATEZZE'] = average_acc
    df3['ACCURATEZZA_FINALE'] = final_accuracy
    df3['MEDIE_INFERIORI'] = threshold
    df3['MAXDRAWDOWN'] = mdd
    df3['END'] = end
    df3['END/MDD'] = float(end/mdd)
    df3['THRESHOLD'] = "{}%".format(decision_number * 100.0)

    if os.path.isfile(dataset_name + 'result.csv'.format(dataset_name)):
        with open(dataset_name + 'result.csv', 'a') as f:
            df3.to_csv(f, index=False, sep=',', header=False)
    else:
        df3.to_csv(dataset_name + 'result.csv', index=False, sep=',')

def append_to_csv(n_predictors,coverage_percentage,avg_accuracies,final_accuracy,medie_inf,mdd,end,end_mdd,percentuale):
    fields = [n_predictors,coverage_percentage,avg_accuracies,final_accuracy,medie_inf,mdd,end,end_mdd,percentuale]
    with open(r'SP500result.csv', 'a') as f:
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


def save_values_as_csv(destination_path, data, decision, up_or_down, open, max_value, min_value, close, volume, diff):
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

    df4.to_csv('{}/values.csv'.format(destination_path), sep=',')

def get_final_accuracy(n_hold,n_error,length_pred):
    accuracy = 1.0 - (n_error/(length_pred - n_hold))
    print(n_error)
    return accuracy

# Split the dataset into 7 days of train and 1 predicted
def preparation(f_train, l_train, dates, step_size, feature_size, offset):
    train_list = []
    test_list = []
    dates_list = []
    #Provo a togliere - feature size
    for i in range(offset, offset + step_size - feature_size):
        dates_list.append(dates[i + feature_size])
        train_list.append(f_train[i:i + feature_size])
        test_list.append(l_train[i + feature_size])

    return train_list, test_list, dates_list

def make_prediction(model, diffs, signs, dates, train_size, test_size, feature_size, offset):
    # Prepare the feature and the label for the fit ( 0-train size)
    featureTr_list, labelTr_list, _  = preparation(diffs,
                                                signs,
                                                dates,
                                               train_size,
                                               feature_size,
                                               offset)
    featureTr_list = normalize(featureTr_list)

    # Executes the fit
    model.fit(featureTr_list, labelTr_list)

    # Prepare the feature and the label for the predict (train-size - test-size)
    featureTe_list, labelTe_list, dates_list = preparation(diffs,
                                               signs,
                                               dates,
                                               test_size,
                                               feature_size,
                                               offset + train_size)
    featureTe_list = normalize(featureTe_list)

    # Executes the predict
    prediction = model.predict(featureTe_list)
    return prediction, accuracy_score(labelTe_list, prediction), dates_list

def predict(dataframe, parameters, dataset_name):

    # Extracts features from the dataframe
    dates = dataframe['DATE'].values
    diffs = get_diff_column(dataframe)
    signs = get_sign_column_from_diff(diffs)

    # Internal parameters
    loss = 'deviance'
    n_estimators = parameters['n_estimators']
    max_depth = parameters['max_depth']
    min_samples_leaf = parameters['min_samples_leaf']
    test_size = parameters['test_size']
    train_size =  parameters['train_size']
    feature_size = parameters['feature_size']

    # Iteration parameters
    total_length = len(diffs) - 1
    offset = test_size
    in_sample_size = total_length

    predictions_accuracies = []
    no_name = 0
    # Initializes Gradient Boosting model
    model = GradientBoostingClassifier(
        loss=loss,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=0)

    # Add predictions to a list
    prediction_values = []
    prediction_dates = []

    # Iterates through the dataframe at the given offset and sample size
    for i in range(0,
                   in_sample_size - train_size - test_size,
                   test_size - feature_size):

        prediction, accuracy, dates_list = make_prediction(model, diffs, signs, dates, train_size, test_size, feature_size, i)

        # Stores the prediction accuracy
        predictions_accuracies.append(accuracy)

        # Concantenate the new predictions
        prediction_values = np.hstack((prediction_values, prediction))
        prediction_dates = np.hstack((prediction_dates, dates_list))
        # Store associated prediction dates

    # Computes the mean value of all predictions accuracies
    accuracies_mean = np.mean(predictions_accuracies)

    # Stores the out parameters in a dictionary
    out_parameters = dict(train_size=train_size,
                          test_size=test_size,
                          feature_size=feature_size,
                          in_sample_size=in_sample_size)

    # Stores the in parameters in a dictionary
    in_parameters = dict(loss=loss,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf)

    # Populates dictionary to initialize the final Prediction object
    longdict = dict(dataset_name=dataset_name,
                    mean=accuracies_mean,
                    out_params=out_parameters,
                    in_params=in_parameters,
                    no_name=no_name,
                    predictions=prediction_values,
                    labels=diffs,
                    features=signs,
                    dates=prediction_dates)

    return Prediction(longdict)

def plotCovariance(destination_path, nuovo_acc, name_fig, dataset_name, accuratezza_media,coverage_percentage, n_under_thre, number_elements,final_accuracy):
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
    plt.savefig("{}/{}".format(destination_path,name_fig))
#---------------------------------------------------------------------------

def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def extract_parameters(config_dataframe, predictors_number):
    parameters_list = []
    accuracies_list = []
    for idx, rows in config_dataframe.iterrows():
        if idx < predictors_number:
            # Retrieve features values from the Dataframe 'df'
            train_size = config_dataframe['TRAIN_SIZE'][idx]
            test_size = config_dataframe['TEST_SIZE'][idx]
            feature_size = config_dataframe['FEATURE_SIZE'][idx]
            min_samples_leaf = config_dataframe['MIN_SAMPLES_LEAF'][idx]
            max_depth = config_dataframe['MAX_DEPTH'][idx]
            n_estimators = config_dataframe['N_ESTIMATORS'][idx]

            # Adds features values to the parameters list
            parameters_list.append(dict(
                train_size=train_size,
                test_size=test_size,
                feature_size=feature_size,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                n_estimators=n_estimators))

            # Adds the accuracy value to the accuracies list
            accuracies_list.append(config_dataframe['ACCURACY'][idx])
        else:
            break
    return parameters_list, accuracies_list

# Loads dataframe and preprocesses it
def load_dataframe(dataframe_file_path):
    df = pd.read_csv(dataframe_file_path)

    df = df.dropna()

    cut_index = None
    for idx, row in df.iterrows():
        try:
            year = get_date_object(row["DATE"]).tm_year
        except:
            print("ERROR IN {} , row: {}. Date is {}".format(dataframe_file_path, row, row["DATE"]))
            exit(1)
        if year > 2015:
            cut_index = idx
            break
    if cut_index is None:
        df = df.iloc[:cut_index]

    return df

def make_decisions(predictions_values, decision_number):
    # If predictions are empty exit and return none
    if len(predictions_values) == 0:
        return None
    # A list containing all the decisions made by the algorithm
    decisions_list = []
    # A list containing the values at the index 'i' for each prediction values column
    column_list = []
    # Counts the decisions made by the algorithm (only counts BUY and SELL)
    buy_sell_decisions_count = 0

    for i in range(len(predictions_values[0])):
        # The count for each positive or negative values in columns
        positives_counter = 0
        negatives_counter = 0

        # Populate current column list
        for values in predictions_values:
            column_list.append(values[i])

        # Counts positive and negative values
        for value in column_list:
            if value == 1:
                positives_counter += 1
            else:
                negatives_counter += 1

        # Decide action based on decision threshold
        if positives_counter > negatives_counter and positives_counter >= decision_number:
            flag = 'BUY'
            buy_sell_decisions_count += 1
        elif positives_counter < negatives_counter and negatives_counter >= decision_number:
            flag = 'SELL'
            buy_sell_decisions_count += 1
        else:
            flag = 'HOLD'

        decisions_list.append(flag)
        column_list = []

    return decisions_list, buy_sell_decisions_count

def get_date_object(date_string, date_format = "%d/%m/%Y"):
    return time.strptime(date_string, date_format)


def slice_predictions(predictions_list):
    max_start_date = None
    min_end_date = None
    for index, prediction in enumerate(predictions_list):
        start_date = get_date_object(prediction.dates[0])
        end_date = get_date_object(prediction.dates[-1])

        if max_start_date is None or start_date > max_start_date:
            max_start_date = start_date
        if min_end_date is None or end_date < min_end_date:
            min_end_date = end_date

    prediction_dates_start_cuts = []
    prediction_dates_end_cuts = []
    for index, prediction in enumerate(predictions_list):
        start_cut = False
        end_cut = False
        for date_index, date in enumerate(prediction.dates):
            if get_date_object(date) >= max_start_date and start_cut == False:
                prediction_dates_start_cuts.append(date_index)
                start_cut = True
            if get_date_object(date) >= min_end_date and start_cut == True and end_cut == False:
                prediction_dates_end_cuts.append(date_index)
                end_cut = True

    for prediction, start_cut, end_cut in zip(predictions_list, prediction_dates_start_cuts, prediction_dates_end_cuts):
        prediction.dates = prediction.dates[start_cut:end_cut+1]
        prediction.predictions = prediction.predictions[start_cut:end_cut+1]

    return predictions_list, max_start_date, min_end_date
#------------------------------------------------

def accumulator (f_train_preparation,accumulator, dollars, predictions, l_train_preparation, open):
    acc = []
    for i in range(0, len(predictions)):
        if predictions[i] == l_train_preparation[i]:
            accumulator += abs((f_train_preparation[i] * dollars))
        elif predictions[i] == 0:
            pass
        else:
            accumulator -= abs((f_train_preparation[i] * dollars))
        acc.append(accumulator)
    return acc

def plotMaxGraph(f_train_preparation,accumulatore,dollars,best_pred,l_train_preparation, best_mean, best_out_params, dataset_name,
                 best_in_params, open):
    train_size = best_out_params['train_size']
    test_size = best_out_params['test_size']
    feature_size = best_out_params['feature_size']
    total_length = best_out_params['in_sample_size']
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

def create_destination_folder(destination_path):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
#------------------------------------------------------------------------------


# Main prediction method.
# - Parameter predictors_number: The number of predictors to use (taken from the
#   configuration file in end/mdd order)
# - Parameter decision_threshold: The percentage threshold that indicates how many predictors have
#   to agree in order to trust their prediction
# - Parameter dataset_file_path: The dataset file path
# - Parameter dataset_name: The dataset name
# - Parameter destination_path: The destination folder path
def combined_predict(predictors_number, decision_threshold, dataset_file_path, dataset_name, destination_path):
    #-----------------------#
    # Constants definitions #
    #-----------------------#
    # The configuration file path (contains predictors parameters ordered by end/mdd)
    config_file_path = 'sp500configurazionicorte.csv'
    # The accuracy threshold used to determine the algorithm effectiveness
    accuracy_threshold = 0.50
    # The value in dollars of each BUY and SELL decision
    buy_sell_dollars = 50
    # The initial capital in dollars
    start_capital = 100000

    #-----------------------#
    # Variables definitions #
    #-----------------------#

    # A list that will contain all the Prediction objects generated by the predict function
    all_predictions_list = []

    # A list containing the parameters of the n (predictors_number) config file entries
    parameters_list = None

    # A list containing the accuracy value for each one of the config file entries
    accuracies_list = None

    #-----------------------#
    #      PREDICTION       #
    #-----------------------#

    print("---------------------------------")
    print("STARTING {} PREDICTIONS".format(destination_path))
    print("---------------------------------")

    # Loads the configuration file into a Dataframe object
    print("- Loading configuration file ({})".format(config_file_path))
    config_dataframe = pd.read_csv(config_file_path)

    # Extracts parameters and accuracies values from the configuration
    print("- Extracting configuration...".format(config_file_path))
    parameters_list, accuracies_list = extract_parameters(config_dataframe, predictors_number)
    print("- Configuration extracted!".format(config_file_path))

    # Loads the dataframe
    print("- Loading dataframe...".format(config_file_path))
    dataframe = load_dataframe(dataset_file_path)

    # Extracts dataframe features
    date = dataframe['DATE']
    open_value = dataframe['OPEN']
    close_value = dataframe['CLOSE']
    max_value = dataframe['HIGH']
    min_value = dataframe['LOW']
    volume = dataframe['VOLUME']
    diff = get_diff_column(dataframe)
    sign = get_sign_column_from_diff(diff)
    print("- Dataframe loaded!".format(config_file_path))

    # Make predictions for each set of parameters in parameters_list
    print("- Computing predictions using {} sets of parameters...".format(predictors_number))
    for index, parameters in enumerate(parameters_list):
        print("\nComputing prediction {} of {}".format(index+1, predictors_number))
        all_predictions_list.append(predict(dataframe, parameters, dataset_name))

    print("- Succesfully computed {} predictions.".format(len(all_predictions_list)))

    all_predictions_list, max_start_date, min_end_date = slice_predictions(all_predictions_list)

    # Cut dataframe
    cut_start_index = None
    cut_end_index = None
    for index, obj_date in enumerate(date):
        if obj_date == time.strftime('%d/%m/%Y', max_start_date):
            cut_start_index = index
        elif obj_date == time.strftime('%d/%m/%Y', min_end_date):
            cut_end_index = index

    date = date[cut_start_index:cut_end_index+1]
    open_value = open_value[cut_start_index:cut_end_index+1]
    sign = sign[cut_start_index:cut_end_index+1]
    diff = diff[cut_start_index:cut_end_index+1]
    close_value = close_value[cut_start_index:cut_end_index+1]
    max_value = max_value[cut_start_index:cut_end_index+1]
    min_value = min_value[cut_start_index:cut_end_index+1]
    volume = volume[cut_start_index:cut_end_index+1]
    print("- Predictions sliced and aligned (length: {})".format(cut_end_index - cut_start_index + 1))

    # Get the list of predicted values for each one of the predictions
    predictions_values = [obj.predictions for obj in all_predictions_list]

    # Determines which is the number of predictors that have to agree in order to accept their prediction
    decision_number = int(decision_threshold * predictors_number)

    # Makes the decisions and returns them as a list, along with a value
    # how many actions (BUY OR SELL) were taken
    print("- Making final decisions...")
    decisions_list, buy_sell_decisions_count = make_decisions(predictions_values, decision_number)
    print("- Decisions made!")

    #----------------#
    #   STATISTICS   #
    #----------------#

    # Computes the percentage of BUY and SELL in the total decisions made
    coverage_percentage = buy_sell_decisions_count/ len(decisions_list) * 100

    # Computes the number of accuracies under the mean accuracy of all predictors combined
    average_accuracies = np.mean(accuracies_list)
    n_accuracy_under = 0
    for element in accuracies_list:
        if element < accuracy_threshold:
            n_accuracy_under +=1

    # Computes the sign (should match diff) of each decision made, and counts the hold actions
    hold_decisions_count = 0
    decisions_signs = []
    for decision in decisions_list:
        if decision == 'BUY':
            decisions_signs.append(1)
        elif decision == 'SELL':
            decisions_signs.append(-1)
        else:
            hold_decisions_count += 1
            decisions_signs.append(0)

    # Counts the number of correct and wrong decisions made (by comparing the decision signs
    # with the actual dataset signs)
    correct_decisions = 0
    for i in range(0,len(decisions_signs)):
        if decisions_signs[i] == sign[i]:
            correct_decisions += 1
    wrong_decisions = (len(decisions_signs) - hold_decisions_count) - correct_decisions

    print('decision',decisions_signs)
    # Computes the covariance using the decision_signs
    covariance = calculate_covariance(decisions_signs, start_capital, buy_sell_dollars, diff, sign)

    # Computes mdd and end values
    mdd = maxDrawdown(covariance)
    end = covariance[(len(covariance)-1)] - 100000

    # Computes the final algorithm accuracy
    final_accuracy = get_final_accuracy(hold_decisions_count, wrong_decisions, len(decisions_list))

    create_destination_folder(destination_path)

    # Save the results to a csv file
    results_to_csv(coverage_percentage,
                   average_accuracies,
                   final_accuracy,
                   n_accuracy_under,
                   predictors_number,
                   dataset_name,
                   mdd,
                   decision_threshold,
                   end)

    # Plots and save graph of the computed covariance
    plotCovariance(destination_path,
                   covariance,
                   dataset_name,
                   dataset_name,
                   average_accuracies,
                   coverage_percentage,
                   n_accuracy_under,
                   predictors_number,
                   final_accuracy)

    # Save values to CSV file
    save_values_as_csv(destination_path, date, decisions_list, sign, open_value, max_value, min_value, close_value, volume, diff)

    # Prints statistics
    print("\n STATISTICS:\n ----------")
    print('- Coverage: ', coverage_percentage)
    print('- Average accuracies: ', average_accuracies)
    print('- Accuracies under the threshold: ', n_accuracy_under)
    print('- Final accuracy: ', final_accuracy)
    print('- Errors: {}/{}'.format(wrong_decisions, len(decisions_signs)))
    print("-------------------------------")
    print("- Predictions completed succesfully!")

def get_files_in_path(dest_path):
    return [ f for f in os.listdir(dest_path) if os.path.isfile(os.path.join(dest_path, f))]

def get_csv_in_path(dest_path):
    all_files = get_files_in_path(dest_path)
    return [os.path.splitext(f)[0] for f in all_files if os.path.splitext(f)[1] == ".csv"]

#------------------------------------------------------------------------------------------------------------------

datasets_path = "Datasets"
datasets_paths = get_csv_in_path(datasets_path)
for dataset_name, dataset in zip(datasets_paths, [ "{}/{}.csv".format(datasets_path, dataset) for dataset in datasets_paths]):
    combined_predict(10, 1.0, dataset, dataset_name, "{} (10 - 100)".format(dataset_name))
    combined_predict(10, 0.9, dataset, dataset_name, "{} (10 - 90)".format(dataset_name))
    combined_predict(10, 0.8, dataset, dataset_name, "{} (10 - 80)".format(dataset_name))
    combined_predict(20, 1.0, dataset, dataset_name, "{} (20 - 100)".format(dataset_name))
    combined_predict(20, 0.9, dataset, dataset_name, "{} (20 - 90)".format(dataset_name))
    combined_predict(20, 0.8, dataset, dataset_name, "{} (20 - 80)".format(dataset_name))
