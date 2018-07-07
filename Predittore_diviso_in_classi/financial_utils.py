from Prediction import Prediction as Prd
# Accumulator in order to create a series of values up or down
# Questo è l equity e l'equity che dico io è invece il profitto e l'equity è il capitale netto(tolte tutte le spese)

'''''''''
def accumulator (f_train_preparation,accumulator, dollars, predictions, l_train_preparation, open):
    acc = []
    for i in range(0, len(predictions)):
        if predictions[i] == l_train_preparation[i]:
            accumulator += abs((f_train_preparation[i] * dollars))
        else:
            accumulator -= abs((f_train_preparation[i] * dollars))
        acc.append(accumulator)
'''''

def get_features_and_labels(df):
    features = get_features_from_csv(df)
    labels = get_labels_from_features(diff)
    return features,labels

def get_market_value(df):
    # Valore del mercato giornaliero
    open = df.iloc[:, [1]].values.tolist()
    return open

def get_in_parameters(loss,n_estimators,max_depth,min_samples_leaf):
    in_param_grid = ParameterGrid(
        dict(loss=loss, n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf))
    return in_param_grid

def get_out_parameters(train_size,test_size,total_length,features_size):
    out_param_grid = ParameterGrid(
        dict(train_size=train_size, test_size=test_size, total_length=total_length, features_size=features_size))
    return out_param_grid

# Ex accumulator: allinea da solo features e labels con predictions
def get_equity(Prd):
    equity = []
    train_size = Prd.params['train_size']
    feature_size = Prd.params['feature_size']
    features = Prd.features[train_size+feature_size:]
    labels = Prd.labels[train_size+feature_size]

    for i in range(o, len(Prd.predictions)):
        if Prd.predictions[i] == labels[i]:
            tmp += abs((features[i] * dollars))
        else:
            tmp -= abs((features[i] * dollars))
        equity.append(tmp)
    return equity

def get_maxdrawdown(equity):
    # Calculate Max Drawdown
    maxValue = float(equity[0])
    drawdown = []

    for i in range(0, len(equity) - 1, 1):
        if (float(equity[i + 1])) > maxValue:
            maxValue = float(equity[i + 1])
        else:
            drawdown.append(abs(maxValue - equity[i + 1]))

    mdd = max(drawdown)
    return mdd

# Ex calculateMdd
def get_mdd_and_equity(Prd):
    equity = get_equity(Prd)
    mdd = get_maxdrawdown(equity)
    return mdd, equity

'''''''''
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
'''''