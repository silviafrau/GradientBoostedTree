

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
        print('fx',element.params['train_size'])
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
    print('list',featureSizeList)
    df3['MIN_SAMPLES_LEAF'] = minSLList
    df3['MAX_DEPTH'] = maxDList
    df3['N_ESTIMATORS'] = n_estList
    df3['LOSS'] = 'DEVIANCE'

    df3.to_csv(dataset_name + 'listaDelleConfigurazioni.csv', sep=',')


def saveParams(prediction):
    with open(str(prediction.name)+str(prediction.params['feature_size'])+'DaysGraph.txt','w+') as f:
        f.write(prediction.getAllParams())
