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

'''''
# Add a diff column if that does not exist
def add_diff_column(df):
    tmp = df['Close'].tolist()
    tmp2 = df['Open'].tolist()
    diff = []
    for i in range(0, len(tmp)):
        diff.append(tmp[i] - tmp2[i])

    return diff
'''''

# Add a diff column if that does not exist
def get_features_from_csv(df):
    tmp = df['CLOSE'].tolist()
    tmp2 = df['OPEN'].tolist()
    features = []
    for i in range(0, len(tmp)):
        features.append(tmp[i] - tmp2[i])

    return features

# If the sign column does not exists, calculate it
def get_labels_from_features(df):
    sign = []
    for element in df:
        if element >= 0:
            sign.append(1)
        else:
            sign.append(-1)

    return sign


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


def csv_to_test(open, f_train_preparation, predictions, acc):
    open = np.array(open)
    open.flatten()

    df2 = pd.DataFrame()
    df2['Diff'] = f_train_preparation[:200]
    df2['Predizioni'] = predictions[:200]
    df2['Equity'] = acc[:200]
    df2['ValoreMercato'] = open[:200]
    df2.to_csv('Test.csv', sep=',')


