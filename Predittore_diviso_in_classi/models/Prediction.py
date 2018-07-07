
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
