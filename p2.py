from pyspark.mllib.classification import LogisticRegressionWithSGD
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint

class myLogistic:
    def __init__(self, train_data, eta, n_epoch):
        self.eta = eta
        n_features = len(train_data.first().features) + 1
        self.w = np.zeros(n_features)
        
        for epoch in range(n_epoch):
            w = train_data.map(lambda x: self.train_epoch(x)).reduce(lambda x, y: x + y)
            
            self.w += w / float(train_data.count())
            print("epoch = {}".format(epoch))
            print(self.w)
            
    def train_epoch(self, point):
        y = int(point.label)
        x = point.features
        x = np.insert(x, 0, 1)  # insert the bias
        if y == 0:
            y = -1
        delta = self.eta * (1.0 / (1.0 + np.exp(y * np.asscalar(np.dot(self.w, x))))) * (-y * x)
        return -delta
    
    def predict(self, x):
        x = np.insert(x, 0, 1)  # insert the bias
        return round(1.0 / (1.0 + np.exp(-np.asscalar(np.dot(self.w, x)))))

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """    
    feats = line.strip().split(",") 
    
    # labels must be at the beginning for LRSGD
    label = feats[len(feats) - 1] 
    feats = feats[: len(feats) - 1]
    features = [ float(feature) for feature in feats ] # need floats

    return LabeledPoint(label, features)

sc = SparkContext()

# Load and parse the data
data = sc.textFile("./hw4/data.txt")
parsedData = data.map(mapper)

# Train model
model = myLogistic(parsedData, eta = 1.5, n_epoch = 100)

# Predict the first elem will be actual data and the second 
# item will be the prediction of the model
labelsAndPreds = parsedData.map(lambda point: (int(point.label), model.predict(point.features)))

# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda vp: (vp[0] != vp[1])).count() / float(parsedData.count())

# Print some stuff
print("Training Error = " + str(trainErr))
