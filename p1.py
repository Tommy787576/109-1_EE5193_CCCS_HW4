from pyspark.mllib.classification import LogisticRegressionWithSGD
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint

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
model = LogisticRegressionWithSGD.train(parsedData)

# Predict the first elem will be actual data and the second 
# item will be the prediction of the model
labelsAndPreds = parsedData.map(lambda point: (int(point.label), 
       model.predict(point.features)))


# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda vp: (vp[0] != vp[1])).count() / float(parsedData.count())

# Print some stuff
print("Training Error = " + str(trainErr))