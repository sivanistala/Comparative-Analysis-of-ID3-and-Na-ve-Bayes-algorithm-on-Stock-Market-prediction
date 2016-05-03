#Naive Bayes implementation
import csv
import random
import math
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from Tkinter import *
import tkFileDialog
from tkMessageBox import *
from PIL import ImageTk,Image
import sklearn
from sklearn.metrics import mean_squared_error

def loadTrainCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    i = 1
    while i < len(dataset) :
        dataset[i] = [float(x) for x in dataset[i]]
        i = i+1
    return dataset

def loadTestCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    i = 0
    while i < len(dataset) :
        dataset[i] = [float(x) for x in dataset[i]]
        i = i+1
    return dataset

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def summarize(dataset):
    #print type(attribute) 'type tuple'
    attributeList = list(dataset) #converted to list
    summaries = [(mean(attributeList), stdev(attributeList)) for attributeList in zip(*dataset)]
    del summaries[-1]
    #print "Summaries successfully sent"
    return summaries

def add(attributeList):
    result = 0
    i = 0
    for i in range(len(attributeList)):
            if type(attributeList[i]) is str :
                result = result
            else:
                result = result + attributeList[i]
    return result

def mean(numbers):
    return add(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = 0
    for x in numbers:
        if type(x) is str:
            variance = variance
        else:
            variance = add([pow(x-avg,2) for x in numbers])/float(len(numbers))
    return math.sqrt(variance)

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    #print type(separated) 'dict type'
    summaries = {}
    #print type(summaries) 'dict type'
    for classValue, instances in separated.iteritems():
        #print type(classValue) 'float type'
        #print "Class value: " + str(classValue)+" instances: "+str(instances) 'values of class and their corresponding values'
        #print type(instances) 'list type'
        summaries[classValue] = summarize(instances)
    return summaries

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    #print type(probabilities) 'dict type'
    #print summaries
    #print type(summaries) 'dict type'
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mn, std = classSummaries[i]
            if std == 0:
                std = std + 0.01
            x = inputVector[i]
            probabilities[classValue] = probabilities[classValue] * calculateProbability(x, mn, std)
    return probabilities

def calculateProbability(x, mn, std):
    #print str(x) + str(type(x)) #'value from the test data set'
    #print str(mn) + str(type(mn))
    #print str(std) + str(type(std))
    a = -math.pow(x-mn,2)
    #print "a "+str(a)
    b = 2* math.pow(std,2)
    #print "b "+str(b)
    exponent = math.exp(a/b)
    #print "exponent "+str(exponent)
    #exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    #return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    return ((1 / (math.sqrt(2*math.pi) * std)) * exponent)

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def naive_bayes(trainfilename,testfilename,originalValuefilename):
    #trainfilename = 'AAPLTraining.csv'
    trainingdataset = loadTrainCsv(trainfilename)
    #testfilename = 'AAPLTest.csv'
    testdataset = loadTestCsv(testfilename)
    summaries = summarizeByClass(trainingdataset)
    predictions = getPredictions(summaries, testdataset)
    #print "Predictions made"
    #print predictions
    predicted_2=predictions
    open_values = gettingOriginalOpenValues(originalValuefilename)
    original_close_values = gettingOriginalCloseValues(originalValuefilename)
    #print open_values
    #print original_close_values
    #print predicted_2
    #plotting
    plt.title("Results for given dataset using Naive Bayes Algorithm",fontsize="10")
    plt.plot(open_values,predicted_2,'r.',markersize=np.sqrt(150.),label ='Naive Bayes Prediction')
    plt.plot(open_values,original_close_values,'b.',markersize=np.sqrt(100.),label = 'Orignial Values')
    plt.legend(loc='upper left')
    plt.xlabel("Open Values")
    plt.ylabel("Close Values")
    plt.grid()
    #plt.show()
    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    ax=plt.subplot(111)
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09),fancybox=True, shadow=True, ncol=5,fontsize="10")
    fig.savefig('test_result_naivebayes.jpg', dpi=100)
    #showinfo("Naive Bayes Algorithm","Plotting Completed")'''


    x = Image.open("E:\\4.2\Final Year Project\Code\Complete Project\\test_result_naivebayes.jpg")
    y = ImageTk.PhotoImage(x)

    label6 = Label(image=y)
    label6.image = y
    label6.place(x=50, y=290)
    result=accuracy_calculation(original_close_values,predicted_2)
    return result

def accuracy_calculation(original_close_values,predicted_2):
    #accuracy calculation
    rms = sqrt(mean_squared_error(original_close_values, predicted_2))
    #print "Mean Squared Error: "+str(rms)

    #print "Total number of observations: "+ str(len(original))


    i = 0
    error = []
    abs_error = []
    while i < len(original_close_values):
        error.append(original_close_values[i] - predicted_2[i])
        abs_error.append(abs(original_close_values[i] - predicted_2[i]))
        i = i+1

    MFE = sum(error)/len(original_close_values)
    #print "Mean Forecast Error: "+ str(MFE)

    model_tend = 0
    if MFE < 0:
        model_tend = 2
    elif MFE > 0:
        model_tend = 1
    else:
        model_tend = 0

    MAD = sum(abs_error)/len(original_close_values)
    #print "Mean Absolute Deviation: "+ str(MAD)

    if model_tend == 0:
        result = "Conclusion: Model tends to ideal-forecast"#, with an average absolute error of "+str(MAD) +" units"
    elif model_tend == 1:
        result = "Conclusion: Model tends to under-forecast"#, with an average absolute error of "+str(MAD) +" units"
    if model_tend == 2:
        result = "Conclusion: Model tends to over-forecast"#, with an average absolute error of "+str(MAD) +" units"
    TS = sum(error)/MAD
    #print "Tracking Signal: "+str(TS)
    #print "Root mean square"+str(rms)
    '''
    if -4<TS<4:
        print "Model is working correctly within error limits"+str(TS)
    else:
        print "Model is not working within error limits"+str(TS)
    '''

    #calculating forecast errors
    forecast_error = []
    i = 0
    while i < len(abs_error):
        forecast_error.append((abs_error[i]/original_close_values[i])*100)
        i = i+1

    #print forecast_error
    mean_forecast_error = sum(forecast_error)/len(forecast_error)
    #print "Mean absolute error:"+str(mean_forecast_error)
    forecast_accuracy = max(0,100 - mean_forecast_error)
    #print "Forecast accuracy:"+str(forecast_accuracy)
    #print "Mean absolute Percentage Error: "+str(forecast_accuracy)
    return str(forecast_accuracy) +"\n"+ result+"\n"+"Root mean square deviation:"+str(rms)


def loadingOriginalCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    i = 0
    while i < len(dataset) :
        dataset[i] = [float(x) for x in dataset[i]]
        i = i+1
    return dataset

def gettingOriginalOpenValues(filename):
    #print "Plotting"
    originalValueSet = loadingOriginalCsv(filename)
    originalValueSetLength = len(originalValueSet)
    openValues = []
    i = 0
    while i < originalValueSetLength:
        openValues.append(originalValueSet[i][0])
        i = i+1
    return openValues

def gettingOriginalCloseValues(filename):
    closeValueSet = loadingOriginalCsv(filename)
    closeValueSetLength = len(closeValueSet)
    closeValues = []
    i = 0
    while i < closeValueSetLength:
        closeValues.append(closeValueSet[i][2])
        i = i+1
    return closeValues

def main():
    print ""
main()