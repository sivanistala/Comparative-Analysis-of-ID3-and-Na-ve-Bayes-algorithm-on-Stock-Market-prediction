import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from Tkinter import *
import tkFileDialog
from tkMessageBox import *
from PIL import ImageTk,Image
import DecisionTree
import recheck
import Node
from math import sqrt
import sklearn
from sklearn.metrics import mean_squared_error
import bayesianNetworks
import decisionTrees

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

def completeExexution(trainfilename,testfilename,originalValuefilename):

	#id3 algorithm
	trainingFile = open(trainfilename)
	target_attribute = "Close"

	data = [[]]
	for line in trainingFile:
		line = line.strip("\r\n")
		data.append(line.split(','))
	data.remove([])

	attributes = data[0]
	data.remove(attributes)
	#Run ID3
	tree = DecisionTree.makeTree(data, attributes, target_attribute, 0)
	#print "generated decision tree"

	data = [[]]
	testFile = open(testfilename)
	for line in testFile:
		line = line.strip("\r\n")
		data.append(line.split(','))
	data.remove([])
	#tree = str(tree)
	#tree = "%s\n" % str(tree)
	attributes = ['Open', 'High', 'Low', 'Close']
	prediction = []
	count = 0

	for entry in data:
		count += 1
		tempDict = tree.copy()
		result = ""
		while(isinstance(tempDict, dict)):
			root = Node.Node(tempDict.keys()[0], tempDict[tempDict.keys()[0]])
			tempDict = tempDict[tempDict.keys()[0]]
			index = attributes.index(root.value)
			value = entry[index]
			if(value in tempDict.keys()):
				child = Node.Node(value, tempDict[value])
				result = tempDict[value]
				tempDict = tempDict[value]
			else:
				result = recheck.some_func(value,trainfilename,testfilename)
				break

		prediction.append(result)

	total_predictions = len(prediction)
	predicted_2 = []
	i = 0
	while i < total_predictions:
		temp = float(prediction[i])
		predicted_2.append(temp)
		i = i+1

	#naive bayes algorithm
	trainingdataset = loadTrainCsv(trainfilename)
	testdataset = loadTestCsv(testfilename)
	summaries = summarizeByClass(trainingdataset)
	naive_predictions = getPredictions(summaries, testdataset)
	predicted_1=naive_predictions


	open_values = gettingOriginalOpenValues(originalValuefilename)
	original_close_values = gettingOriginalCloseValues(originalValuefilename)

	#print "Naive Predictions"+str(predicted_1)
	#print "ID3"+str(predicted_2)

	plt.title("Results for given dataset using ID3 & Naive Bayes Algorithm")
	plt.plot(open_values,predicted_1,'r.',markersize=np.sqrt(150.),label ='Naive Bayes Prediction')
	plt.plot(open_values,predicted_2,'g.',markersize=np.sqrt(150.),label ='ID3 Prediction')
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
	fig.savefig('test_result_id3_naivebayes.jpg', dpi=100)
	#showinfo("Naive Bayes Algorithm","Plotting Completed")'''


	x = Image.open("E:\\4.2\Final Year Project\Code\Complete Project\\test_result_id3_naivebayes.jpg")
	y = ImageTk.PhotoImage(x)

	label6 = Label(image=y)
	label6.image = y
	label6.place(x=50, y=290)

	result=accuracy_calculation(original_close_values,predicted_1,predicted_2)
	return result


def accuracy_calculation(original_close_values,predicted_1,predicted_2):
	result1 = bayesianNetworks.accuracy_calculation(original_close_values,predicted_1)
	result2 = decisionTrees.accuracy_calculation(original_close_values,predicted_2)
	return "\n"+"Naive Bayes Algorithm:"+result1+"\n"+"ID3 Algorithm:"+result2

def main():
	print "Complete Execution started"

main()
