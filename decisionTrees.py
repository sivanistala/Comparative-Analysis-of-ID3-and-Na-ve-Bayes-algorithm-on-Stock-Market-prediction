import DecisionTree
import recheck
import Node
import matplotlib.pyplot as plt
from Tkinter import *
import tkFileDialog
from tkMessageBox import *
from PIL import ImageTk,Image
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from math import sqrt
import sklearn
from sklearn.metrics import mean_squared_error

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
	#Insert input file
	"""
	IMPORTANT: Change this file path to change training data
	"""
	print ""

def id3(trainfilename,testfilename,originalValuefilename):
	trainingFile = open(trainfilename)
	"""
	IMPORTANT: Change this variable too change target attribute
	"""
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
		#print ("entry%s = %s" % (count, result))
		prediction.append(result)
	#showinfo("ID3 Algorithm","Predictions are done"+str(prediction[0]))


	total_predictions = len(prediction)
	predicted_2 = []
	i = 0
	while i < total_predictions:
		temp = float(prediction[i])
		predicted_2.append(temp)
		i = i+1
	#showinfo("ID3 Algorithm","Predictions are done"+str(temp)+str(type(temp)))
	open_values = gettingOriginalOpenValues(originalValuefilename)
	original_close_values = gettingOriginalCloseValues(originalValuefilename)
	#print open_values
	#print original_close_values
	# print predicted_2
	# plotting
	plt.title("Results for given dataset using ID3 Algorithm")
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
	fig.savefig('test_result_id3.jpg', dpi=100)
	#showinfo("Naive Bayes Algorithm","Plotting Completed")'''


	x = Image.open("E:\\4.2\Final Year Project\Code\Complete Project\\test_result_id3.jpg")
	y = ImageTk.PhotoImage(x)

	label6 = Label(image=y)
	label6.image = y
	label6.place(x=50, y=290)

	#Generate program
	'''
	file = open('program.py', 'w')
	file.write("import Node\n\n")
	file.write("import recheck\n\n")
	#open input file
	file.write("data = [[]]\n")
	"""
	IMPORTANT: Change this file path to change testing data
	"""
	file.write("f = open('AAPLTest.csv')\n")
	#gather data
	file.write("for line in f:\n\tline = line.strip(\"\\r\\n\")\n\tdata.append(line.split(','))\n")
	file.write("data.remove([])\n")
	#input dictionary tree
	file.write("tree = %s\n" % str(tree))
	file.write("attributes = %s\n" % str(attributes))
	file.write("prediction = []\n")
	file.write("count = 0\n")
	file.write("for entry in data:\n")
	file.write("\tcount += 1\n")
	#copy dictionary
	file.write("\ttempDict = tree.copy()\n")
	file.write("\tresult = \"\"\n")
	#generate actual tree
	file.write("\twhile(isinstance(tempDict, dict)):\n")
	file.write("\t\troot = Node.Node(tempDict.keys()[0], tempDict[tempDict.keys()[0]])\n")
	file.write("\t\ttempDict = tempDict[tempDict.keys()[0]]\n")
	#this must be attribute
	file.write("\t\tindex = attributes.index(root.value)\n")
	file.write("\t\tvalue = entry[index]\n")
	#ensure that key exists
	file.write("\t\tif(value in tempDict.keys()):\n")
	file.write("\t\t\tchild = Node.Node(value, tempDict[value])\n")
	file.write("\t\t\tresult = tempDict[value]\n")
	file.write("\t\t\ttempDict = tempDict[value]\n")
	#otherwise, break
	file.write("\t\telse:\n")


	#file.write("\t\t\t#print \"can't process input %s\" % count\n")
	file.write("\t\t\tresult = recheck.some_func(value)\n")
	file.write("\t\t\tbreak\n")
	#print solutions
	file.write("\t#print (\"entry%s = %s\" % (count, result))\n")
	file.write("\tprediction.append(result)\n")
	print "written program"
	'''

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
        result =  "Conclusion: Model tends to ideal-forecast"#, with an average absolute error of "+str(MAD) +" units"
    elif model_tend == 1:
        result =  "Conclusion: Model tends to under-forecast"#, with an average absolute error of "+str(MAD) +" units"
    if model_tend == 2:
        result = "Conclusion: Model tends to over-forecast"#, with an average absolute error of "+str(MAD) +" units"
    TS = sum(error)/MAD
    #print "Tracking Signal: "+str(TS)
    '''
    if -4<TS<4:
        print "Model is working correctly within error limits with"+str(TS)
    else:
        print "Model is not working within error limits"+str(TS)
	print "Root mean Square deviation"+str(rms)
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

if __name__ == '__main__':
	main()