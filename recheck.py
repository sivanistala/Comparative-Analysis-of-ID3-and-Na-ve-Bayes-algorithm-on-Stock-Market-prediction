import csv
from math import sqrt

def mean(a):
    mn=sum(a)/len(a)
    return mn

def std_dev(a):
    b=[]
    mn=mean(a)
    for i in a:
        c=i-mn
        c=pow(c,2)
        b.append(c)
        variance = mean(b)
    return sqrt(variance)

def some_func(mi_value,trainfilename,testfilename):
    # open the file in universal line ending mode
    infile = open(trainfilename, 'rU')
    # read the file as a dictionary for each row ({header : value})
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]

    # extract the variables you want
    open_list_string = data['Open']
    close_list_string = data['Close']

    open_train = []
    close_train = []

    n=0
    while n < len(open_list_string):
        open_train.append(float(open_list_string[n]))
        close_train.append(float(close_list_string[n]))
        n=n+1
#print str(type(open_train[0])) + str(type(close_train[0])) + str(type(close_train)) + str(type(close_train)) + str(len(close_train)) + str(len(open_train))
    #print "Training Data Loaded"

    file = open(testfilename)
    reader = csv.reader(file)
    test_len=0
    open_test_str=[]
    open_test=[]
    for line in reader:
        open_test_str.append(line[0])
        test_len=test_len+1
    #print test_len
    n=0
    while n < test_len:
        open_test.append(float(open_test_str[n]))
        n=n+1
    #print "Testing data loaded"

    test=float(mi_value)
        #print "test value "+str(test)
    related_values=[]
    j=0
    while j < len(open_train):
        diff = float(test)-open_train[j]
        if -2<=diff<=2 :
            related_values.append(open_train[j])
        j=j+1
    #print "Related values of "+str(test)+" are "+str(related_values)

    if len(related_values)<1:
        related_values = open_train

    k=0
    diff_train_list=[]
    while k < len(related_values):
        if related_values[k] in open_train:
            #print "True" + str(related_values[k])
            index_in_open_train_list = open_train.index(related_values[k])
            diff_train = open_train[index_in_open_train_list] - close_train[index_in_open_train_list]
            diff_train_list.append(diff_train)
        k=k+1
    #print diff_train_list
    #print "Related values length " + str(len(related_values)) + " diff_train_list length "+ str(len(diff_train_list))

    #print std_dev(diff_train_list)

    close_list=[]
    for i in related_values:
        value=i-test
        if value<0:
            value = -value
            close_list.append(value)

    #print "RelatedList "+str(related_values)+" CloseList "+str(close_list)

    index_in_close_list =  close_list.index(min(close_list))
    #print close_list[index_in_close_list]
    closest_value =  related_values[index_in_close_list]
        #print closest_value
    index_in_open_train_list = open_train.index(closest_value)
    #print index_in_open_train_list
    corresponding_close= close_train[index_in_open_train_list]
    #print close_list[close_list.index(min(close_list))]
    return str( std_dev(diff_train_list) + corresponding_close )



if __name__ == '__main__':
    # do something
    some_func()