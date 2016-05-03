import Tkinter
from Tkinter import *
import tkFileDialog
from tkMessageBox import *
from PIL import ImageTk,Image
import decisionTrees
import bayesianNetworks
import id3_naive
root = Tk()

def selectfile1():
    fileName1 = tkFileDialog.askopenfilename(parent=root, title='Choose a Training Data File', initialdir='E:\\4.2\Final Year Project\Code\Complete Project')
    training_data_path.set(fileName1) #Populate the text field with the selected file
    return fileName1

def selectfile2():
    fileName2 = tkFileDialog.askopenfilename(parent=root, title='Choose a Testing Data File', initialdir='E:\\4.2\Final Year Project\Code\Complete Project')
    testing_data_path.set(fileName2) #Populate the text field with the selected file
    return fileName2

def selectfile3():
    fileName3 = tkFileDialog.askopenfilename(parent=root, title='Choose a Original Value Data File', initialdir='E:\\4.2\Final Year Project\Code\Complete Project')
    #training_data_path.set(fileName3) #Populate the text field with the selected file
    return fileName3

def bayes():
    filename1 = selectfile1()
    filename2 = selectfile2()
    filename3 = selectfile3()
    if filename1[-4:]==".csv" and filename2[-4:]==".csv" and filename3[-4:]==".csv":
        index1 = filename1.rfind("/")
        index2 = filename2.rfind("/")
        index3 = filename3.rfind("/")
        accuracy = bayesianNetworks.naive_bayes(filename1[index1+1:],filename2[index2+1:],filename3[index3+1:])
        label7 = Label(root, text="Forecast Accuracy:"+accuracy,fg="blue",bg="lavender",justify=LEFT,font=("Helvetica", 15,"italic")).place(x=900,y=290)
        showinfo("Status","Graph generated")
    else:
        showerror("Extension Error","All the files should be in .csv format")

def id3():
    filename1 = selectfile1()
    filename2 = selectfile2()
    filename3 = selectfile3()
    if filename1[-4:]==".csv" and filename2[-4:]==".csv" and filename3[-4:]==".csv":
        index1 = filename1.rfind("/")
        index2 = filename2.rfind("/")
        index3 = filename3.rfind("/")
        accuracy = decisionTrees.id3(filename1[index1+1:],filename2[index2+1:],filename3[index3+1:])
        showinfo("Status","Graph Generated")
        label7 = Label(root, text="Forecast Accuracy:"+accuracy,fg="blue",bg="lavender",justify=LEFT,font=("Helvetica", 15,"italic")).place(x=900,y=290)
    else:
        showerror("Extension Error","All the files should be in .csv format")

def complete():
    filename1 = selectfile1()
    filename2 = selectfile2()
    filename3 = selectfile3()
    if filename1[-4:]==".csv" and filename2[-4:]==".csv" and filename3[-4:]==".csv":
        index1 = filename1.rfind("/")
        index2 = filename2.rfind("/")
        index3 = filename3.rfind("/")
        accuracy=id3_naive.completeExexution(filename1[index1+1:],filename2[index2+1:],filename3[index3+1:])
        label7 = Label(root, text="Forecast Accuracy:"+accuracy,fg="blue",bg="lavender",justify=LEFT,font=("Helvetica", 15,"italic")).place(x=900,y=290)
        showinfo("Status","Graph Generated")
    else:
        showerror("Extension Error","All the files should be in .csv format")
'''
scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)
listbox = Listbox(root, yscrollcommand=scrollbar.set)
for i in range(1000):
    listbox.insert(END, str(i))
listbox.pack(side=LEFT, fill=BOTH)
scrollbar.config(command=listbox.yview)
'''

label1 = Label(root, text="Comparative Analysis of ID3 and Naive Bayes algorithm on Stock Market prediction",bg="lavender",justify=CENTER,font=("Helvetica", 25,"bold")).place(x=40,y=20)
label4 = Label(root, text="Upload your Stock Market datasets .csv extension here:",fg="blue",bg="lavender",justify=LEFT,font=("Helvetica", 15,"italic")).place(x=20,y=60)

label2 = Label(root, text="Training dataset ->",justify=LEFT,bg="lavender",font=("Helvetica", 12)).place(x=50,y=100)

training_data_path = StringVar(None)
entry1 = Entry(root, width ='60', textvariable=training_data_path).place(x=350,y=100)
#entry1 = Entry(root, textvariable=training_data_path).place(x=350,y=100)
button1 = Button(root, text="Browse", relief = 'raised', width=8, command=selectfile1, cursor='hand2').place(x=750,y=100)
#button1 = Button(root, text="Browse",command=lambda:training_data_path.set(tkFileDialog.askopenfilename())).place(x=550,y=100)

label3 = Label(root, text="Testing dataset ->",justify=LEFT,bg="lavender",font=("Helvetica", 12)).place(x=50,y=140)
testing_data_path = StringVar(None)
entry2 = Entry(root, width ='60', textvariable=testing_data_path).place(x=350,y=140)
#entry2 = Entry(root, textvariable=testing_data_path).place(x=350,y=140)
button2 = Button(root, text="Browse", relief = 'raised', width=8, command=selectfile2, cursor='hand2').place(x=750,y=140)
#button2 = Button(root, text="Browse",command=lambda:testing_data_path.set(tkFileDialog.askopenfilename())).place(x=550,y=140)

label5 = Label(root, text="Choose your algorithm:",justify=LEFT,fg="blue",bg="lavender",font=("Helvetica", 15,"italic")).place(x=20,y=180)
button3 = Button(root, text="ID3",command = id3,font=("Helvetica", 15)).place(x=50,y=220)
button4 = Button(root, text="NAIVE BAYES",command = bayes,font=("Helvetica", 15)).place(x=350,y=220)
button5 = Button(root, text="ID3 AND NAIVE BAYES",command=complete,font=("Helvetica", 15)).place(x=650,y=220)


root.minsize(width=1320, height=710)
root.configure(background="lavender")
root.wm_title("Comparitive Analysis of ID3 and Naive Bayes algorithm on Stock Market prediction")
root.mainloop()