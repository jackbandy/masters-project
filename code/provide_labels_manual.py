'''
provide_labels.py
Run this script to add ground-truth labels to the database
Optionally, provide labels for images chosen by the model
'''

import tkinter as tk
import os
import numpy as np
from PIL import Image, ImageTk
from tkinter import ttk


class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid()
        self.createWidgets()
        self.current_cluster = 0
        self.images = os.listdir('test-labels/')
        self.images.sort()

        if 'DS_Store' in self.images[0]:
            del self.images[0]

        if 'npy' in self.images[-1]:
            self.cluster_data = np.load('test-labels/{}'.format(self.images[-1]))
            del self.images[-1]


    def createWidgets(self):
        imgPath = r"data/and-raw.gif"
        self.quitButton = tk.Button(self, text='Save labels and exit', command=self.quit)
        self.skipButton = tk.Button(self, text='Skip this sample', command=self.skip)
        self.badButton = tk.Button(self, text='This shows more than one word', command=self.skip)
        self.submitButton = tk.Button(self, text='Submit', command=self.submit)
        
        self.labelInput = tk.Entry(self, width=30)
        self.testImage = tk.PhotoImage(file= imgPath)
        self.wordPic = tk.Label(self)
        self.wordPic['image'] = self.testImage

        self.wordPic.grid(row=0,rowspan=5,columnspan=2,ipady=10)
        self.labelInput.grid(row=6,column=0)
        self.submitButton.grid(row=6,column=1)
        self.skipButton.grid(row=7)
        self.badButton.grid(row=8,rowspan=2)
        self.quitButton.grid(row=10)


    def skip(self):
        self.showNextWord()
        print("Skipping")


    def submit(self):
        label = self.labelInput.get()
        self.cluster_data[self.current_cluster]['word'] = label
        self.labelInput.delete(0, len(label))
        self.showNextWord()
        np.save('labeled-clusters.npy', self.cluster_data)
        print("Label: -{}-".format(label))
        print("Submitting")


    def showNextWord(self):
        image_name = self.getWordForCluster(cluster_num=self.current_cluster)
        image_path = 'test-labels/{}'.format(image_name)

        self.clusterImage = ImageTk.PhotoImage(Image.open(image_path))
        self.wordPic = tk.Label(self)
        self.wordPic['image'] = self.clusterImage
        self.wordPic.grid(row=0,rowspan=5,ipady=10)
        self.current_cluster += 1



    def getWordForCluster(self, cluster_num):
        i = 0
        image_name = self.images[i]
        # assumes first character of file name is cluster number
        while int(image_name[0]) < cluster_num:
            i += 1
            image_name = self.images[i]

        return image_name 



    def searchForMatches(self, image_to_match, images_to_search):
        pass

    
    def chooseImageForLabel(self):
        print("Choosing new image...")



app = Application()
app.master.title('Provide labels')
app.mainloop()
