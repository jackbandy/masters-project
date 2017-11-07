'''
provide_labels.py
Run this script to add ground-truth labels to the database
Optionally, provide labels for images chosen by the model
'''

import tkinter as tk
from tkinter import ttk

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid()
        self.createWidgets()

    def createWidgets(self):
        imgPath = r"and-raw.gif"
        self.quitButton = tk.Button(self, text='Save labels and exit', command=self.quit)
        self.skipButton = tk.Button(self, text='Skip this sample', command=self.skip)
        self.badButton = tk.Button(self, text='This shows more than one word', command=self.skip)
        self.submitButton = tk.Button(self, text='Submit', command=self.submit)
        
        self.labelInput = tk.Entry(self, width=30)
        self.testImage = tk.PhotoImage(file= imgPath)
        self.wordPic = tk.Label(self)
        self.wordPic['image'] = self.testImage

        self.wordPic.grid(row=0,rowspan=5,ipady=10)
        self.labelInput.grid(row=6,column=0)
        self.submitButton.grid(row=6,column=1)
        self.skipButton.grid(row=7)
        self.badButton.grid(row=8,rowspan=2)
        self.quitButton.grid(row=10)

    def skip(self):
        print("Skipping")

    def submit(self):
        print("Submitting")

    def chooseImageForLabel(self):
        print("Choosing new image...")

app = Application()
app.master.title('Provide labels')
app.mainloop()
