'''
provide_labels.py
Run this script to add ground-truth labels to the database
Optionally, provide labels for images chosen by the model
'''

import tkinter as tk


class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid()
        self.createWidgets()

    def createWidgets(self):
        imgPath = r"test.gif"
        self.quitButton = tk.Button(self, text='Quit', command=self.quit)
        self.skipButton = tk.Button(self, text='Skip this sample', command=self.skip)
        self.testImage = tk.PhotoImage(file= imgPath)
        self.label = tk.Label(image = self.testImage)
        self.label.grid()
        self.skipButton.grid()
        self.quitButton.grid()

    def skip(self):
        print("Skipping")

app = Application()
app.master.title('Provide labels')
app.mainloop()
