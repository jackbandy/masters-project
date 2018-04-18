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
import PIL.Image
import PIL.ImageTk
import util
import pdb
from sklearn.metrics.pairwise import pairwise_distances
import pickle
import cluster


#WORD_IMAGES_PATH = '../gw-data/data/word_images_normalized/'
WORD_IMAGES_PATH = '../data/norm_auto_segmented_samples/'
WORD_LABELS_PATH = '../data/word_labels.pkl'
#ORDER = 'random'
ORDER = 'common'
#ORDER = 'rare'
#ORDER = 'appearance'


class Application(tk.Frame):
    def __init__(self, master=None):
        print("Initializing...")
        # extract features
        samples, _, _= util.collectSamples(WORD_IMAGES_PATH, binarize=False,
                scale_to_fill=True, fixed_max_width=624, fixed_max_height=128)
        self.sample_features = util.getHogForSamples(samples, scale=2)
        self.current_cluster = 0
        self.current_index = 0
        # distance matrix
        self.distances = pairwise_distances(self.sample_features, metric='cosine')

        self.images = os.listdir(WORD_IMAGES_PATH)
        self.images.sort()
        if 'DS_Store' in self.images[0]:
            del self.images[0]
        self.n_images = len(self.images)

        if ORDER == 'common':
            self.images = self.sortWordsByCommonality()
        elif ORDER == 'rare':
            self.images = self.sortWordsByCommonality(reverse=True)

        try:
            self.labels = pickle.load(open(WORD_LABELS_PATH))
        except Exception:
            self.labels = dict()

        tk.Frame.__init__(self, master)
        self.grid()
        self.createWidgets()



    def createWidgets(self):
        imgPath = r"{}/{}".format(WORD_IMAGES_PATH, self.images[self.current_index])
        im = PIL.Image.open(imgPath)
        self.raw_word_image = PIL.ImageTk.PhotoImage(im)

        self.quitButton = tk.Button(self, text='Save labels and exit', command=self.quit)
        self.skipButton = tk.Button(self, text='Skip this sample', command=self.skip)
        self.searchButton = tk.Button(self, text='Search for similar samples', command=self.search)
        self.badButton = tk.Button(self, text='This shows more than one word', command=self.skip)
        self.submitButton = tk.Button(self, text='Submit', command=self.submit)
        
        self.labelInput = tk.Entry(self, width=30)
        self.wordPic = tk.Label(self)
        self.wordPic['image'] = self.raw_word_image

        self.wordPic.grid(row=0,rowspan=5,columnspan=2,ipady=10)
        self.labelInput.grid(row=6,column=0)
        self.submitButton.grid(row=6,column=1)
        self.skipButton.grid(row=7)
        self.searchButton.grid(row=8)
        self.badButton.grid(row=9)
        self.quitButton.grid(row=10)



    def skip(self):
        self.showNextWord()
        print("Skipping")



    def sortWordsByCommonality(self, reverse=False):
        new_images = []
        n_clusters = int(self.sample_features.shape[0] / 2)
        print("Clustering to determine most common words...")
        preds = cluster.predictAgglomNClusters(self.sample_features, n_clusters)
        counts = np.bincount(preds)
        print("Sorting by most common...")
        self.starting_inds = [0]
        while len(new_images) < len(self.images):
            most_common = np.argmax(counts)
            count = counts[most_common]
            self.starting_inds.append(self.starting_inds[-1] + count)
            if reverse:
                most_common = np.argmin(counts)
            ims = np.where(preds==most_common)[0]
            for ind in ims:
                new_images.append(self.images[ind])
            counts[most_common] = 0
            if reverse:
                counts[most_common] = np.max(counts) + 1

        return new_images



    def submit(self):
        label = self.labelInput.get()
        to_put = dict()
        to_put['label'] = label
        to_put['segmentation'] = 'accepted'
        self.labels[self.images[self.current_index]] = to_put
        '''
        self.cluster_data[self.current_cluster]['word'] = label
        np.save('labeled-clusters.npy', self.cluster_data)
        '''
        self.labelInput.delete(0, len(label))
        self.showNextWord()
        print("Label: -{}-".format(label))
        print("Submitting")
        with open(WORD_LABELS_PATH, 'wb') as handle:
            pickle.dump(self.labels, handle)



    def showNextWord(self):
        if ORDER == 'random':
            self.current_index = np.random.randint(len(self.images))
        if ORDER == 'common':
            current_cluster = 0
            while self.starting_inds[current_cluster] < self.current_index:
                current_cluster += 1
            current_cluster +=1
            self.current_index = self.starting_inds[current_cluster]
        else:
            self.current_index += 1
        imgPath = r"{}/{}".format(WORD_IMAGES_PATH, self.images[self.current_index])
        im = PIL.Image.open(imgPath)
        self.raw_word_image = PIL.ImageTk.PhotoImage(im)
        self.im1= self.im2= self.im3= self.im4= self.im5 = self.raw_word_image

        self.wordPic = tk.Label(self)
        self.wordPic['image'] = self.raw_word_image
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



    def search(self):
        print("Searching...")

        # distances from this word to all other words
        dists_to_words = self.distances[self.current_index]

        # closest n words
        n=5
        closest_n_inds = dists_to_words.argsort()[:n]

        # display them somehow
        im_names = []
        for i in closest_n_inds:
            print("Nearest image: {}".format(self.images[i]))
            im_names.append(self.images[i])
        self.displayResultsWindow(im_names)



    def displayResultsWindow(self, results):
        t = tk.Toplevel(self)
        t.wm_title("Query Results")

        l1 = tk.Label(t)
        l1t = tk.Label(t,text="{}".format(results[0]))
        imp1 = r"{}/{}".format(WORD_IMAGES_PATH, results[0])
        self.im1 = PIL.ImageTk.PhotoImage(PIL.Image.open(imp1))
        l1['image'] = self.im1

        l2 = tk.Label(t)
        l2t = tk.Label(t,text="{}".format(results[1]))
        imp2 = r"{}/{}".format(WORD_IMAGES_PATH, results[1])
        self.im2 = PIL.ImageTk.PhotoImage(PIL.Image.open(imp2))
        l2['image'] = self.im2

        l3 = tk.Label(t)
        l3t = tk.Label(t,text="{}".format(results[2]))
        imp3 = r"{}/{}".format(WORD_IMAGES_PATH, results[2])
        self.im3 = PIL.ImageTk.PhotoImage(PIL.Image.open(imp3))
        l3['image'] = self.im3

        l4 = tk.Label(t)
        l4t = tk.Label(t,text="{}".format(results[3]))
        imp4 = r"{}/{}".format(WORD_IMAGES_PATH, results[3])
        self.im4 = PIL.ImageTk.PhotoImage(PIL.Image.open(imp4))
        l4['image'] = self.im4

        l5 = tk.Label(t)
        l5t = tk.Label(t,text="{}".format(results[4]))
        imp5 = r"{}/{}".format(WORD_IMAGES_PATH, results[4])
        self.im5 = PIL.ImageTk.PhotoImage(PIL.Image.open(imp5))
        l5['image'] = self.im5

        l1.pack(fill='both', expand=True, padx=100)
        l1t.pack(fill='both', expand=True, padx=100)
        l2.pack(fill='both', expand=True, padx=100)
        l2t.pack(fill='both', expand=True, padx=100)
        l3.pack(fill='both', expand=True, padx=100)
        l3t.pack(fill='both', expand=True, padx=100)
        l4.pack(fill='both', expand=True, padx=100)
        l4t.pack(fill='both', expand=True, padx=100)
        l5.pack(fill='both', expand=True, padx=100)
        l5t.pack(fill='both', expand=True, padx=100)


    
    def chooseImageForLabel(self):
        print("Choosing new image...")



app = Application()
app.master.title('Provide labels')
app.mainloop()
