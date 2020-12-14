from keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import Image, ImageOps
import numpy as np
import pyscreenshot as ImageGrab
import matplotlib.pyplot as plt
from joblib import dump, load


from collections import Counter

from sklearn.metrics import confusion_matrix

from sklearn import svm, datasets, metrics



model= load_model('numbers.h5')
model_linear =load('./SVM_models/linear.joblib')
model_radial =load('./SVM_models/radial.joblib')
model_poly =load('./SVM_models/poly.joblib')


class App(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=600, height=600, bg = "white", cursor="cross")
        
        self.label = tk.Label(self, text="Normal", font=("Helvetica", 24))
        self.label_linear = tk.Label(self, text="SVM_linear", font=("Helvetica", 24))
        self.label_radial = tk.Label(self, text="SVM_radial", font=("Helvetica", 24))
        self.label_poly = tk.Label(self, text="SVM_poly", font=("Helvetica", 24))
        #self.classify_btn = tk.Button(self, text = "Recognise", command = self.classify_handwriting)
 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        self.button_pred = tk.Button(self,text="Predict",command=self.pred_draw)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, padx=2, sticky=W, )

        self.label.grid(row=0, column=1,pady=2, padx=10)
        self.label_linear.grid(row=0, column=2,pady=2, padx=10)
        self.label_radial.grid(row=0, column=3,pady=2, padx=10)
        self.label_poly.grid(row=0, column=4,pady=2, padx=10)


        #self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.button_pred.grid(row=1, column=1, pady=2, padx=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)
    def clear_all(self):
        self.canvas.delete("all")

    def pred_draw(self):
        import pyscreenshot as ImageGrab
        x=self.winfo_rootx()+self.canvas.winfo_x()
        y=self.winfo_rooty()+self.canvas.winfo_y()
        x1=x+self.canvas.winfo_width()
        y1=y+self.canvas.winfo_height()
        #ImageGrab.grab().crop((x,y,x1,y1)).save("./image.png")
        ima=ImageGrab.grab().crop((x,y,x1,y1))
	
	#resize image to 28x28 pixels
        img = ima.resize((28,28),Image.LINEAR)
        #convert rgb to grayscale
        img = img.convert('L')
        img = np.array(img)

        img=-img+255

        #print(img.shape)
        #plt.imshow(img, cmap=plt.cm.binary)
        #plt.show()	


        #reshaping to support our model input and normalizing
        img = img.reshape(1,28,28)
        imgsv=img.reshape(1,784)
        img = img/255.0
        

        #predicting the class
        res = model.predict(img)

        digit=np.argmax(res)
        
        self.label.configure(text= "Normal:"+ str(digit))
	

        res = model_linear.predict(imgsv)
        
        self.label_linear.configure(text= "SVM_linear:"+ str(res[0]))
	

        res = model_radial.predict(imgsv)
        
        self.label_radial.configure(text= "SVM_radial:"+ str(res[0]))

        
        res = model_poly.predict(imgsv)
        
        self.label_poly.configure(text= "SVM_poly:"+ str(res[0]))


    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=35
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
app = App()
mainloop()
