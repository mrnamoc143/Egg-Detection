from PIL import Image, ImageTk
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
import tensorflow as tf
import sys
import serial
import shutil
import matplotlib.pyplot as plt
import os
import threading

ser1 = serial.Serial('COM3',9600)


female = 0
male = 0
previous = "None"
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
        

MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

        

# Number of classes the object detector can identify
NUM_CLASSES = 2



# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `1`, we know that this corresponds to `male`.
# network predicts `2` we know that this corresponds to `female`
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
     od_graph_def = tf.GraphDef()
     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')

     sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
 
class App:
     def __init__(self, window, window_title, video_source=0):
         self.window = window
         self.window.title(window_title)
         self.video_source = video_source
         self.window.resizable(0,0)
         self.window.geometry("1500x900")
         self.window.configure(background='white')

         #terminal
        
 
         # para ma open ang webcam
         self.vid = MyVideoCapture(self.video_source)
 
         # para sa live video
         self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
         self.canvas.pack()
         self.canvas.place(x=40,y=50)

         # current photo
         self.photo = Image.open('current/current.jpg')
         self.resized = self.photo.resize((400,300),Image.ANTIALIAS)
         self.images = ImageTk.PhotoImage(self.resized)
         self.currentImg = tkinter.Label(image=self.images)
         self.currentImg.pack()
         self.currentImg.place(x=900,y=400,width=400,height=300)
         

         # Button that lets the user take a snapshot
         self.btn_snapshot=tkinter.Button(window, text="Start", command=self.snapshot,bg='white')
         self.btn_snapshot.config(font=("Cambria", 50))
         self.btn_snapshot.pack()
         self.btn_snapshot.place(x=85,y=550,width=550,height=100)

         # move
         self.btn_snapshot=tkinter.Button(window, text=".5", command=self.move,bg='white')
         self.btn_snapshot.config(font=("Cambria", 50))
         self.btn_snapshot.pack()
         self.btn_snapshot.place(x=85,y=665,width=100,height=100)

         # small move
         self.btn_snapshot=tkinter.Button(window, text=".25", command=self.small,bg='white')
         self.btn_snapshot.config(font=("Cambria", 50))
         self.btn_snapshot.pack()
         self.btn_snapshot.place(x=85,y=790,width=100,height=100)
         

         
        
         #Label sa GUI
         
         self.previousLbl = tkinter.Label(window,text="Previous: "+previous ,bg='white')
         self.previousLbl.pack()
         self.previousLbl.place(x=900,y=775,width=450,height=100)
         self.previousLbl.config(font=("Cambria",44))

         #female
         nameF= str(female)
         self.femaleLbl = tkinter.Label(window,text="Female: "+nameF,bg='white')
         self.femaleLbl.pack()
         self.femaleLbl.place(x=240,y=675,width=400,height=100)
         self.femaleLbl.config(font=("Cambria",44))

         #male
         nameM= str(male)
         self.maleLbl = tkinter.Label(window,text="Male: "+nameM,bg='white')
         self.maleLbl.pack()
         self.maleLbl.place(x=270,y=775,width=400,height=100)
         self.maleLbl.config(font=("Cambria",44))
         #terminal
         self.terminal = tkinter.Text(window,height=10,width=40,bg='black',fg='white')
         self.terminal.pack()
         self.terminal.place(x=750,y=60)
         self.terminal.config(font=("Cambria",20))
         
     
 
         # After it is called once, the update method will be automatically called every delay milliseconds
         
         self.delay = 15
         self.update()
 
         self.window.mainloop()
 
     def snapshot(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()
 
         if ret:
             
             cv2.imwrite("current/current.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
             cv2.imwrite("current/move.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
             self.photo = Image.open('current/current.jpg')
             self.resized = self.photo.resize((400,300),Image.ANTIALIAS)
             self.images = ImageTk.PhotoImage(self.resized)
             self.currentImg = tkinter.Label(image=self.images)
             self.currentImg.pack()
             self.currentImg.place(x=900,y=400,width=400,height=300)
             #cv2.imwrite("current/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
             self.predict()
     def move(self):
          ser1.write('m'.encode())
     def small(self):
          ser1.write('s'.encode())
             
     def update(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()
 
         if ret:
             self.femaleLbl.pack_forget()
             self.maleLbl.pack_forget()
             self.previousLbl.pack_forget()
             
             nameM= str(male)
             self.maleLbl = tkinter.Label(text="Male: "+nameM,bg='white')
             self.maleLbl.pack()
             self.maleLbl.place(x=270,y=775,width=400,height=100)
             self.maleLbl.config(font=("Cambria",44))

             nameF= str(female)
             self.femaleLbl = tkinter.Label(text="Female: "+nameF,bg='white')
             self.femaleLbl.pack()
             self.femaleLbl.place(x=240,y=675,width=400,height=100)
             self.femaleLbl.config(font=("Cambria",44))
             
             self.previousLbl = tkinter.Label(text="Previous: "+previous ,bg='white')
             self.previousLbl.pack()
             self.previousLbl.place(x=900,y=775,width=450,height=100)
             self.previousLbl.config(font=("Cambria",44))

             
             self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
             self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
         self.window.after(self.delay, self.update)
     def predict(self):
         
        # Name of the directory containing the object detection module we're using
        
        #IMAGE_NAME = 'current/move.jpg'
        IMAGE_NAME = 'current/current.jpg'

        # Path to image
        PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

        

        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image = cv2.imread(PATH_TO_IMAGE)

        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # Draw the results of the detection (aka 'visulaize the results')


        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.80)

        def GetClassName(data):
           for cl in data:
            return cl['name']

        #data processed


        data = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.8]

        print(GetClassName(data))

        timestr = time.strftime("%Y%m%d-%H%M%S")
        global male
        global female
        global previous
        time.sleep(5)

        if (GetClassName(data) == "Female"):
            ser1.write('f'.encode())
            name = timestr + "-female"
            self.terminal.insert('1.0', '>>The system detect: Female\n')
            shutil.move("current/move.jpg","history/%s.jpg"% name)
            #shutil.move("current/current2.jpg","history/current2.jpg")
            female = female + 1
            previous = "Female"
            print("Go")
            self.terminal.insert('1.0', '>>Open the egg separator\n')
            #self.wait()
        elif (GetClassName(data) =="Male"):
            ser1.write('m'.encode())
            name = timestr + "-Male"
            self.terminal.insert('1.0', '>>The system detect: Male\n')
            shutil.move("current/move.jpg","history/%s.jpg"% name)
            #shutil.move("current/current2.jpg","history/current2.jpg")
            male = male + 1
            previous = "Male"
            print("Stop")
            self.terminal.insert('1.0', '>>Close the egg separator\n')
            #self.wait()
        else:
            print("Error")
            self.terminal.insert('1.0', '>>Error No Egg Found\n')
            ser1.write('e'.encode())
            
            

             
 
 
class MyVideoCapture:
     def __init__(self, video_source=0):
         # Open the video source
         self.vid = cv2.VideoCapture(video_source)
         if not self.vid.isOpened():
             raise ValueError("Unable to open video source", video_source)
 
         # Get video source width and height
         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
     def get_frame(self):
         if self.vid.isOpened():
             ret, frame = self.vid.read()
             if ret:
                 # Return a boolean success flag and the current frame converted to BGR
                 return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
             else:
                 return (ret, None)
         else:
             return (ret, None)
 
     # Release the video source when the object is destroyed
     def __del__(self):
         if self.vid.isOpened():
             self.vid.release()
 
# Create a window and pass it to the Application object
App(tkinter.Tk(), "Egg Predictor")
