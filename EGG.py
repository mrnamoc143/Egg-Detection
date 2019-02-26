from tkinter import *
from PIL import Image, ImageTk
import cv2


windows = Tk()
windows.resizable(0,0)
windows.geometry("1500x900")

photo = Image.open('current/current.png')
resized = photo.resize((350,300),Image.ANTIALIAS)
images = ImageTk.PhotoImage(resized)
currentImg = Label(windows, image=images)
currentImg.pack()
currentImg.place(x=1000,y=50,width=350,height=300)
windows.mainloop()
