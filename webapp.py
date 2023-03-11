"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
from PIL import Image
import datetime

import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
from camera import VideoCamera

app = Flask(__name__)
imgpath = ""
names =  [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush' ]
num_classes = len(names)


@app.route("/")
def hello_world():
    return render_template('index.html')


# function for accessing rtsp stream, To be removed
@app.route("/rtsp_feed")
def rtsp_feed():
    cap = cv2.VideoCapture(0)#'rtsp://admin:hello123@192.168.29.126:554/cam/realmonitor?channel=1&subtype=0'
    return render_template('index.html')


# Function to start webcam and detect objects, to be modified

# call getFrame function from camera package


def gen(camera):
    while True:
        frame = camera.getFrame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route("/webcam_feed")
def webcam_feed():
    return Response(gen(VideoCamera()), mimetype="multipart/x-mixed-replace;boundary=frame")
    # webcam code
    # model_dict = torch.load('yolov7.pt', map_location=torch.device('cpu'))
    # model = model_dict['model'].float()


#obj detection program
def detect_objects(cap):
    # model = torch.hub.load("./part1/yolov7", 'custom',path='yolov7.pt', local=True) # does not work
    model_dict = torch.load('yolov7.pt', map_location=torch.device('cpu'), force_reload=True)
    model = model_dict['model'].float()
    # cap = cv2.VideoCapture(0)
    while (True):
        
        ret, frame = cap.read() 
        # tensor = torch.from_numpy(frame).to(torch.float32) 
        if not ret:
            break

        results = model(frame)
        results.render()
        ret, jpeg = cv2.imencode('.jpg', frame) # convert to byte code
                # Yield the byte array as an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.1)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    


# function to get the frames from video (output video)
def get_frame():
    global imgpath
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    filename = imgpath 
    image_path = folder_path+'/'+latest_subfolder+'/'+filename    
    video = cv2.VideoCapture(image_path)  # detected video path
    # video = cv2.VideoCapture("C:\\Users\\pvroh\\OneDrive\\Desktop\\Deep Learning\\Review 2\\flash\\Object-Detection-Web-App-Using-YOLOv7-and-Flask\\uploads\\Site - 40132.mp4")
    detect_objects(video)


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



#The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ",directory)  
    filename = imgpath
    file_extension = filename.rsplit('.', 1)[1].lower()
    print("printing file extension from display function : ",file_extension)
    environ = request.environ
    if file_extension in ['jpg','png','jpeg']:      
        return send_from_directory(directory,filename,environ)

    elif file_extension in ['mp4', 'avi']:
        return render_template('index.html')

    else:
        return "Invalid file format"

    
@app.route("/", methods=["GET", "POST"])
def predict_img():
    
    global imgpath
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            
            imgpath = f.filename
            print("printing predict_img :::::: CHECK THIS:  ",f.filename, predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()    
            if file_extension in ['jpg', 'jpeg', 'png']:
                process = Popen(["python", "detect.py", '--source', filepath, "--weights","yolov5s.pt"], shell=True)
                process.wait()
                
                
            elif file_extension in ['mp4', 'avi']:
                process = Popen(["python", "detect.py", '--source', filepath, "--weights","yolov5s.pt"], shell=True)
                process.communicate()
                process.wait()

            
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    image_path = folder_path+'/'+latest_subfolder+'/'+f.filename 
    return render_template('index.html', image_path=image_path)
    #return "done"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = torch.hub.load('.', 'custom','yolov5s.pt', source='local')
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat


