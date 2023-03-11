import cv2
import torch
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

class VideoCamera(object):
    def __init__(self):
        global model
        self.cap = cv2.VideoCapture(0) 

    def __del__(self):
        self.cap.release() 


    def getFrame(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed Camera!")
                break

            results = model(frame)
            list_temp = pd.DataFrame(results.pandas().xyxy[0])

            # To find Bounding Box
            if(list_temp.empty):
                print("No Elements Detected!")
            else:
                for i, row in list_temp.iterrows():
                    xmin = int(row['xmin'])
                    ymin = int(row['ymin'])
                    xmax = int(row['xmax'])
                    ymax = int(row['ymax'])
                    label = row["name"]
                    conf = row['confidence']
                    conf = row['confidence']
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            #results.pandas().xyxy[0]
            # print(results.pandas().xyxy[0])

            # results.print()

            # for index, result in results.pandas().xyxy[0].iterrows():
        

        
            # for result in results.pandas().xyxy[0]:
            #     xmin, ymin, xmax, ymax = result[:4].astype(int)
            #     label = result[-1]
            #     conf = result[-2]
            #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            #     cv2.putText(frame, f'{label} {conf:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            ret1, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()



#model code 
'''

        

        model_dict = torch.load('yolov7.pt', map_location=torch.device('cpu'))
        model = model_dict['model'].float()

            #tensorflow frame to tensor format conversion
            tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            tensor = tensor.transpose((2, 0, 1))  # Transpose from (height, width, channels) to (channels, height, width)
            tensor = np.expand_dims(tensor, axis=0)  # Add batch dimension
            tensor = torch.from_numpy(tensor).to(torch.float32)
            print("Shape: ", tensor.shape)


            results = model(tensor)
            # drawing Box
            for result in results.xyxy[0]:
                label = result[-1]
                conf = result[-2]
                box = result[:4].astype(int)
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # results = model(frame)
            results.render()



'''