import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv
from google.colab.patches import cv2_imshow
import mtcnn
from facenet_pytorch import MTCNN

def ur_mom():
    print("ur_mom")

class PPGVideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, max=-1):
        self.ground_truths = np.array([])
        self.root_dir = root_dir
        self.set_ground(max)

    def set_ground(self, max):
      # os.chdir(self.root_dir)
      size = len(os.listdir(self.root_dir))
      # print(size)
      path = self.root_dir + "/subject"
      ret_vals = []

      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      print(f"using device: {device}")
      detector = MTCNN(device=device)

      for i in range(size+1):
        current_path = path +str(i)
        # print(current_path)
        if not os.path.exists(current_path):
          continue

        temp_path = current_path + '/' + str(i)
        # print(temp_path)
        current_size = len(os.listdir(current_path))
        for j in range(1,current_size):
          final_path = temp_path+'_'+str(j)

          if not os.path.exists(final_path):
            continue
          # print(final_path)
          # ar = np.loadtxt(final_path, delimiter=',')
          # element = {'ppg': ar, 'path': final_path}
          
          cap = cv.VideoCapture(final_path+'/vid.avi')
          ret, frame = cap.read()

          # detector = mtcnn.MTCNN()
          # print("detect start")
          frame = cv.resize(frame, (int(frame.shape[1] * .2), int(frame.shape[0] * .2)))
          
          face = detector.detect(frame)
          # print("detect end")

          # lx, ly = face[0]['keypoints']["left_eye"]
          # rx, ry = face[0]['keypoints']["right_eye"]
          # x = int((lx+rx)/2)
          # y = int((ry+ly)/2)

          # frame[int(cropped[0][0][1]):int(cropped[0][0][3]), int(cropped[0][0][0]):int(cropped[0][0][2]), :: ]

          x = int((cropped[0][0][0]+cropped[0][0][2])*5/2)
          y = int((cropped[0][0][1]+cropped[0][0][2])*5/2)
          
          element = {"path": final_path, 'center': {'x':x, 'y':y}}
          print(element)
          ret_vals.append(element)

          # If we only want to load some data
          max -= 1
          if(max == 0):
            self.ground_truths = np.array(ret_vals)
            return

      self.ground_truths = np.array(ret_vals)

        

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        # get truths dict
        # use path to get video
        # crop video
        # add video to dcitionary, remove path
        # return
        
        path = self.ground_truths[i]['path']
        x = self.ground_truths[i]['center']['x']
        y = self.ground_truths[i]['center']['y']
        item = {'label': np.loadtxt(path+'/ppg.csv', delimiter=',')}
        
        print("1")
        cap = cv.VideoCapture(path+'/vid.avi')
        print("2")
        t = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        print("3")
        video = np.zeros((t, 256, 256, 3))
        
        print("4")
        
        for i in range(t):
          ret, frame = cap.read()
          video[i] = frame[y-128:y+128,x-128:x+128,::]
          print(f"4.{i}")
        print("5")
        item["data"] = video

        return item