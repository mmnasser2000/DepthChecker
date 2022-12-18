import cv2
import mediapipe as mp
import time
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

class DepthChecker:

    '''
    Initializes object 
    Input: URL as a String
    Output: None
    '''
    def __init__(self, url=None):
        self.setURL(url)
    

    '''
    Checks depth on one side of the body
    Input: List of hip positions as floats, List of knee positions as floats
    Output: True if depth is hit on that side of the body, False otherwise
    '''
    def checkDepthOnSide(self, hip, knee):
        for hipPosIdx in argrelextrema(np.array(hip), np.less)[0]:
            if hip[hipPosIdx] < knee[hipPosIdx]:
                return True
        return False

    '''
    Sets/Updates the URL of the object
    Input: URL as a String
    Output: True on Success, False Otherwise
    '''
    def setURL(self, url):
        try:
            self.url = url
        except:
            return False
        return True
    
    '''
    Checks if the URL of the object has been set
    Input: None
    Output: True if URL has been set, False Otherwise
    '''
    def canCheckDepth(self):
        if self.url is None:
            return False
        return True
    '''
    Checks if the person in the video has hit depth
    Input: None
    Output: True if depth has been hit, False Otherwise
    '''
    def checkDepth(self):
        if not self.canCheckDepth():
            raise Exception("URL has not been initialized")

        mpPose = mp.solutions.pose
        pose = mpPose.Pose()
        mpDraw = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(self.url)
        sTime = time.time()
        pTime = 0

        rightHip = []
        leftHip = []
        rightKnee = []
        leftKnee = []

        positions = [rightHip, leftHip, rightKnee, leftKnee]

        while True:
            success, img = cap.read()
            if not success:
                break
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            
            if results.pose_landmarks:
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    '''
                    23: right hip
                    24: left hip
                    25: right knee
                    26: left knee
                    '''
                    if 23 <= id <= 26:
                        positions[id - 23].append(lm.y)
            
        return self.checkDepthOnSide(rightHip, rightKnee) or self.checkDepthOnSide(leftHip, leftKnee)

url = "./data/squatOneRep.mov"
s = DepthChecker(url)
print(s.checkDepth())