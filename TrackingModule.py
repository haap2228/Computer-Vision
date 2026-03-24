import cv2
import time
import math
import mediapipe as mp

class HandDetector():
    def __init__(self,mode=False,handno=2,mcom=1,detcon=0.5,trcon=0.5):
        self.mode = mode
        self.handno = handno
        self.mcom = mcom
        self.detcon = detcon
        self.trcon = trcon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.handno,self.mcom,self.detcon,self.trcon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]

    def findhands(self,img,draw=True):
        self.result = self.hands.process(img)
        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)
        
        return img
    
    def handposi(self,img,hand=0,draw=True):

        self.lmList=[]
        xList = []
        yList = []
        bbox = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[hand] 
            for id,lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id,cx,cy])
            xMin,yMin = min(xList),min(yList)
            xMax,yMax = max(xList),max(yList)
            bbox = xMin,yMin,xMax,yMax

            if draw:
                cv2.rectangle(img,(xMin-20,yMin-20),(xMax + 20,yMax + 20),(0,255,0),2)

        return self.lmList,bbox
    
    def FingersUp(self):
        fingers = []

        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1] :
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def FindDistance(self,p1,p2,img,draw=True,r=15,t=3):
        x1,y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        cx,cy = (x1+x2)//2 , (y1+y2)//2

        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,0,255),t)
            cv2.circle(img,(x1,y1),r,(255,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),r,(255,0,255),cv2.FILLED)
            cv2.circle(img,(cx,cy),r,(255,0,255),cv2.FILLED)
        self.lenght = math.hypot(x1-x2,y1-y2)

        return self.lenght,img,[x1,y1,x2,y2,cx,cy]

def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector=HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findhands(img)
        lmList = detector.handposi(img)
        if len(lmList) != 0:
            print(lmList[4])
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),3)

        cv2.imshow("Image",img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()