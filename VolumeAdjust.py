import cv2
import time
import numpy as np
import TrackingModule as tm
import math
from pycaw.pycaw import AudioUtilities

cap = cv2.VideoCapture(0)

## Variables ##
smooth = 5
cTime = 0
pTime = 0
plen = 0
clen = 0

detector = tm.HandDetector(detcon=0.7)

## Get Volume Range ##
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
print(f"Audio output: {device.FriendlyName}")
print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")
maxVol = volume.GetVolumeRange()[0]
minVol = volume.GetVolumeRange()[1]

while True:
    success, img = cap.read()
    ## Get Hand LandMarks ##
    img = detector.findhands(img)
    lmList,_ = detector.handposi(img)
    if len(lmList) != 0: 
        x1,y1 = lmList[4][1],lmList[4][2]
        x2,y2 = lmList[8][1],lmList[8][2]
        cx,cy = (x1+x2)//2,(y1+y2)//2

        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        
        ## Interpret the distance b/w Index Fimger and Thumb in Volume Range ##
        lenght = math.hypot(x1-x2,y1-y2)
        vol = np.interp(lenght,[15,125],[maxVol,minVol])
        clen = plen + (vol - plen) / smooth

        ## Close middle Finger to Select Volume ##
        if lmList[12][2] < lmList[11][2]:
            vol1 = clen

        volume.SetMasterVolumeLevel(vol1, None)
        plen = clen

        if lenght<=15:
            cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
    
    ## FPS ##
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),3)
    
    ## Close Camera with 'Q' ##
    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()