import cv2
import numpy as np
import TrackingModule as tm
import time
import autopy

## Variables ##
cTime = 0
pTime = 0
wCam,hCam = 1280,720
frameR = 200
smooth = 5
plocX,plocY = 0,0
clocX,clocY = 0,0

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
detector = tm.HandDetector()
wScr,hScr = autopy.screen.size()


while True:
    success, img = cap.read()

    ## Get Hand LandMarks ##
    img = detector.findhands(img)
    lmList,bbox = detector.handposi(img)
    cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(0,255,0),2)
    
    if len(lmList) != 0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        fingers = detector.FingersUp()

        ## If Index Finger up : Mouse Moving ##
        if fingers[1]==1 and fingers[2]==0:
            x3 = np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3 = np.interp(y1,(frameR,hCam-frameR),(0,hScr))
            
            ## Smoothen Value ##
            clocX = plocX + (x3 - plocX) / smooth
            clocY = plocY + (y3 - plocY) / smooth

            autopy.mouse.move(wScr - clocX,clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocX,plocY = clocX,clocY

        ## If Index and Middle Finger up, Short Distance : Mouse Clicking ##
        if fingers[1]==1 and fingers[2]==1:
            dist,img,Lineinfo = detector.FindDistance(8,12,img)
            if dist < 45:
                autopy.mouse.click()
                cv2.circle(img,(Lineinfo[4],Lineinfo[5]),15,(0,255,0),cv2.FILLED)
    
    ## FPS ##
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),3)

    ## Close Camera with 'Q' ##
    cv2.imshow("Img",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
