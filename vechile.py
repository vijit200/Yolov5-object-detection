import cv2
import numpy as np
import simpleaudio as sa

cap = cv2.VideoCapture('video.mp4')

min_width_rectangle = 80
min_height_rectangle = 80
count_line_position = 550
offset = 6 #alloable error between pixel
counter = 0
#Initialize substracter
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1 = int(w)
    y1 = int(h)
    #cx = x
    cx = x+x1
    cy = y+y1
    return cx,cy

def center_2(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    #cx = x
    cx1 = x+x1
    cy1 = y+y1
    return cx1,cy1

detect = []
detect2 = []



while True:
    ret,frame = cap.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    #applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    counterShape,h = cv2.findContours(dilatada,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame,(25,count_line_position),(1200,count_line_position),(255,127,0),3)
    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        val_countor = (w>=min_width_rectangle) and (h>=min_height_rectangle)
        if not val_countor:
            continue
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(frame,"Vechile counter : "+ str(counter),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0),3)
        center = center_handle(x,y,w,h)
        center2 = center_2(x,y,w,h)
        detect.append(center)
        detect2.append(center2)
        cv2.circle(frame,center,4,(0,0,255),-1)
        cv2.circle(frame,center2,4,(0,0,255),-1)

        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter +=1
                filename = 'alert.wav'
                wave_obj = sa.WaveObject.from_wave_file(filename)
                play_obj = wave_obj.play()
                play_obj.wait_done() 

            cv2.line(frame,(25,count_line_position),(1200,count_line_position),(0,255,255),3)
            detect.remove((x,y))
        for (x,y) in detect2:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                #counter +=1
                filename = 'alert.wav'
                wave_obj = sa.WaveObject.from_wave_file(filename)
                play_obj = wave_obj.play()
                play_obj.wait_done() 

            cv2.line(frame,(25,count_line_position),(1200,count_line_position),(0,255,255),3)
            detect2.remove((x,y))
            print("car counter : " + str(counter))
    cv2.putText(frame,"Vechile counter : "+ str(counter),(470,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
    #cv2.imshow('detect',dilatada)
    cv2.imshow('Video',frame)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()