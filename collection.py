import numpy as np
import cv2
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt

# we do this to capture the video through our own inbuilt camera

cap = cv2.VideoCapture(0)

inp = input("Enter the emotion we are going to recognize")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils


data = [] 

data_size = 0


while True:
	listofdata = []

	_,frm = cap.read()

	frm = cv2.flip(frm,1)

	result = holis.process(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB))


	if result.face_landmarks:
		for i in result.face_landmarks.landmark:
			listofdata.append(i.x - result.face_landmarks.landmark[1].x)
			listofdata.append(i.y - result.face_landmarks.landmark[1].y)



		if result.right_hand_landmarks:
			for i in result.right_hand_landmarks.landmark:
				listofdata.append(i.x - result.right_hand_landmarks.landmark[8].x)
				listofdata.append(i.y - result.right_hand_landmarks.landmark[8].y)


		else:
			for i in range(42):
				listofdata.append(0.0)


		if result.left_hand_landmarks:
			for i in result.left_hand_landmarks.landmark:
				listofdata.append(i.x - result.left_hand_landmarks.landmark[8].x)
				listofdata.append(i.y - result.left_hand_landmarks.landmark[8].y)

		else:
			for i in range(42):
				listofdata.append(0.0)	


		data.append(listofdata)
		data_size = data_size+1	



	drawing.draw_landmarks(frm,result.face_landmarks,holistic.FACEMESH_CONTOURS)
	drawing.draw_landmarks(frm,result.right_hand_landmarks,hands.HAND_CONNECTIONS)
	drawing.draw_landmarks(frm,result.left_hand_landmarks,hands.HAND_CONNECTIONS)

	cv2.putText(frm,str(data_size),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

	cv2.imshow("window",frm)


	if cv2.waitKey(1) == 27 or data_size > 99:
		cv2.destroyAllWindows()
		cap.release()
		break


np.save(f"{inp}.npy",np.array(data))
print(np.array(data).shape)




