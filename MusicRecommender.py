import numpy as np
import cv2
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
# we have imported to_categorical to make our one hot vectors
from keras.layers import Dense,Input
from keras.models import Model
import os
from keras.models import load_model
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import webbrowser




resultingLabels = np.load("labels.npy")
model = load_model("model.h5")


holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils


st.header("MUSIC RECOMMENDATION BASED ON YOUR EMOTION")


if "running" not in st.session_state:
	st.session_state["running"] = "true"

try:
	emotion = np.load("OurEmotion.npy")[0]
except:
	emotion = ""

if not(emotion):
	st.session_state["running"] = "true"
else:
	st.session_state["running"] = "false"


cap = cv2.VideoCapture(0)

class EmotionProcessing:
	def recv(self,frame):
		frm = frame.to_ndarray(format="bgr24")
		listofdata = []

		# _,frm = cap.read()

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


			listofdata = np.array(listofdata).reshape(1,-1)

			outputPred = resultingLabels[np.argmax(model.predict(listofdata))]

			print(outputPred)
			cv2.putText(frm,outputPred,(50,50),cv2.FONT_ITALIC,1,(255,0,0),2)
			np.save("OurEmotion.npy",np.array([outputPred]))

		drawing.draw_landmarks(frm,result.face_landmarks,holistic.FACEMESH_CONTOURS)
		drawing.draw_landmarks(frm,result.right_hand_landmarks,hands.HAND_CONNECTIONS)
		drawing.draw_landmarks(frm,result.left_hand_landmarks,hands.HAND_CONNECTIONS)

		return av.VideoFrame.from_ndarray(frm,format="bgr24")

language = st.text_input("Enter the language you prefer for your Songs")

if language and st.session_state["running"] != "false":
	webrtc_streamer(key="key",desired_playing_state=True,video_processor_factory = EmotionProcessing)


bttn = st.button("GIVE SOME RECOMMENDATION")


if bttn:

	if not(emotion):
		st.warning("LET ME CAPTURE THE EMOTION AGAIN")
		st.session_state["running"] = "true"
	else:
		# if(OurEmotion == )
		webbrowser.open("https://open.spotify.com/search/{language}%20{emotion}%20songs")
		np.save("OurEmotion.npy",np.array([""]))
		st.session_state["running"] = "false"

		