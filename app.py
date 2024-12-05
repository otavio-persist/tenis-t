import cv2
import mediapipe as mp
import numpy as np
import time
import datetime
from supabase import create_client, Client
import streamlit as st
import os

os.environ["PYTHON_EGG_CACHE"] = "/tmp/.python-eggs"

# Configuração do Supabase
url = "https://ckixycdahzshcdkiqxxs.supabase.co"  # Substitua pela URL do seu projeto Supabase
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNraXh5Y2RhaHpzaGNka2lxeHhzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzMyNTQ4OTQsImV4cCI6MjA0ODgzMDg5NH0.AuMuwRSpVKytamrF2D2Ea961dVd6N6RTMxf-LQicLxw"  # Substitua pela sua chave de API
supabase: Client = create_client(url, key)

# Inicializa o MediaPipe Pose
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def store_data_in_supabase(timestamp, angle, speed, path):
    data = {
        "timestamp": timestamp,
        "angulo": angle,
        "velocidade": speed,
        "trajeto": path
    }
    response = supabase.table("movimento_braco").insert(data).execute()
    if response.data:
        st.success("Dados enviados com sucesso!")
    else:
        st.error("Erro ao enviar os dados.")

def main():
    st.title("Pose Tracking com Supabase")
    st.write("Capture e analise movimentos diretamente no navegador.")

    run = st.checkbox("Iniciar Captura")

    if run:
        cap = cv2.VideoCapture(0)
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    h, w, _ = frame.shape

                    # Posição do braço direito
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w,
                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h]

                    # Cálculo do ângulo
                    angle = calculate_angle(shoulder, elbow, wrist)

                    # Exibição no Streamlit
                    st.image(frame, channels="BGR", caption=f"Angulo: {int(angle)} graus")

        cap.release()

if __name__ == "__main__":
    main()
