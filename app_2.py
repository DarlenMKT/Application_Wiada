import cv2
from ultralytics import YOLO
import streamlit as st

model_path = "weights/yolov8n.pt"

st.set_page_config(
    page_title='Object detection using YOLOV8',     # Setting page title
    page_icon='🤩',      # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state='expanded'        # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    st.header('Video/webcam Config')        # Adding header to sidebar
    # Adding file uploader to sidebar for selecting videos
    uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4"])
    source_vid = st.sidebar.selectbox(
        "Or select Webcam", ["Webcam"])
    
    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("Object Detection using YOLOv8")

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
    
    # Mise en place du model
    model = YOLO(model_path)

if uploaded_file is not None:
    source_vid = uploaded_file.name

elif source_vid == "Webcam":
    source_vid = 0      # Use 0 to indicate the default webcam

if source_vid is not None: 
    vid_cap = cv2.VideoCapture(source_vid)
    # Si detectObject n'est pas lancer la fonction ne s'execute pas 
    if st.sidebar.button('Detect Objects'):
        st_frame = st.empty()
        while vid_cap.isOpened():
            success, image = vid_cap.read()
            if success:
                image = cv2.resize(image, (720, int(720*(9/16))))        # 720*9/16 ,16 la largeur de notre page et 9 la hauteur 
                # L'image aura une forme rectangulaire et non carrée (720, 720)

                # Une prédiction de l'image obtenue
                res = model.predict(image, conf=confidence)

                result_tensor = res[0].boxes

                res_plotted = res[0].plot()
                
                st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)
                    
            else:
                vid_cap.release()
                break