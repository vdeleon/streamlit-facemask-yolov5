from io import StringIO
from pathlib import Path
import streamlit as st
import time
import detect
import os
import sys
import argparse
from PIL import Image
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode,AudioProcessorBase, RTCConfiguration, VideoProcessorBase
import cv2
import torch
import numpy as np
import pandas as pd
from typing import List
import matplotlib.colors as mcolors
from aiortc.contrib.media import MediaPlayer
import pafy


from config import CLASSES, WEBRTC_CLIENT_SETTINGS

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
# --------------------------------------------

@st.cache(max_entries=2)
def get_yolo5(model_type='s'):
    '''

Devuelve un modelo YOLOv5 de Torch Hub de tipo `model_type`

    Argumentos
    ----------
    modelo_tipo: str, 's', 'm', 'l' o 'x'
        tipo de modelo - s - más rápido e inexacto, x - más preciso y más lento

    devoluciones
    -------
    modelo de antorcha
        modelo de antorcha de tipo `<class 'models.common.autoShape'>`
    '''
    return torch.hub.load('ultralytics/yolov5', 'custom' , path='bestM.pt')

@st.cache(max_entries=10)
def get_preds(img : np.ndarray) -> np.ndarray:
    """
Returns predictions received from YOLOv5

     Arguments
     ---------
     img : np.ndarray
         RGB image loaded with OpenCV

     returns
     -------
     2d np.ndarray
         List of found objects in the format
         `[xmin,ymin,xmax,ymax,conf,label]`
    """
    return model([img]).xyxy[0].numpy()

def get_colors(indexes : List[int]) -> dict:
    '''
Returns the colors for all selected classes. Colors are formed
     based on the TABLEAU_COLORS and BASE_COLORS sets from Matplotlib

     Arguments
     ----------
     indexes : list of int
         list of class indexes in default order for
         MS COCO (80 classes, no background)

     returns
     -------
     dict
         a dictionary in which the keys are the id classes specified in
         indexes and values are tuple with rgb color components, for example (0,0,0)
    '''
    to_255 = lambda c: int(c*255)
    tab_colors = list(mcolors.TABLEAU_COLORS.values())
    tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) 
                                                for name_color in tab_colors]
    base_colors = list(mcolors.BASE_COLORS.values())
    base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
    rgb_colors = tab_colors + base_colors
    rgb_colors = rgb_colors*5
    rgb_colors[0] = (255,0,0)
    rgb_colors[1] = (0,255,0)



    color_dict = {}
    for i, index in enumerate(indexes):
        if i < len(rgb_colors):
            color_dict[index] = rgb_colors[i]
        else:
            color_dict[index] = (0,0,255)

    return color_dict

def get_legend_color(class_name : int):
    """
Returns the cell color for `pandas.Styler` when creating the legend.
     Colorize the cell with the same color as the boxes of the corresponding class

     Arguments
     ---------
     class_name : int
         class name according to MS COCO class list

     returns
     -------
     str
         background-color for the cell containing the class_name
    """  

    index = CLASSES.index(class_name)
    print(index)
    color = rgb_colors[index]
    return 'background-color: rgb({color[0]},{color[1]},{color[2]})'.format(color=color)



class VideoTransformer(VideoTransformerBase):
    '''Componente para crear una transmisión de cámara web'''
    def __init__(self):
        self.model = model
        self.rgb_colors = rgb_colors
        self.target_class_ids = target_class_ids

    def get_preds(self, img : np.ndarray) -> np.ndarray:
        return self.model([img]).xyxy[0].numpy()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.get_preds(img)
        result = result[np.isin(result[:,-1], self.target_class_ids)]
        
        for bbox_data in result:
            xmin, ymin, xmax, ymax, _, label = bbox_data
            p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
            img = cv2.rectangle(img, p0, p1, self.rgb_colors[label], 2) 

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


classes_selector = 'All classes'
all_labels_chbox = 'All classes'

if all_labels_chbox:
    target_class_ids = list(range(len(CLASSES)))
elif classes_selector:
    target_class_ids = [CLASSES.index(class_name) for class_name in CLASSES]
else:
    target_class_ids = [0]

rgb_colors = get_colors(target_class_ids)
detected_ids = None


if __name__ == '__main__':
    model = get_yolo5('s')
    #st.success('Loading the model.. Done!')

    st.title('Facemask Detection CNN FMAT Streamlit App')
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'bestM.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'test1.png', help='file/dir/URL/glob, 0 for webcam')
    #parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    print(opt)

    source = ("Detección de imagen", "Detección de vídeo", "Detección por WebCam", "Detección por URL")
    source_index = st.sidebar.selectbox("seleccionar entrada", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "cargar imagen", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='carga de recursos...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                opt.source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False

    elif source_index == 1:
        uploaded_file = st.sidebar.file_uploader("subir video", type=['mp4', 'mov','avi'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='carga de recursos...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False

    elif source_index == 2:
        is_valid = False
        ctx = webrtc_streamer(key="sample", video_transformer_factory=VideoTransformer, client_settings=WEBRTC_CLIENT_SETTINGS,)
        
        if ctx.video_transformer:
            ctx.video_transformer.model = model
            ctx.video_transformer.rgb_colors = rgb_colors
            ctx.video_transformer.target_class_ids = target_class_ids

        '''
        uploaded_file = webrtc_streamer(key="sample")
        if uploaded_file is not None:
            is_valid = True

        else:
            is_valid = False
        '''
    else:
        user_input = st.text_input("URL:")
        url = "https://www.youtube.com/watch?v=bnzPJhW9XQg"
        video = pafy.new(url)
        best = video.getbest()


        if "youtube" in user_input:
            video = pafy.new(user_input)
            best = video.getbest()
            MEDIAFILES = {
            "web": {
            #"url": "rtmp://wcs5-eu.flashphoner.com:1935/live",
            "url": best.url,
            "type": "video",
            },
            }
        else:
             MEDIAFILES = {
            "web": {
            "url": user_input,
            #"url": best.url,
            "type": "video",
            },
            }

        is_valid = False


        media_file_label = tuple(MEDIAFILES.keys())
        media_file_info = MEDIAFILES["web"]
        print(media_file_info["url"])

        def create_player():
            return MediaPlayer(media_file_info["url"])


        webrtc_ctx = webrtc_streamer(key=f"media-streaming-{media_file_label}", mode=WebRtcMode.RECVONLY,  rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
            "video": media_file_info["type"] == "video",
            "audio": media_file_info["type"] == "audio",
            },
            player_factory=create_player,
            video_processor_factory=VideoTransformer,
            )

        if webrtc_ctx.video_transformer:
            webrtc_ctx.video_transformer.model = model
            webrtc_ctx.video_transformer.rgb_colors = rgb_colors
            webrtc_ctx.video_transformer.target_class_ids = target_class_ids






        #uploaded_file = webrtc_streamer(key="sample")
        #title = st.text_input('Ingrese la url para la detección', '')
        #is_valid = True
        #with st.spinner(text="carga de recursos..."):
        #    st.sidebar.video(title)
        #    print(title)
        #    opt.source = f'{title}'

        #st.write('Detección usando la url', title)
        '''
        if uploaded_file is not None:
            is_valid = True
           

        else:
            is_valid = False
        '''
        
    if is_valid:
        print('valid')
        if st.button('Comenzar detección'):
            detect.main(opt)

            if source_index == 0:
                with st.spinner(text='Preparando Imagen'):
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))

                    st.balloons()

            elif source_index == 1:
                with st.spinner(text='Preparando Video'):
                    for vid in os.listdir(get_detection_folder()):
                        print(vid)
                        st.video(str(Path(f'{get_detection_folder()}') / vid))

                    st.balloons()

            elif source_index == 3:
                print("se ha presionado el boton presionar para la deteccion")
                with st.spinner(text='Preparando de la URL destino'):
                    print("entrando al boton de presionar")
                    for url in os.listdir(get_detection_folder()):
                        st.video(str(Path(f'{get_detection_folder()}') / url))

                    st.balloons()
