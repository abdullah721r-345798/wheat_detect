import streamlit as st
import torch
import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time




def imageInput(device, src):
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            # call Model prediction--
           
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#                        model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/last.pt', force_reload=True)

            model.cuda() if device == 'cuda' else model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            # --Display predicton

            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')

    elif src == 'From test set.':
        # Image selector slider
        test_images = os.listdir('data/images/')
        test_image = st.selectbox('Please select a test image:', test_images)
        image_file = 'data/images/' + test_image
        submit = st.button("Predict!")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:
            if image_file is not None and submit:
                # call Model prediction--
                model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/last.pt', force_reload=True)
                pred = model(image_file)
                pred.render()  # render bbox in image
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))

                        
                    st.write(pred)
                    original_title = '<p style="font-family:Courier; color:; font-size: 14px;">DETAILS</p>'
                    st.markdown(original_title, unsafe_allow_html=True)

                    # --Display prediction
                    img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                    st.image(img_, caption='Model Prediction(s)')
                
            st.write(pred)
            original_title = '<p style="font-family:Courier; color:Black; font-size: 14px;">AIRSA\'s goal is to provide statistics on road safety using AI recognition. The AI model recognizes key factors of road safety (such as traffic light presence and stop sign presence) that are used in the safety formula. Other outputs such as road width, lane count, and individual lane width are factors of the safety formula.</p>'
            st.markdown(original_title, unsafe_allow_html=True)





def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ['From test set.', 'Upload your own data.'])

    

    # option = st.sidebar.radio("Select input type.", ['Image', 'Video'])
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], index=0)
    # -- End of Sidebar

    st.header('Customer Counter')
    st.subheader('👈🏽Select the options')

    imageInput(deviceoption, datasrc)

if __name__ == '__main__':
    main()


