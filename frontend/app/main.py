import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from io import StringIO
from PIL import Image
import PIL.Image as Image
import io
import requests
import json
import os, random
from os import listdir
from os.path import isfile, join


# Get all example files paths
imgs_paths = []
imgs_root_path = "example_images/"
for (dirpath, dirnames, filenames) in os.walk(imgs_root_path):
    for dir in dirnames:
        dir_path = join(imgs_root_path, dir)
        imgs = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
        imgs_paths.extend(imgs)

st.markdown("<h1 style='text-align: center; color: white;'>Classification de races de chiens</h1>", unsafe_allow_html=True)
st.write("\n")
st.markdown("Application de prédiction de la race d'un chien présent sur une image.")
st.markdown("Le modèle a été entraîné dans le cadre de la compétition Kaagle [Dog Breed Identification](https://www.kaggle.com/competitions/dog-breed-identification/overview)")
st.markdown("Le jeu de données domprend 120 races de chiens différentes.")
st.markdown("La précision du modèle en test est de 93%.")

st.write("\n\n\n\n")
st.markdown("<h2 style='text-align: center; color: white;'>Uploader une image</h2>", unsafe_allow_html=True)
st.markdown("---")


uploaded_file = st.file_uploader("Choose a dog image")
api_endpoint = "http://dog-breed-api:8000/api/v1/predict"

# Uploaded images
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    
    response = requests.post(api_endpoint, files={"img_bytes": bytes_data},)
    data = response.json()
    predicted_breed = data["breed"]
    image = Image.open(io.BytesIO(bytes_data))
    resizedImg = image.resize((225, 250), Image.LANCZOS)
    cols = st.columns(3, gap="large")
    cols[1].image(resizedImg)
    #st.write(predicted_breed)
    cols[1].markdown(f"##### Predicted Breed : {predicted_breed.replace('_', ' ').title()}")

st.write("\n\n\n\n")
#st.header("Random images", divider="grey")
st.markdown("<h2 style='text-align: center; color: white;'>Images aléatoires</h2>", unsafe_allow_html=True)
st.markdown("---")
st.write("\n\n\n\n")

# Random images
num_examples = 4
example_images = []
api_results = []
cols = st.columns(4, gap="large") 

for image_idx in range(num_examples):
    img_path = random.choice(imgs_paths)
    
    with open(img_path, "rb") as f:
        img_bytes = f.read()
        
    response = requests.post(api_endpoint, files={"img_bytes": img_bytes},)
    data = response.json()
    predicted_breed = data["breed"]
    api_results.append(predicted_breed)
    
    image = Image.open(io.BytesIO(img_bytes))
    
    resizedImg = image.resize((225, 250), Image.LANCZOS)
    cols[(image_idx%2)+1].image(resizedImg, use_column_width=True)
    cols[(image_idx%2)+1].markdown(f"##### Predicted Breed : {predicted_breed.replace('_', ' ').title()}")