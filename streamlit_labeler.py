import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from PIL import Image
import os
from random import randrange
import random
import math

st.title('Label tool')

processed_path = "./processed.pkl"
zips_path = "./zips"
print("start refresh")
if not os.path.exists(zips_path):
    raise FileNotFoundError("folder containing zip files not found!")

if not os.path.exists(processed_path):
    # csv's in dir
    csv_list = ["Selecteer"] + [i for i in os.listdir() if i.endswith(".csv")]
    if len(csv_list) == 1:
        st.write("Place the .csv file in the same directory as the python script")

    else:
        option = st.selectbox('Select .csv file to start', csv_list)
        if option != "Selecteer":
            st.write('Processing ', option, "this can take a while ...")

            df = pd.read_csv(option, sep=";", skiprows=8, nrows=28901, engine='python')  # python engine is nodig omdat de C engine de encoding van de file niet kan lezen
            df = df[df.columns[:-50]]  # Er worden 50 columns teveel ingelezen
            df.dropna(how="all", axis=1, inplace=True)  # Lege collumns verwijderen

            file_names = df["Camera4 : Hyperlink original picture filename"]
            # columns verwijderen
            df.drop("Cause sequence interrupt", axis=1, inplace=True)  # voorlopig verwijderen, later misschien one hot encoden
            for col in df.columns:
                # Columns met 1 unieke value verwijderen
                counts = pd.value_counts(df[col])
                if len(counts) == 1:
                    df.drop(col, axis=1, inplace=True)
                    continue
                # object columns (strings) omzetten naar float
                if df[col].dtype == "object":
                    # als er een komma is dit vervangen met een . en omzetten
                    if df[col].str.contains(',').all():
                        df[col] = df[col].str.replace(',', '.').astype("float32")
                    # Er is geen komma dus het is geen getal
                    else:
                        df.drop(col, axis=1, inplace=True)

                # integer omzetten naar float32
                elif df[col].dtype == "int64":
                    df[col] = df[col].astype("float32")

            df = df.fillna(0)  # Nan's vervangen met 0
            df["file_names"] = file_names

            # voorlopig enkel focussen op camera 4, deze hoort het meest accuraat te zijn
            verkeerde_cameras = ["Camera1", "Camera2", "Camera3", "Camera5"]
            for col in df.columns:
                for camera in verkeerde_cameras:
                    if camera in col:
                        df.drop(col, axis=1, inplace=True)
                        continue
            df.drop("Sequence #", axis=1, inplace=True)

            df = df.set_index("file_names")
            df["Quality_label"] = np.nan
            df.to_pickle("processed.pkl")

            st.write("Processing done")
            st.button("Start labeling")
            print(df.shape)
else:
    def load_data(file_path):
        # data = pd.read_csv(file_path)
        data = pd.read_pickle(file_path)
        # data = data.set_index(["file_names"])
        return data


    @st.cache
    def get_available_images():
        available_zips = []
        for zip in os.listdir(zips_path):
            with zipfile.ZipFile(os.path.join(zips_path, zip), "r") as zip_data:
                available_zips += [i[:-12] for i in zip_data.namelist()]
        return available_zips


    # @st.cache
    def get_unlabled_row(data):
        # random beschikbare image selecteren
        # random.seed(0)
        # random_index = randrange(len(available_images))
        # index = available_images[random_index]
        # row = data.loc[index]
        # if not math.isnan(row["Quality_label"]):
        #     get_unlabled_row(data)
        selection = data.loc[available_images]
        non_labled = selection[selection["Quality_label"].isna()]
        random_index = randrange(len(non_labled))
        row = selection[selection["Quality_label"].isna()].iloc[random_index]
        return row


    @st.cache
    def get_zipped_image(file_name):
        full_name = file_name + "_AIimage.png"
        # find image in zips
        for zip in os.listdir(zips_path):
            with zipfile.ZipFile(os.path.join(zips_path, zip), "r") as zip_data:
                if full_name in zip_data.namelist():
                    imagePIL = Image.open(zip_data.open(full_name))  # Image data lezen uit de zip file
                    imagePIL = np.flip(imagePIL, 0)  # Image horizontaal spiegelen omdat Pil de image omgekeerd inleest
                    imagePIL = imagePIL[348:482]
                    return imagePIL
        raise Exception("Image is niet gevonden")


    data = load_data(processed_path)
    available_images = get_available_images()

    app_state = st.experimental_get_query_params()
    if "file_name" in app_state:
        row = data.loc[app_state["file_name"][0]]
    else:
        row = get_unlabled_row(data)
        st.experimental_set_query_params(file_name=row.name)

    image_filename = row.name

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        # st.dataframe(data)
        aantal_gelabelde = len(data) - len(data[data["Quality_label"].isna()])
        st.write("Aantal gelabelde:", aantal_gelabelde)
        # st.dataframe(data[["MV Pressure (bar)", "RV4 off time (ms)", "file_names", "Quality_label"]])
        st.dataframe(row)

    st.write("file name:", image_filename)

    image = get_zipped_image(image_filename)
    st.image(image)

    start_pos = row["Quality_label"] if type(row["Quality_label"]) is np.nan else 5
    qual = st.slider("What's the quality score:", 0, 10, start_pos)

    if st.button('Confirm label'):
        row["Quality_label"] = qual
        # data.loc[image_filename, "Quality_label"] = qual
        data.loc[image_filename] = row
        print(data.loc[image_filename])
        print(data.loc[image_filename]["Quality_label"])
        data.to_pickle("./processed.pkl")
        st.write("Label:", qual, "confirmed")
        st.write('On to the next ...')

        st.experimental_set_query_params()
        st.experimental_rerun()

    if st.button("Export to .csv"):
        data.to_csv("./processed_labels.csv", index=True)
        st.write('Saved to processed_labels.csv')
