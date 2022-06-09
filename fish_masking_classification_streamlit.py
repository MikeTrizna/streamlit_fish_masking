import streamlit as st
import os
import urllib
import fastai.vision.all as fai_vision
import numpy as np
from pathlib import Path
import pathlib
from PIL import Image
import platform
import altair as alt
import pandas as pd

def main():
    st.title('Fish Masker and Classifier')

    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)
    
    data_loader, segmenter = load_unet_model()
    classification_model = load_classification_model()
    
    st.markdown("Upload an Amazonian fish photo for masking.")
    uploaded_image = st.file_uploader("", IMAGE_TYPES)
    if uploaded_image:
        image_data = uploaded_image.read()
        st.markdown('## Original image')
        st.image(image_data, use_column_width=True)

        original_pil = Image.open(uploaded_image)

        original_pil.save('original.jpg')

        single_file = [Path('original.jpg')]
        single_pil = Image.open(single_file[0])
        input_dl = segmenter.dls.test_dl(single_file)
        masks, _ = segmenter.get_preds(dl=input_dl)
        masked_pil, percentage_fish = mask_fish_pil(single_pil, masks[0])

        st.markdown('## Masked image')
        st.markdown(f'**{percentage_fish:.1f}%** of pixels were labeled as "fish"')
        st.image(masked_pil, use_column_width=True)

        masked_pil.save('masked.jpg')

        st.markdown('## Classification')

        prediction = classification_model.predict('masked.jpg')
        pred_chart = predictions_to_chart(prediction, classes = classification_model.dls.vocab)
        st.altair_chart(pred_chart, use_container_width=True)


def mask_fish_pil(unmasked_fish, fastai_mask):
    unmasked_np = np.array(unmasked_fish)
    np_mask = fastai_mask.argmax(dim=0).numpy()
    total_pixels = np_mask.size
    fish_pixels = np.count_nonzero(np_mask)
    percentage_fish = (fish_pixels / total_pixels) * 100
    np_mask = (255 / np_mask.max() * (np_mask - np_mask.min())).astype(np.uint8)
    np_mask = np.array(Image.fromarray(np_mask).resize(unmasked_np.shape[1::-1], Image.BILINEAR))
    np_mask = np_mask.reshape(*np_mask.shape, 1) / 255
    masked_fish_np = (unmasked_np * np_mask).astype(np.uint8)
    masked_fish_pil = Image.fromarray(masked_fish_np)
    return masked_fish_pil, percentage_fish

def predictions_to_chart(prediction, classes):
    pred_rows = []
    for i, conf in enumerate(list(prediction[2])):
        pred_row = {'class': classes[i],
                    'probability': round(float(conf) * 100,2)}
        pred_rows.append(pred_row)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.head()
    top_probs = pred_df.sort_values('probability', ascending=False).head(4)
    chart = (
        alt.Chart(top_probs)
        .mark_bar()
        .encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=(0, 100))),
            y=alt.Y("class:N",
                    sort=alt.EncodingSortField(field="probability", order="descending"))
        )
    )
    return chart

@st.cache(allow_output_mutation=True)
def load_unet_model():
    data_loader = fai_vision.SegmentationDataLoaders.from_label_func(
        path = Path("."),
        bs = 1,
        fnames = [Path('test_fish.jpg')],
        label_func = lambda x: x,
        codes = np.array(["Photo", "Masks"], dtype=str),
        item_tfms = [fai_vision.Resize(256, method = 'squish'),],
        batch_tfms = [fai_vision.IntToFloatTensor(div_mask = 255)],
        valid_pct = 0.2, num_workers = 0)
    segmenter = fai_vision.unet_learner(data_loader, fai_vision.resnet34)
    segmenter.load('fish_mask_model')
    return data_loader, segmenter

@st.cache(allow_output_mutation=True)
def load_classification_model():
    plt = platform.system()

    if plt == 'Linux' or plt == 'Darwin': 
        pathlib.WindowsPath = pathlib.PosixPath
    inf_model = fai_vision.load_learner('models/fish_classification_model.pkl', cpu=True)

    return inf_model


def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()
    
    return

IMAGE_TYPES = ["png", "jpg","jpeg"]

EXTERNAL_DEPENDENCIES = {
    "models/fish_mask_model.pth": {
        "url": "https://figshare.com/ndownloader/files/31976030?private_link=2e7b8378b20d3537a643",
        "size": 494929527
    },
    "models/fish_classification_model.pkl": {
        "url": "https://figshare.com/ndownloader/files/31975979?private_link=2e7b8378b20d3537a643",
        "size": 179319095
    }
}

if __name__ == "__main__":
    main()