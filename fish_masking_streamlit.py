import streamlit as st
import os
import urllib
import fastai.vision.all as fai_vision
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def main():
    st.title('Fish Masker')

    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)
    
    data_loader, segmenter = load_model()
    
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
        masked_pil = mask_fish_pil(single_pil, masks[0])

        st.markdown('## Masked image')
        st.image(masked_pil, use_column_width=True)

def mask_fish_pil(unmasked_fish, fastai_mask):
    unmasked_np = np.array(unmasked_fish)
    np_mask = fastai_mask.argmax(dim=0).numpy()
    np_mask = (255 / np_mask.max() * (np_mask - np_mask.min())).astype(np.uint8)
    np_mask = np.array(Image.fromarray(np_mask).resize(unmasked_np.shape[1::-1], Image.BILINEAR))
    np_mask = np_mask.reshape(*np_mask.shape, 1) / 255
    masked_fish_np = (unmasked_np * np_mask).astype(np.uint8)
    masked_fish_pil = Image.fromarray(masked_fish_np)
    return masked_fish_pil

@st.cache(allow_output_mutation=True)
def load_model():
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
    segmenter.load('fish_mask_model_2021_08_17')
    return data_loader, segmenter


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

IMAGE_TYPES = ["png", "jpg"]

EXTERNAL_DEPENDENCIES = {
    "models/fish_mask_model_2021_08_17.pth": {
        "url": "https://www.dropbox.com/s/e9c4oi6tf5qnqyd/fish_mask_model_2021_08_17.pth?dl=1",
        "size": 494929527
    }
}

if __name__ == "__main__":
    main()