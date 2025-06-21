import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import Generator  # Defined exactly like in your notebook

# Load generator model
@st.cache_resource
def load_generator():
    model = Generator()
    model.load_state_dict(torch.load("generator_state.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

def generate_digit(generator, digit):
    images = []
    for _ in range(5):
        z = torch.randn(1, 100)
        label = torch.LongTensor([digit])
        with torch.no_grad():
            img = generator(z, label).squeeze(0).cpu()
        img = 0.5 * img + 0.5  # Rescale from [-1, 1] to [0, 1]
        img = transforms.ToPILImage()(img)
        images.append(img)
    return images

# --- Streamlit UI ---
st.title("Handwritten Digit Generator (0–9)")

digit = st.slider("Choose a digit", 0, 9, 0)

if st.button("Generate"):
    gen = load_generator()
    images = generate_digit(gen, digit)

    st.subheader(f"Generated Samples for Digit {digit}")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(images[i], use_column_width=True)
