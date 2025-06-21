import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import Generator  # Your class from model.py
from torch.autograd import Variable

# Load generator model
@st.cache_resource
def load_generator():
    model = Generator()
    model.load_state_dict(torch.load("generator_state.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

# Function to generate 5 images
def generate_digits(generator, digit, n=5):
    images = []
    for _ in range(n):
        z = Variable(torch.randn(1, 100))
        label = torch.LongTensor([digit])
        with torch.no_grad():
            img = generator(z, label).data.cpu()
        img = 0.5 * img + 0.5  # Rescale from [-1, 1] to [0, 1]
        img_pil = transforms.ToPILImage()(img.squeeze(0))
        images.append(img_pil)
    return images

# --- Streamlit UI ---
st.set_page_config(page_title="Digit Generator", layout="wide")
st.title("🧠 Handwritten Digit Generator (0–9)")

digit = st.slider("Choose a digit to generate", 0, 9, 0)

if st.button("Generate"):
    gen = load_generator()
    images = generate_digits(gen, digit, n=5)

    st.subheader(f"Generated Samples for Digit {digit}")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.image(images[i], caption=f"Sample {i+1}", use_column_width=True)
