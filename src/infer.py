import torch
import torchaudio
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from src.model import YourModel

# Load model
model = YourModel()
model.load_state_dict(torch.load("model_checkpoint.pth", map_location="cpu"))
model.eval()

# Load audio
(rate, sig) = wav.read("sample.wav")
features = mfcc(sig, rate)

# Convert to tensor
input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    print("Model output:", output)

# Post-process output if necessary
