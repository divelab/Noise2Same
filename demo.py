import os, fnmatch, cv2
import numpy as np
import matplotlib.pyplot as plt
from models import Noise2Same

os.system("mkdir trained_models/denoising_ImageNet")
os.system("cd trained_models/denoising_ImageNet")
os.system("gdown https://drive.google.com/uc?id=1asrwULW1lDFasystBc3UfShh5EeTHpkW")
os.system("gdown https://drive.google.com/uc?id=1Re1ER7KtujBunN0-74QmYrrOx77WpVXK")
os.system("gdown https://drive.google.com/uc?id=1QdlyUPUKyyGtqD0zBrj5F7qQZtmUELSu")
os.system("gdown https://drive.google.com/uc?id=1LQsYR26ldHebcdQtP2zt4Mh-ZH9vXQ2S")
os.system("gdown https://drive.google.com/uc?id=1AxTDD4dS0DtzmBywjGyeJYgDrw-XjYbc")
os.system("gdown https://drive.google.com/uc?id=1w4UdNAbOjvWSL0Jgbq8_hCniaxqsbLaQ")
os.system("cd ../..")

model = Noise2Same('trained_models/', 'denoising_ImageNet', dim=2, in_channels=3)

def norm(x):
    x = (x-x.min())/(x.max()-x.min())
    return x
    
def predict(img):
  pred = model.predict(img.astype('float32'))
  return norm(pred)

img = gr.inputs.Image()

gr.Interface(predict, "image", "image", examples=[["lion2.png"]], title="Noise2Same: Optimizing A Self-Supervised Bound for Image Denoising").launch()
