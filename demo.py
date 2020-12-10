import os
from models import Noise2Same
import gradio as gr

os.system("mkdir trained_models/denoising_ImageNet")
os.system("cd trained_models/denoising_ImageNet; gdown https://drive.google.com/uc?id=1asrwULW1lDFasystBc3UfShh5EeTHpkW; gdown https://drive.google.com/uc?id=1Re1ER7KtujBunN0-74QmYrrOx77WpVXK; gdown https://drive.google.com/uc?id=1QdlyUPUKyyGtqD0zBrj5F7qQZtmUELSu; gdown https://drive.google.com/uc?id=1LQsYR26ldHebcdQtP2zt4Mh-ZH9vXQ2S; gdown https://drive.google.com/uc?id=1AxTDD4dS0DtzmBywjGyeJYgDrw-XjYbc; gdown https://drive.google.com/uc?id=1w4UdNAbOjvWSL0Jgbq8_hCniaxqsbLaQ; cd ../..")
os.system("wget -O arch.png https://i.imgur.com/NruRABn.png")
os.system("wget -O parrot.png https://i.imgur.com/zdji3xv.png")
os.system("wget -O lion.png https://i.imgur.com/qNT0lJJ.png")

model = Noise2Same('trained_models/', 'denoising_ImageNet', dim=2, in_channels=3)

def norm(x):
    x = (x-x.min())/(x.max()-x.min())
    return x
    
def predict(img):
  pred = model.predict(img.astype('float32'))
  return norm(pred)

img = gr.inputs.Image()

title = "Noise2Same: Optimizing A Self-Supervised Bound for Image Denoising"
description = "A demo of Noise2Same, an image denoising method developed by Yaochen Xie et al. and presented in NeurIPS 2020. This demo uses the ImageNet-trained model. Try it by uploading an image or clicking on an example (could take up to 20s if running on CPU)."

gr.Interface(predict, "image", "image", examples=[["lion.png"], ["arch.png"], ["parrot.png"]], title=title, description=description).launch()
