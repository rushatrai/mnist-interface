import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import gradio as gr
from urllib.request import urlretrieve

# Loads latest model state from Github
urlretrieve("https://github.com/equ1/mnist-interface/tree/main/demo_model.pt", "demo_model.pt")

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

device = torch.device(dev)

model = torch.load(f"./saved_models/mnist-cnn-{latest_timestamp}.pt", map_location=device)
model.eval()

# inference function
def inference(img):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))])
    img = transform(img).unsqueeze(0)  # transforms ndarray and adds batch dimension

    with torch.no_grad():
        output_probabilities = F.softmax(model(img), dim=1)[0]  # probability prediction for each label

    return {labels[i]: float(output_probabilities[i]) for i in range(len(labels))}

# Creates and launches gradio interface
labels = range(10)  # 1-9 labels
outputs = gr.outputs.Label(num_top_classes=5)
gr.Interface(fn=inference, inputs='sketchpad', outputs=outputs, title="MNIST Interface",
             description="Draw a number from 0-9 in the box and click submit to see the model's predictions.").launch()
