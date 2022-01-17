import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import gradio as gr
from urllib.request import urlretrieve
from model import Net

# Loads latest model state from Github
model_timestamps = [filename[10:-3]
                    for filename in os.listdir("./saved_models")]
latest_timestamp = max(model_timestamps)

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

device = torch.device(dev)

model = Net()
model.load_state_dict(torch.load(f"./saved_models/mnist-cnn-{latest_timestamp}.pt", map_location=device))
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
gr.Interface(fn=inference, inputs='sketchpad', outputs=outputs, title="MNIST interface",
             description="Draw a number from 0-9 in the box and click submit to see the model's predictions.").launch()
