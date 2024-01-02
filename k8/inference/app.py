import torch
import torchvision.models as models
from torchvision import datasets, transforms
import torchvision
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, json, request
from flask import Flask, render_template, request, redirect, url_for, render_template
app = Flask(__name__)
device = torch.device('cpu')
@app.route('/')
def home():
    return render_template('newhome.html')
#the code of the Net class needed to load the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
def predict2(inputs):
    print("inside predict2")
    inputs = inputs.to(device)
    output = model(inputs).data.numpy().argmax()
    return str(output)
@app.route('/predict', methods=['POST'])
def predict():
    res = {}
    file = request.files['file']
    print("here 1.0")
    if not file:
        print("here 1")
        res['message'] = 'Error. Cannot find image'
        res['Result'] = 'NA'
    else:
        print("here 2")
        res['message'] = 'Success'
        image = Image.open(file)
        image = transform(image).unsqueeze(0)
        res['Result'] = predict2(image)
    print("Returning result json")
    return res['Result']
model = Net()
model.load_state_dict(torch.load("/mnt/ag8733_model.pth"))
model.eval()
