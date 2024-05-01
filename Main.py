
from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib as plt
import base64
from collections import Counter
import streamlit.components.v1 as components
import pandas as pd
import streamlit as st
import smtplib
import numpy as np
import plotly.figure_factory as ff
import plotly.figure_factory as px
import plotly.figure_factory as line
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,)
from datasets import (filter_data, filter_dataframe)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions

import numpy as np
import pandas as pd
import csv
import torch
import jovian
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torch.utils.data import DataLoader

train_ds = ImageFolder(train_dir, transform=T.Compose([T.Resize((64,64)),T.ToTensor(),]))
valid_ds = ImageFolder(valid_dir, transform=T.Compose([T.Scale((64,64)),T.ToTensor()]))
test_ds = ImageFolder(test_dir, transform=T.Compose([T.Scale((64,64)),T.ToTensor()]))
train_dl = DataLoader(test_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
val_loader = DataLoader(valid_ds, batch_size*2,shuffle=True, num_workers=3, pin_memory=True)
test_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)

st.title("Image Diagnosis - Aplication")
img_tensor, label = train_ds[0]
st.header(img_tensor.shape, label)

image, label = train_ds[2456]
plt.imshow(image[0], cmap='gray')
st.header('Label:', label)

image, label = train_ds[1]
plt.imshow(image[2], cmap='gray')
st.header('Label:', label)

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.view(xb.shape[0],-1)
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        st.h1("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
model = MnistModel()


for images, labels in train_dl:
    st.header("images.shape: " , images.shape)
    
    st.header(img_tensor.shape, label)
    output = model(images)
    break;
st.header("output.shape: ", output.shape)
st.header("output: ", output[:3].data)

probs = F.softmax(output, dim=1)
st.write("Probability: \n" ,probs[122:126].data)

evaluate(model, val_loader)

loss_fn = F.cross_entropy
loss = loss_fn(output, labels) 

history = fit(5, 0.0001, model, train_dl, val_loader)


accuracies = [r['val_acc'] for r in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');

img, label = test_ds[256]
plt.imshow(img[0], cmap='gray')
st.header('Label:', label, ', Predicted:', predict_image(img, model))
