# Import necessary modules from PyTorch for neural network development
import torch
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn.functional as F
from torch.nn import (
    Sequential, Conv2d, Linear, ReLU, Dropout, Module,
    MaxPool2d, BatchNorm2d, L1Loss, CrossEntropyLoss,
    BCEWithLogitsLoss, SmoothL1Loss, MSELoss, LSTM
)

# Define a CNN class that inherits from ResNet for image feature extraction
class CNN(ResNet):
    def __init__(self, config, v_weights, o_weights, t_weights, e_weights):
        # Initialize the ResNet model with a basic block and layer configuration [2,2,2,2] similar to ResNet18
        super(CNN, self).__init__(block=BasicBlock, layers=[2,2,2,2])
        # Configuration settings for the model, including input channels and LSTM time dimension
        self.input_dim = config.n_channels
        self.t_dim = config.t_dim
        self.dropout = config.dropout
        # Define the first convolutional layer adjusted for the input dimensions
        self.conv1 = Conv2d(self.input_dim, out_channels=64, kernel_size=(7,7), 
                            stride=(2,2), padding=(3,3), bias=False)

        # Define four sequential Bidirectional LSTM networks for different output processing
        self.rnn1 = Sequential(BidirectionalLSTM(512, 128, 128), BidirectionalLSTM(128, 128, 4))
        self.rnn2 = Sequential(BidirectionalLSTM(512, 128, 128), BidirectionalLSTM(128, 128, 4))
        self.rnn3 = Sequential(BidirectionalLSTM(512, 128, 128), BidirectionalLSTM(128, 128, 4))
        self.rnn4 = Sequential(BidirectionalLSTM(512, 128, 128), BidirectionalLSTM(128, 128, 4))
        
        # Loss functions for each LSTM output, potentially weighted for class imbalance
        self.loss1 = CrossEntropyLoss(weight=torch.tensor(v_weights))
        self.loss2 = CrossEntropyLoss(weight=torch.tensor(o_weights))
        self.loss3 = CrossEntropyLoss(weight=torch.tensor(t_weights))
        self.loss4 = CrossEntropyLoss(weight=torch.tensor(e_weights))
        
    def forward(self, batch, time_steps):
        # Forward pass of the model, processes a batch through CNN and LSTM stages
        # Pre-allocate tensor for CNN outputs with shape [time_steps, batch_size, 512]
        cnn_outputs = torch.zeros([time_steps, len(batch['name']), 512])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        frames = []
       
        # Process each frame in the time sequence through the CNN
        for j in range(time_steps):
            img = batch['imgs'][j][:,0,:,:].unsqueeze(1)  # Extract and adjust dimensions of the image
            frames.append(batch['name'])
            img = img.to(device, dtype=torch.float)  # Move image to the configured device
            # Pass the image through the CNN layers defined in the ResNet base class
            img = self.conv1(img)
            img = self.bn1(img)
            img = self.relu(img)
            img = self.maxpool(img)
            img = self.layer1(img)
            img = self.layer2(img)
            img = self.layer3(img)
            img = self.layer4(img)
            img = self.avgpool(img)
            img = torch.flatten(img, 1)  # Flatten the output for the LSTM
            cnn_outputs[j,:,:] = img  # Store the CNN output
        
        # Move the CNN outputs to the same device as the model
        cnn_outputs = cnn_outputs.to(device, dtype=torch.float)
        # Pass the CNN outputs through each of the LSTM networks
        rnn_output1 = self.rnn1(cnn_outputs)
        rnn_output2 = self.rnn2(cnn_outputs)
        rnn_output3 = self.rnn3(cnn_outputs)
        rnn_output4 = self.rnn4(cnn_outputs)
        # Return the outputs from the LSTM networks
        return rnn_output1, rnn_output2, rnn_output3, rnn_output4
    

# Define a Bidirectional LSTM network module for sequence processing
class BidirectionalLSTM(Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        # LSTM layer with bidirectional processing
        self.rnn = LSTM(nIn, nHidden, bidirectional=True)
        # Linear layer to map LSTM outputs to the desired output size
        self.embedding = Linear(nHidden * 2, nOut)

    def forward(self, features):
        # Forward pass through the LSTM and linear layers
        recurrent, _ = self.rnn(features)  # Process features through LSTM
        T, b, h = recurrent.size()  # Get the dimensions of LSTM output
        t_rec = recurrent.view(T * b, h)  # Reshape for linear layer

        output = self.embedding(t_rec)  # Process through linear layer
        output = output.view(T, b, -1)  # Reshape back to sequence format
        return output
