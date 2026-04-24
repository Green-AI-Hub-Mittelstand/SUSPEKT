import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class ZustandModel(nn.Module):
    def __init__(self, input_shape, num_features, num_labels, feat_active):
        super(ZustandModel, self).__init__()

        # Define the feature activation function
        if feat_active == 'relu':
            self.feat_activation = nn.ReLU()
        elif feat_active == 'sigmoid':
            self.feat_activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {feat_active}")

        # Define the layers
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.dropout1 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.dropout2 = nn.Dropout(0.25)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.dropout3 = nn.Dropout(0.25)

        # Fully connected layers
        flat_size = self._get_flat_size(input_shape)

        self.fc1 = nn.Linear(16384, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_features)
        self.fc5 = nn.Linear(num_features, num_labels)

        self.dropout_fc = nn.Dropout(0.5)

    def _get_flat_size(self, input_shape):

        x = torch.zeros(1, *input_shape)  # Create a dummy input tensor with shape [1, C, H, W]
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))  # After conv1 -> conv2 -> pool1
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))  # After conv3 -> conv4 -> pool2
        x = self.pool3(F.relu(self.conv5(x)))  # After conv5 -> pool3
        x = self.pool4(F.relu(self.conv6(x)))  # After conv6 -> pool4
        flattened_size = x.shape[1] * x.shape[2] * x.shape[3]  # Flatten the output shape
        return flattened_size

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.dropout1(x)
        x = self.pool3(F.relu(self.conv5(x)))
        x = self.dropout2(x)
        x = self.pool4(F.relu(self.conv6(x)))
        x = self.dropout3(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc3(x))
        x = self.dropout_fc(x)
        x = self.feat_activation(self.fc4(x))
        x = F.softmax(self.fc5(x), dim=1)
        return x


class Single_Transformer():
    def __init__(self, width, height):
        self.transform = transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ToTensor(),  # Converts to (C, H, W) and scales pixel values to [0, 1]
            transforms.Lambda(lambda x: x.unsqueeze(0))
        ])

    def __call__(self, image):
        return self.transform(image)
