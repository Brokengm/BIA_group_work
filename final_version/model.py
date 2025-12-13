# models.py

# ----------------- Part 1: General & DL dependencies -----------------
import os
import torch
import torch.nn as nn
import timm

# Base paths and constants
# (These paths are needed for model loading)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# these files should be in the same directory as this script
UNET_WEIGHTS_PATH = os.path.join(BASE_DIR, "384_unet.pth")
EFFICIENTNET_WEIGHTS_PATH = os.path.join(BASE_DIR, "efficientnet.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# UNet segmentation model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()
# The basic block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    def _block(self, in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        logits = self.conv(dec1)
        return self.sigmoid(logits)


# Classification model (EfficientNet-B3 + FC head)
class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = timm.create_model(
            'efficientnet_b3',
            pretrained=False,
            num_classes=0
        )
        num_features = self.cnn_model.num_features

        self.cnn_projection = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 30)
        )

        self.final_layers = nn.Sequential(
            nn.Linear(30, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        x1 = self.cnn_model(x)
        x1 = self.cnn_projection(x1)
        x = self.final_layers(x1)
        return x


_unet_instance = None
_effnet_instance = None


def get_unet_model():
    global _unet_instance
    if _unet_instance is None:
        if not os.path.exists(UNET_WEIGHTS_PATH):
            raise FileNotFoundError(f"UNet weights not found: {UNET_WEIGHTS_PATH}")

        model = UNet(in_channels=3, out_channels=1).to(DEVICE)
        model.eval()
        #load model weights
        checkpoint = torch.load(UNET_WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint
        model.load_state_dict(state)

        _unet_instance = model

    return _unet_instance


def get_effnet_model():
    global _effnet_instance
    if _effnet_instance is None:
        if not os.path.exists(EFFICIENTNET_WEIGHTS_PATH):
            raise FileNotFoundError(f"EfficientNet weights not found: {EFFICIENTNET_WEIGHTS_PATH}")

        model = CombinedModel().to(DEVICE)
        model.eval()
        # load model weights
        checkpoint = torch.load(EFFICIENTNET_WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint
        model.load_state_dict(state)

        _effnet_instance = model

    return _effnet_instance
