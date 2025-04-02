import torch
import torch.nn as nn
import timm


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Downward(nn.Module):
    def __init__(self):
        super(Downward, self).__init__()
        self.conv1 = DoubleConv(1, 16)
        self.conv2 = DoubleConv(16, 32)
        self.conv3 = DoubleConv(32, 64)
        self.conv4 = DoubleConv(64, 128)
        self.maxpool = nn.MaxPool2d(2)
        self.fc = nn.Linear(128*28*28, 768)
        self.features = {}

    def forward(self, x):
        # B, 1, 224, 224
        x = self.conv1(x)
        self.features['conv1'] = x
        # B, 16, 224, 224
        x = self.maxpool(x)
        # B, 16, 112, 112
        x = self.conv2(x)
        self.features['conv2'] = x
        # B, 32, 112, 112
        x = self.maxpool(x)
        # B, 32, 56, 56
        x = self.conv3(x)
        self.features['conv3'] = x
        # B, 64, 56, 56
        x = self.maxpool(x)
        # B, 64, 28, 28
        x = self.conv4(x)
        # B, 128, 28, 28
        B, _, _, _ = x.shape
        return self.fc(x.reshape(B, -1)) # [B, 768]


class Upward(nn.Module):
    def __init__(self):
        super(Upward, self).__init__()
        self.fc = nn.Linear(768, 128 * 28 * 28)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)

        self.conv1 = DoubleConv(128, 64)
        self.conv2 = DoubleConv(64, 32)
        self.conv3 = DoubleConv(32, 16)

        self.out = nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_res):
        # x_res contains intermediate conv faetures conv1, conv2, conv3

        # B, 768
        x = self.fc(x)
        B, _ = x.shape
        x = x.reshape(B, 128, 28, 28)
        # B, 128, 28, 28
        x = self.up1(x)
        # B, 64, 56, 56
        x = torch.cat((x, x_res['conv3']), dim=1)
        # B, 128, 56, 56
        x = self.conv1(x)
        # B, 64, 56, 56
        x = self.up2(x)
        # B, 32, 112, 112
        x = torch.cat((x, x_res['conv2']), dim=1)
        # B, 64, 112, 112
        x = self.conv2(x)
        # B, 32, 112, 112
        x = self.up3(x)
        # B, 16, 224, 224
        x = torch.cat((x, x_res['conv1']), dim=1)
        # B, 32, 224, 224
        x = self.conv3(x)
        # B, 16, 224, 224

        x = self.out(x)  # B, 1, 224, 224

        return self.sigmoid(x)


class ViT(nn.Module):
    def __init__(self, num_frames=4):
        super().__init__()
        # Load pretrained ViT (excluding patch embedding)
        self.vit = timm.create_model('vit_base_patch16_224',
                                     pretrained=True,
                                     num_classes=0)

        # Remove original patch embedding
        del self.vit.patch_embed

        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames + 1, 768))

        # Initialize CLS token from original ViT
        self.cls_token = self.vit.cls_token

    def forward(self, x):
        """Input: [batch_size, num_frames, 768]"""
        batch_size = x.shape[0]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, 768]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 5, 768]

        # Add temporal positional encoding
        x += self.temporal_pos

        # Process through ViT
        return self.vit.norm(self.vit.blocks(x))  # [B, 5, 768]


class Network(nn.Module):
    def __init__(self, num_frames=4):
        super(Network, self).__init__()
        self.encoders = nn.ModuleList([
            Downward()
            for _ in range(num_frames)
        ])
        self.num_frames = num_frames

        self.decoder = Upward()

        self.vit = ViT()

    def forward(self, x):
        x = [self.encoders[i](x[:, i, :, :].unsqueeze(1)) for i in range(self.num_frames)]
        x = torch.stack(x, dim=1)
        x = self.vit(x)
        x = self.decoder(x[:, 0, :], self.encoders[3].features)

        return x
