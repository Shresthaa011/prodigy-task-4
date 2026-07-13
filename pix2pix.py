import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. Generator (U-Net based Architecture)
# --------------------------------------------------------
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


# --------------------------------------------------------
# 2. Discriminator (PatchGAN Architecture)
# --------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


# --------------------------------------------------------
# 3. Test the Architecture
# --------------------------------------------------------
def test():
    print("Initializing Generator (U-Net) and Discriminator (PatchGAN)...")
    generator = GeneratorUNet()
    discriminator = Discriminator()

    # Create dummy images (Batch Size, Channels, Height, Width)
    # Shape: (1, 3, 256, 256)
    dummy_input_image = torch.randn((1, 3, 256, 256))
    dummy_target_image = torch.randn((1, 3, 256, 256))

    print(f"Input image shape: {dummy_input_image.shape}")
    
    # Pass through Generator
    fake_generated_image = generator(dummy_input_image)
    print(f"Generator output shape: {fake_generated_image.shape}")
    
    # Pass through Discriminator
    validity = discriminator(fake_generated_image, dummy_input_image)
    print(f"Discriminator output shape: {validity.shape} (PatchGAN output grid)")
    
    # Visualize the architecture test
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(dummy_input_image[0].permute(1, 2, 0).detach().numpy() * 0.5 + 0.5)
    axes[0].set_title("Input Image (Random Noise)")
    axes[0].axis('off')
    
    axes[1].imshow(dummy_target_image[0].permute(1, 2, 0).detach().numpy() * 0.5 + 0.5)
    axes[1].set_title("Target Image (Random Noise)")
    axes[1].axis('off')
    
    axes[2].imshow(fake_generated_image[0].permute(1, 2, 0).detach().numpy() * 0.5 + 0.5)
    axes[2].set_title("Generator Output")
    axes[2].axis('off')
    
    plt.savefig("architecture_test.png")
    print("Test visualization saved as 'architecture_test.png'.")
    print("Pix2Pix architecture successfully implemented!")

if __name__ == "__main__":
    test()
