import torch
import torch.nn.functional as F
from torch import nn
from pl_bolts.models.autoencoders.components import EncoderBlock, DecoderBlock

# TODO: Libraries only for debugging. Should be latter removed.
import numpy as np
from PIL import Image


class ResNetEncoder(nn.Module):

    def __init__(self, first_layer_num_filters=64):

        super().__init__()
        self.first_layer_num_filters = first_layer_num_filters

        ##### First convolution
        self.conv = nn.Conv2d(3, first_layer_num_filters, kernel_size=7, stride=4, padding=3)
        self.bn = nn.BatchNorm2d(first_layer_num_filters)
        self.relu = nn.ReLU(inplace=True)

        ##### Define Convolutional Layers to downsample image on residual connection.
        residual_conv_1 = nn.Conv2d(first_layer_num_filters, first_layer_num_filters*2, kernel_size=3, stride=2, padding=1)
        residual_conv_2 = nn.Conv2d(first_layer_num_filters*2, first_layer_num_filters*2, kernel_size=3, stride=2, padding=1)

        ##### Residual layers
        self.residual_1 = EncoderBlock(first_layer_num_filters, first_layer_num_filters*2, stride=2, downsample=residual_conv_1)
        self.residual_2 = EncoderBlock(first_layer_num_filters*2, first_layer_num_filters*2, stride=2, downsample=residual_conv_2)

    
    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.residual_1(x)
        z = self.residual_2(x)

        return z


    def compute_latent_shape(self, input_shape):
        latent_shape = [self.first_layer_num_filters*2, int(input_shape[1]/16), int(input_shape[1]/16)]
        return latent_shape



class ResNetDecoder(nn.Module):

    def __init__(self, last_layer_num_filters=64):

        ##### Module Builder.
        super().__init__()

        ##### Convolution for upsampling on residual branch.
        residual_deconv_1 = nn.ConvTranspose2d(last_layer_num_filters*2, last_layer_num_filters*2, kernel_size=3, stride=2, padding=1)
        residual_deconv_2 = nn.ConvTranspose2d(last_layer_num_filters*2, last_layer_num_filters, kernel_size=3, stride=2, padding=1)
        
        ##### Pad before upsampling
        upsample_1 = nn.Sequential(residual_deconv_1, nn.ConstantPad2d((1, 0, 1, 0), 0))
        upsample_2 = nn.Sequential(residual_deconv_2, nn.ConstantPad2d((1, 0, 1, 0), 0))

        ##### Residual deconvolution.
        self.inverse_residual_1 = DecoderBlock(last_layer_num_filters*2, last_layer_num_filters*2, scale=2, upsample=upsample_1)
        self.inverse_residual_2 = DecoderBlock(last_layer_num_filters*2, last_layer_num_filters, scale=2, upsample=upsample_2)

        ##### Apply transpose convolution
        self.transpose_conv = nn.ConvTranspose2d(last_layer_num_filters, 3, 7, stride=4, padding=1)

    
    def forward(self, z):

        z = self.inverse_residual_1(z)
        z = self.inverse_residual_2(z)
        x = self.transpose_conv(z)

        return x[:, :, :-1, :-1]



if __name__ == "__main__":
    encoder = ResNetEncoder().float()
    decoder = ResNetDecoder().float()

    numpy_img = np.transpose(np.array(Image.open("Datasets/Ultra_Eye_Samples/Training_Images/C37_UHD_35.png")), (2, 0, 1))
    tensor_img = torch.from_numpy(numpy_img)
    tensor_img = tensor_img.type(torch.float) / 255.
    tensor_img = torch.unsqueeze(tensor_img, 0)

    latent = encoder(tensor_img)
    rec    = decoder(latent)

    print(latent.shape, rec.shape)