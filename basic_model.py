import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from resnet_model import ResNetEncoder, ResNetDecoder
#from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder

import torch 
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class VAE(pl.LightningModule):

    def __init__(self, resnet_filters=128):        

        ##### Run builder method defined in parent class.
        super().__init__()
        self.save_hyperparameters()

        ##### Instantiate encoder and decoder
        self.encoder = ResNetEncoder(resnet_filters)
        self.decoder = ResNetDecoder(resnet_filters)

        ##### Obtain distribution parameters from latent output.
        self.flatten = nn.Flatten()
        self.latent_conv = nn.Conv2d(resnet_filters, resnet_filters*2, kernel_size=3, padding=1)

        ##### Learned variance for Gaussian distribution P(x|z)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))


    ##### Define Model Optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


    def gaussian_likelihood(self, mean, sample):
        scale = torch.exp(self.log_scale)
        ##### Define reconstructed image distribution, given by P(x|z) ~ N(pixels, 1)
        dist = torch.distributions.Normal(mean, scale)
        ##### Sample from distribution
        log_pxz = dist.log_prob(sample)
        ##### Get log probability of the original image belonging to the reconstructed distribution.
        prob_sum = log_pxz.sum(dim=(1, 2, 3))
        return prob_sum


    def kl_divergence(self, z):
        ##### Define latent probability as being Standard Normal
        p = torch.distributions.Normal(torch.zeros_like(self.q.mean), torch.ones_like(self.q.stddev))
        ##### Get probability of belonging to the distributions obtained.
        log_qzx = self.q.log_prob(z)
        log_pz = p.log_prob(z)
        ##### Compute KL divergence
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)

        return kl


    def encode_batch(self, x):
        ##### Get encoder latent and generate learned parameters.
        x_encoded = self.encoder(x)
        latent_shape = x_encoded.shape
        flattened_latent = self.flatten(x_encoded)
        mu, log_var = torch.tensor_split(flattened_latent, 2, axis=1)
        ##### Define learned multivariate distribution Q(z|x).
        std = torch.exp(log_var / 2)
        self.q = torch.distributions.Normal(mu, std)
        ##### Sample latent from distribution.
        z = self.q.rsample()
        
        return z, latent_shape

    
    def decode_latent(self, z, latent_shape):
        ##### Reshape 'z' tensor with same spatial dimensions and half channels from latent output by encoder.
        z = z.view((latent_shape[0], latent_shape[1]//2, *latent_shape[2:]))
        ##### Double channel dimension
        latent = self.latent_conv(z)
        ##### Decode latent
        x_hat = self.decoder(latent)
        
        return x_hat


    def compute_losses(self, x, z, x_hat):
        ##### Compute Reconstruction Loss
        recon_lkh = self.gaussian_likelihood(x_hat, x)
        ##### Compute KL divergence from Monte Carlo Approach
        kl = self.kl_divergence(z)
        ##### Combine losses and define ELBO loss.
        elbo = (kl - recon_lkh)
        elbo = elbo.mean()

        return elbo, kl, recon_lkh


    def create_dictionary(self, elbo, kl, recon_lkh):
        ##### Define dictionary with metrics.
        dictionary = {
            'loss': elbo,
            'kl': kl.mean(),
            'recon_lkh': recon_lkh.mean()
        }

        return dictionary


    def get_scalars_and_images(self, dictionary):
        ##### Compute epoch means.
        mean_losses = torch.mean(torch.as_tensor([[batch['loss'], batch['kl'], batch['recon_lkh']] for batch in dictionary]), 0)
        ##### Get images
        random_index = np.random.choice(len(dictionary))
        original_image = dictionary[random_index]['original_img']
        recon_image = dictionary[random_index]['recon_img']

        return mean_losses, original_image, recon_image


    def training_step(self, batch, batch_idx):
        ########## Run batch through network
        ##### Get images from batch.
        x, _ = batch
        ##### Encode Batch
        z, latent_shape = self.encode_batch(x)
        ##### Reconstruct Image from sampled latent.
        x_hat = self.decode_latent(z, latent_shape)
        ########## Compute Losses
        elbo, kl, recon_lkh = self.compute_losses(x, z, x_hat)
        ##### Return batch information
        batch_dictionary = self.create_dictionary(elbo, kl, recon_lkh)
        ##### Save sample as example.
        if np.random.choice([True, False], p=[0.05, 0.95]):
            random_index = np.random.choice(len(x))
            self.orig_training_sample = x[random_index]
            self.rec_training_sample = x_hat[random_index]

        return batch_dictionary


    def validation_step(self, batch, batch_idx):
        ##### Validate images
        x, _ = batch
        z, latent_shape = self.encode_batch(x)
        x_hat = self.decode_latent(z, latent_shape)
        ##### Get Losses
        elbo, kl, recon_lkh = self.compute_losses(x, z, x_hat)
        ##### Create dictionary
        val_dictionary = self.create_dictionary(elbo, kl, recon_lkh)
        ##### Save sample as example.
        if np.random.choice([True, False]):
            self.orig_validation_sample = x[0]
            self.rec_validation_sample = x_hat[0]

        return val_dictionary


    def training_epoch_end(self, outputs):
        ##### Get scalars and images
        mean_losses = torch.mean(torch.as_tensor([[batch['loss'], batch['kl'], batch['recon_lkh']] for batch in outputs]), 0)
        ##### Include scalars in tensorboard
        list(map(lambda metric, value: self.logger.experiment.add_scalar(metric, value, self.current_epoch), 
                                            ['ELBO_Loss', 'KL_Loss', 'Recon_Likelihood'], mean_losses))
        ##### Include images
        try:
            self.logger.experiment.add_image("Original_Image", self.orig_training_sample, self.current_epoch)
            self.logger.experiment.add_image("Reconstructed_Image", self.rec_training_sample, self.current_epoch)
        except AttributeError:
            pass


    def validation_epoch_end(self, validation_step_outputs):
        ##### Get scalars and images
        mean_losses = torch.mean(torch.as_tensor([[batch['loss'], batch['kl'], batch['recon_lkh']] for batch in validation_step_outputs]), 0)
        ##### Include scalars in tensorboard
        list(map(lambda metric, value: self.logger.experiment.add_scalar(metric, value, self.current_epoch), 
                                            ['Val_ELBO_Loss', 'Val_KL_Loss', 'Val_Recon_Likelihood'], mean_losses))
        ##### Include images
        try:
            self.logger.experiment.add_image("Val_Original_Image", self.orig_validation_sample, self.current_epoch)
            self.logger.experiment.add_image("Val_Reconstructed_Image", self.rec_validation_sample, self.current_epoch)
        except AttributeError:
            pass



if __name__ == "__main__":
    ##### Define parser
    parser = argparse.ArgumentParser(description="Receives arguments for encoding.")
    ##### Define arguments.
    parser.add_argument('--training_images_folder', required=True, help='Path to folder with subfolders.')
    parser.add_argument('--testing_images_folder', required=True, help='Path to folder with subfolders.')
    parser.add_argument('--patch_dimension', default=256, type=int, help="Patch dimension used in training stage.")
    parser.add_argument('--tsb_folder', default="tensorboard_logs", help="Folder to save Tensorboard logs.")
    parser.add_argument('--model_logs_folder', default="vae_with_rn18", help='Folder to save logs of current running.')
    parser.add_argument('--batchsize', default=8, type=int, help='Batch size.')
    parser.add_argument('--gpus', default=1, type=int, help="Amount of GPU availables.")
    parser.add_argument('--num_workers', default=0, type=int, help="CPU cores for multi-processing.")
    parser.add_argument('--epochs', default=200000, type=int, help="Epochs to run.")
    ##### Return namespace.
    args = parser.parse_args(sys.argv[1:])

    ##### Get training and testing folder and create data loaders.
    training_folder = datasets.ImageFolder(args.training_images_folder,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.RandomResizedCrop(args.patch_dimension)
                                            ]))
    # TODO: Create testing images folder
    testing_folder = datasets.ImageFolder(args.testing_images_folder,
                                            transform=transforms.Compose([
                                                transforms.ToTensor()
                                            ]))
    train_dl = DataLoader(training_folder, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)
    test_dl = DataLoader(testing_folder, num_workers=args.num_workers)

    ##### Instantiate and run model
    vae = VAE()
    logger = TensorBoardLogger(args.tsb_folder, name=args.model_logs_folder)
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.epochs, progress_bar_refresh_rate=100, logger=logger)
    trainer.fit(vae, train_dl, test_dl)

    print("Training Finished")