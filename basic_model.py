import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder

import torch 
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class VAE(pl.LightningModule):

    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=32):        
        ##### Run builder method defined in parent class.
        super().__init__()
        self.save_hyperparameters()

        ##### Instantiate encoder and decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(latent_dim=latent_dim, input_height=input_height,
                                        first_conv=False, maxpool1=False)

        ##### Obtain distribution parameters from latent output.
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

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


    def training_step(self, batch, batch_idx):

        ########## Run batch through network
        ##### Get images from batch.
        x, _ = batch
        ##### Get encoder latent and generate learned parameters.
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        ##### Define learned multivariate distribution Q(z|x).
        std = torch.exp(log_var / 2)
        self.q = torch.distributions.Normal(mu, std)
        ##### Sample latent from distribution.
        z = self.q.rsample()
        ##### Reconstruct Image from sampled latent.
        x_hat = vae.decoder(z)

        ########## Compute Losses
        ##### Compute Reconstruction Loss
        recon_lkh = self.gaussian_likelihood(x_hat, x)
        ##### Compute KL divergence from Monte Carlo Approach
        kl = self.kl_divergence(z)
        ##### Combine losses and define ELBO loss.
        elbo = (kl - recon_lkh)
        elbo = elbo.mean()
        ##### Return batch information
        batch_dictionary = {
            'loss': elbo,
            'kl': kl.mean(),
            'recon_lkh': recon_lkh.mean(),
            'original_img': x[0],
            'recon_img': x_hat[0]
        }

        return batch_dictionary


    def training_epoch_end(self, outputs):
        ##### Compute epoch means.
        mean_losses = torch.mean(torch.as_tensor([[batch['loss'], batch['kl'], batch['recon_lkh']] for batch in outputs]), 0)
        ##### Get images
        random_index = np.random.choice(len(outputs))
        original_image = outputs[random_index]['original_img']
        recon_image = outputs[random_index]['recon_img']
        ##### Include scalars in tensorboard
        list(map(lambda metric, value: self.logger.experiment.add_scalar(metric, value, self.current_epoch), 
                                            ['ELBO_Loss', 'KL_Loss', 'Recon_Likelihood'], mean_losses))
        ##### Include images
        self.logger.experiment.add_image("Original_Image", original_image, self.current_epoch)
        self.logger.experiment.add_image("Reconstructed_Image", recon_image, self.current_epoch)



if __name__ == "__main__":
    ##### Define parser
    parser = argparse.ArgumentParser(description="Receives arguments for encoding.")
    ##### Define arguments.
    parser.add_argument('--training_images_folder', required=True, help='Path to folder with subfolders.')
    parser.add_argument('--patch_dimension', default=256, type=int, help="Patch dimension used in training stage.")
    parser.add_argument('--tsb_folder', default="tensorboard_logs", help="Folder to save Tensorboard logs.")
    parser.add_argument('--model_logs_folder', default="vae_with_rn18", help='Folder to save logs of current running.')
    parser.add_argument('--batchsize', default=8, type=int, help='Batch size.')
    parser.add_argument('--gpus', default=1, type=int, help="Amount of GPU availables.")
    parser.add_argument('--epochs', default=200000, type=int, help="Epochs to run.")
    ##### Return namespace.
    args = parser.parse_args(sys.argv[1:])

    ##### Get training folder and create data bunch.
    training_folder = datasets.ImageFolder(args.training_images_folder,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.RandomResizedCrop(args.patch_dimension)
                                            ]))
    dataset_loader = DataLoader(training_folder, batch_size=args.batchsize, shuffle=True)

    ##### Instantiate and run model
    vae = VAE(input_height=args.patch_dimension)
    logger = TensorBoardLogger(args.tsb_folder, name=args.model_logs_folder)
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.epochs, progress_bar_refresh_rate=50, logger=logger)
    trainer.fit(vae, dataset_loader)

    print("Training Finished")