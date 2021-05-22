# Final Project - Image Transform with Variational Autoencoder.
  
### Author:
- Name: Matheus Costa de Oliveira
- Registration: 17/0019039
### Course Information.
- Subject: Fundamentals of Signal Compression
- Department of Electrical Engineering
- University of Brasilia

___

## Code Execution

The code was developed so that training and evaluation could be carried out together. If you just want to evaluate the performance of a checkpoint, just use a debugger to operate only on the *validation_step* method, which is invoked before the *training_step*.

The command line below generalizes how the script should be executed.

```bash
<python-version> vae.py --training_images_folder <path to folder> --testing_images_folder <path to folder> --restore_checkpoint <path to checkpoint> --patch_dimension <patch size for training> --tsb_folder <path to tensorboard logs> --model_logs_folder <folder for model logs> --batchsize <training batch size> --gpus <number of gpus to use> --num_workers <number of cpus cores to use> --epochs <training epochs>
```

- **training_images_folder** and **testing_images_folder**: Mandatory arguments. The provided directories must contain sub-folders with the images. It is not allowed to allocate the images directly in the given path.

- **restore_checkpoint**: In case of continuing an interrupted training, path to the *cpkt* file.

- **patch_dimension**: Patch size used when cropping images for training. Default is 256.

- **tsb_folder** and **model_logs_folder**: Folder for saving logs and checkpoints from different versions of the model. In practice, the path used will be *tsb_folder/model_logs_folder*.

- **batchsize**: Default is 8.

- **gpus** and **num_workers**: The amount of GPUs and CPU cores to use, respectively. PyTorch Lightning handles this very efficiently.

- **epochs**: Number of epochs to run in the training. Default is 100.