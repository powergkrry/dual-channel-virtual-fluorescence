# Dual Channel Virtual Fluorescence

This is the code to train a U-Net inspired architecture to predict two channel fluorescence images from a multi-LED illuminated stack of TB bacteria images.
To use the code, run main.py with command line options like follows. The summary of the available options can be found in `config.py`. 
```
 python main.py --epochs 700 --polydecay --name exp_newdata_mae_700_poly --gpu 2 --loss mae --polydecay --lr_reduction_factor 1000 --lr_reduction_steps 25000
```
At the end of the run, a folder is
created to save validation images, a loss curve over epochs, validation metrics and the list of parameters provided which may look like this:
```
{
  "batch_size": 16,
  "val_batch_size": 8,
  "is_green": 0,
  "shuffle": true,
  "epochs": 100,
  "init_lr": 0.001,
  "final_activation": "swish",
  "lamda": 0.0001,
  "loss": "blur",
  "maxpool": false,
  "polydecay": false,
  "plateaudecay": true,
  "random_seed": 0,
  "gpu": "3",
  "n_sample": 21,
  "n_out_channels": 1,
  "name": "exp_newdata_swish_blur_l1_100_nomax_plat"
}
```
TODO: add descroption of loss functions and models.