#!/usr/bin/env python
# coding: utf-8

# # Training distribuito

# In[1]:


import argparse #inserisco il valore dei miei parametri/variabili da riga di comando
import inspect #introspezione su funzioni/moduli/classi
import logging #logging standard Python
import math
import os #per gestire path e variabili d'ambiente
import shutil #operazioni su file/directory (copia,sposta,rimuovi)
from datetime import timedelta #rappresenta intervalli temporali (per stabilire un tempo limite ai processi per sincronizzarsi)
from pathlib import Path #path object-oriented, comodo per file-system
#import data_large
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import skew,kurtosis
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random
from scipy.interpolate import interp1d
import os
import socket, sys

# In[2]:


import accelerate #libreria Hugging Face che gestisce training distribuito
import datasets #libreria hugging face per gestire datasets su HF_repo??
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs #classi che servono per gestire training distribuito
from accelerate.logging import get_logger #logger di accelerator
from accelerate.utils import ProjectConfiguration #gestisce configurazione progetto(cartelle, checkpoints etc.)
#from datasets import load_dataset #per caricare dataset da HF_repo da locale 
#from huggingface_hub import create_repo, upload_folder #per creare o caricare folders su HF
from packaging import version #per confrontare versioni esistenti con quelle richieste
from torchvision import transforms #trasformazioni in preprocessing
from tqdm.auto import tqdm #barra di progresso
from torch.utils.tensorboard import SummaryWriter #libreria per permettere ad accelerator di loggare figure plt su tensorboard
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn.init as init
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau



# In[3]:


from torch.utils.data import Dataset


# In[4]:




# In[5]:


#QUESTO PROBABILMENTE MI SERVE QUANDO VOGLIO CONDIZIONARE IL MIO MODELLO CON CLASSI/LABELS

#definisco funzione che, dato un vettore 1D di parametri (es timesteps, alpha_cumprod, sqrt_alpha_cumprod
#crea un array di lunghezza pari a batch_size con i valori di arr corrispondente ai timesteps estratti
#e ne fa broadcasting aggiungendo dimensioni pari a batch di immagini e espandendolo per applicare
#lo stesso scalare per pixel/canale

#funzione per trasformare un vettore 1D in un tensore broadcastabile per batch


# In[6]:


#scrivo funzione per assegnare tutti i parametri che vengono assegnati in riga di comando alle rispettive variabili
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="MLP_output_dir",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(1216,640),
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_step",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=int,
        default=1,
        help="Value of guidance scale that set how strong the model follow the conditions",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--checkpoints_step",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")




    return args


# In[7]:

#devo condizionare il modello interpretando le y+ non come classi ma come continui e quindi lo devo condizionare con embedding continuo
class Cfd_mlp(pl.LightningModule):
    def __init__(self, batch_size, learning_rate, node_per_layer,freq_interp):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        #self.freq_interp = freq_interp
        #self.node_per_layer = node_per_layer
        layer = []
        for i in range(int(len(node_per_layer)-1)):
            layer.append(nn.Linear(node_per_layer[i], node_per_layer[i+1]))
            #layer.append(nn.BatchNorm1d(node_per_layer[i+1]))
            layer.append(nn.LeakyReLU())
           # layer.append(nn.Dropout(p=0.05))
        layer.append(nn.Linear(node_per_layer[-1],1))
        self.mlp = nn.Sequential(*layer)  #unpack the layer
#           nn.Linear(6, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 2048),
#           nn.ReLU(),
#           nn.Linear(2048, 1),
#       )

        #apply He initialization (works well with ReLU)
        self._init_weights()

    def forward(self, x):
        output = self.mlp(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
       # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-4, max_lr=5e-3, step_size_up=2000, mode='triangular',cycle_momentum=False)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15),
            'monitor': 'val_loss'
        }
       # return {'optimizer': optimizer, 'lr_scheduler': scheduler}
       # scheduler = torch.optim.lr_scheduler.OneCycleLR(
       #          optimizer,
       #          max_lr = 1e-3,
       #          steps_per_epoch = 2844,
       #          epochs = 500,
       #          pct_start = 0.25,
       #          anneal_strategy='cos',
       #          final_div_factor=1e4
       #          )

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _init_weights(self):
        #Apply He initialization to each layer of my model
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
    
    def huber_loss(y_hat, y, delta=1.0):
        error = y - y_hat
        abs_error = torch.abs(error)
        quadratic = torch.minimum(abs_error, torch.tensor(delta))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + delta * linear
        return loss.mean()

    def loss_weights(self,y):
        y = y.unsqueeze(1)
        freqs = torch.from_numpy(freq_interp(y.detach().cpu().numpy())).to(y.device)
        weight = 1.0 / (freqs + 1e-10)
        return weight

    def quantile_loss(self, y, y_hat, tau):
        y = y.unsqueeze(1)
        errore = y - y_hat
        loss_function = torch.mean(torch.maximum(tau * errore, (tau - 1) * errore))
        return loss_function

    def log_cosh_loss(self, y, y_hat):
        y = y.unsqueeze(1)
        arg = torch.mean(torch.cosh(y_hat - y))
        loss_function = torch.mean(torch.log(arg))
        return loss_function

    def MSE_loss(self, y, y_hat):
        y=y.unsqueeze(1)
        loss_function = F.mse_loss(y_hat, y)                             #cambio un attimo mse con mae
        return loss_function

    def log_loss(self, y, y_hat):
        y = y.unsqueeze(1)
        loss =  torch.log(1 + torch.abs(y_hat - y))
        loss = torch.mean(loss)
        return loss

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.mlp(x)
        loss = self.MSE_loss(y, y_hat) #* self.loss_weights(y)

      #  weights = self.loss_weights(y)
      #  mse = (y_hat - y) ** 2
      #  loss = (weights * mse).sum() / weights.sum()
       # loss = huber_loss(y_hat,y,delta=1.0)
       # loss_fn = torch.nn.HuberLoss(delta=1.0)
       # loss = loss_fn(y_hat, y)


       # weight = self.loss_weights(y)
       # loss_01 = self.quantile_loss(y, y_hat, 0.1)
       # loss_05 = self.quantile_loss(y, y_hat, 0.5)
       # loss_09 = self.quantile_loss(y, y_hat, 0.9)
      #  loss = torch.mean(loss)
       # loss = loss * weight
       # loss = torch.mean(loss)
       # loss = self.log_cosh_loss(y, y_hat)
       # loss_01 = self.loss_function(y, y_hat, 0.1)
       # loss_05 = self.loss_function(y, y_hat, 0.5)
       # loss_09 = self.loss_function(y, y_hat, 0.9)
       # loss = loss_01 + loss_05 + loss_09
        self.log('train_loss', loss, prog_bar = True, on_epoch = True)
        prog_bar=True
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.mlp(x)
        loss = self.MSE_loss(y, y_hat)

       # loss = huber_loss(y_hat,y,delta=1.0)
       # loss_fn = torch.nn.HuberLoss(delta=1.0)
       # loss = loss_fn(y_hat, y)

       # weights = self.loss_weights(y)
       # mse = (y_hat - y) ** 2
       # loss = (weights * mse).sum() / weights.sum()


       # weight = self.loss_weights(y)
       # loss_01 = self.quantile_loss(y, y_hat, 0.1)
       # loss_05 = self.quantile_loss(y, y_hat, 0.5)
       # loss_09 = self.quantile_loss(y, y_hat, 0.9)
       # loss = torch.mean(loss)
       # loss = loss * weight
       # loss = torch.mean(loss)
        loss = self.log_cosh_loss(y, y_hat)
       # loss_01 = self.loss_function(y, y_hat, 0.1)
       # loss_05 = self.loss_function(y, y_hat, 0.5)
       # loss_09 = self.loss_function(y, y_hat, 0.9)
       # loss = loss_01 + loss_05 + loss_09
        self.log('val_loss', loss, prog_bar=True, on_epoch = True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.mlp(x)
        loss = self.MSE_loss(y, y_hat) #* self.loss_weights(y)
       # weight = self.loss_weights(y)
       # loss_01 = self.quantile_loss(y, y_hat, 0.1)
       # loss_05 = self.quantile_loss(y, y_hat, 0.5)
       # loss_09 = self.quantile_loss(y, y_hat, 0.9)
     #   loss = torch.mean(loss)

    def train_dataloader(self):
        return DataLoader(train_dataset,
                          batch_size=self.batch_size,
                          num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(test_dataset,
                          batch_size=self.batch_size,
                          num_workers=num_workers)

    #def per loggare metriche in TensorBoard
   # def validation_epoch_end(self, outputs):
   #     y_true = torch.cat([o['y_true'] for o in outputs]).cpu().numpy()
   #     y_pred = torch.cat([o['y_pred'] for o in outputs]).cpu().numpy()
   #
   #     mae = mean_absolute_error(y_true, y_pred)
   #     rmse = mean_squared_error(y_true, y_pred, squared=False)
   #     r2 = r2_score(y_true, y_pred)

   #     self.log("val/mae", mae)
   #     self.log("val/rmse", rmse)
   #     self.log("val/r2", r2)
   #
   #     self.y_true_epoch = y_true
   #     self.y_pred_epoch = y_pred
# # 03. Train


# In[8]:


#creo la classe per salvare il mio dataset da poter poi utilizzare per fittare la reteù Unet
#dataset salvato com un grande dict nel formato .npz

# In[9]:


#creo una funzione che mi crei il Dataset idoneo per fittare la mia unet facendo le opportune trasformazioni sulle mie slice
def create_dataset(path_outputdir, name):
    Retau_mean = 1.02003270E+003
    utau_mean  = 4.97576926E-002
    deltav = 0.5 / Retau_mean
    mu = deltav * utau_mean
    y_wall = 0.0001442222
    y60 = 0.02945271
    y120 = 0.05834579
    #U_bulk_plus = 21.1 #valore tipico per canali lisci a retau1000 mi serve solo per adimensionalizzare
    #Re_bulk = U_bulk_plus * 2 *Retau
    Re_bulk = 41000
    u_bulk = Re_bulk * mu / (2*0.5)
    batch_size = 128
    num_workers = 0
    #carichiamo i dizionari contenenti le slice
    uwall = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/slice_u_wall_less_{}.npz'.format(name)))
    u60 = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/slice_u60_less_{}.npz'.format(name)))
    v60 = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/slice_v60_less_{}.npz'.format(name)))
    w60 = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/slice_w60_less_{}.npz'.format(name)))
    u120 = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/slice_u120_less_{}.npz'.format(name)))
    v120 = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/slice_v120_less_{}.npz'.format(name)))
    w120 = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/slice_w120_less_{}.npz'.format(name)))
    #delta_u = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/delta_u_less_{}.npz'.format(name)))
    #delta_v = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/delta_v_less_{}.npz'.format(name)))
    #delta_w = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/delta_w_less_{}.npz'.format(name)))

    #adimensionale 
    tauwall = {}
    for key in uwall.keys():
        campo = uwall[key]
        campo = campo * mu /y_wall
        campo = campo / (u_bulk**2)
        tauwall[key] = campo
    
    #normalizzo

    delta_u = {}
    delta_v = {}
    delta_w = {}
    for (ku60, kv60,kw60,ku120,kv120,kw120) in zip(u60.keys(),v60.keys(),w60.keys(),u120.keys(),v120.keys(),w120.keys()):
        u60[ku60] = u60[ku60] * y60 / mu
        v60[kv60] = v60[kv60] * y60 / mu
        w60[kw60] = w60[kw60] * y60 / mu
        u120[ku120] = u120[ku120] * y120 / mu
        v120[kv120] = v120[kv120] * y120 / mu
        w120[kw120] = w120[kw120] * y120 / mu
        delta_u[ku120] = u120[ku120] - u60[ku60]
        delta_v[kv120] = v120[kv120] - v60[kv60]
        delta_w[kw120] = w120[kw120] - w60[kw60]
    

    name_fields=['tau_wall','u60','v60','w60','u120','v120','w120','deltau','deltav','deltaw']
    y_list=[60,120]
    #name_fields=['wall', 10, 20, 30,50,70,90,110,130,150,170]
    #y_list=[10, 20, 30,50,70,90,110,130,150,170]

    # Flatten e scaling
    from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
    def scale_dict(d_large, name_field):
        flat = np.concatenate([v.ravel() for v in d_large.values()])
        if name_field == 'tau_wall':
            scaler = PowerTransformer(method='yeo-johnson').fit(flat.reshape(-1, 1))
        elif name_field == 'deltau':
            scaler = RobustScaler().fit(flat.reshape(-1,1))
        else:
            scaler = StandardScaler().fit(flat.reshape(-1,1))

        joblib.dump(scaler, 'scaler/{}_scaler.pkl'.format(name_field))
        scaled = {k: scaler.transform(v.reshape(-1,1)).reshape(v.shape).astype(np.float32)
                  for k,v in d_large.items()}
        return scaled


    def scale_test_dataset(d_large, name_field):
        scaler = joblib.load('scaler/{}_scaler.pkl'.format(name_field))
        scaled = {k: scaler.transform(v.reshape(-1,1)).reshape(v.shape).astype(np.float32)
                for k,v in d_large.items()}
        return scaled


    if name == 'train':
        tauwall_scaled = scale_dict(tauwall, name_fields[0])
        u60_scaled  = scale_dict(u60, name_field = name_fields[1])
        v60_scaled  = scale_dict(v60, name_field = name_fields[2])
        w60_scaled  = scale_dict(w60, name_field = name_fields[3])
        u120_scaled = scale_dict(u120, name_field = name_fields[4])
        v120_scaled = scale_dict(v120, name_field = name_fields[5])
        w120_scaled = scale_dict(w120, name_field = name_fields[6])
        delta_u_scaled = scale_dict(delta_u, name_field = name_fields[7])
        delta_v_scaled = scale_dict(delta_v, name_field = name_fields[8])
        delta_w_scaled = scale_dict(delta_w, name_field = name_fields[9])
    #duplico i values di uwall_scaled per avere corrispondenza 1 a 1 tra condition e target
        
        
    elif name == 'test':
         
        tauwall_scaled = scale_test_dataset(tauwall, name_fields[0])
        u60_scaled     = scale_test_dataset(u60, name_field = name_fields[1])
        v60_scaled     = scale_test_dataset(v60, name_field = name_fields[2])
        w60_scaled     = scale_test_dataset(w60, name_field = name_fields[3])
        u120_scaled    = scale_test_dataset(u120, name_field = name_fields[4])
        v120_scaled    = scale_test_dataset(v120, name_field = name_fields[5])
        w120_scaled    = scale_test_dataset(w120, name_field = name_fields[6])
        delta_u_scaled = scale_test_dataset(delta_u, name_field = name_fields[7])
        delta_v_scaled = scale_test_dataset(delta_v, name_field = name_fields[8])
        delta_w_scaled = scale_test_dataset(delta_w, name_field = name_fields[9])
    else:
         
        tauwall_scaled = scale_test_dataset(tauwall, name_fields[0])
        u60_scaled     = scale_test_dataset(u60, name_field = name_fields[1])
        v60_scaled     = scale_test_dataset(v60, name_field = name_fields[2])
        w60_scaled     = scale_test_dataset(w60, name_field = name_fields[3])
        u120_scaled    = scale_test_dataset(u120, name_field = name_fields[4])
        v120_scaled    = scale_test_dataset(v120, name_field = name_fields[5])
        w120_scaled    = scale_test_dataset(w120, name_field = name_fields[6])
        delta_u_scaled = scale_test_dataset(delta_u, name_field = name_fields[7])
        delta_v_scaled = scale_test_dataset(delta_v, name_field = name_fields[8])
        delta_w_scaled = scale_test_dataset(delta_w, name_field = name_fields[9])
    #duplico i values di uwall_scaled per avere corrispondenza 1 a 1 tra condition e target

    np.savez('dataset_dict/tauwall_scaled_{}.npz'.format(name),**tauwall_scaled) 
    np.savez('dataset_dict/u60_scaled_{}.npz'.format(name),**u60_scaled) 
    np.savez('dataset_dict/v60_scaled_{}.npz'.format(name),**v60_scaled) 
    np.savez('dataset_dict/w60_scaled_{}.npz'.format(name),**w60_scaled) 
    np.savez('dataset_dict/u120_scaled_{}.npz'.format(name),**u120_scaled) 
    np.savez('dataset_dict/v120_scaled_{}.npz'.format(name),**v120_scaled) 
    np.savez('dataset_dict/w120_scaled_{}.npz'.format(name),**w120_scaled)
    np.savez('dataset_dict/delta_u_scaled_{}.npz'.format(name),**delta_u_scaled)
    np.savez('dataset_dict/delta_v_scaled_{}.npz'.format(name),**delta_v_scaled)
    np.savez('dataset_dict/delta_w_scaled_{}.npz'.format(name),**delta_w_scaled)

    tauwall_ravel = np.array([])
    u60_ravel = np.array([])
    v60_ravel = np.array([])
    w60_ravel = np.array([])
    u120_ravel = np.array([])
    v120_ravel = np.array([])
    w120_ravel = np.array([])
    delta_u_ravel = np.array([])
    delta_v_ravel = np.array([])
    delta_w_ravel = np.array([])

    for (kw,ku60,kv60,kw60,ku120,kv120,kw120) in zip(uwall.keys(), u60.keys(), v60.keys(),w60.keys(),u120.keys(),v120.keys(),w120.keys()):
        tauwall_ravel = np.append(tauwall_ravel, tauwall_scaled[kw].ravel())
        u60_ravel = np.append(u60_ravel, u60_scaled[ku60].ravel())
        v60_ravel = np.append(v60_ravel, v60_scaled[kv60].ravel())
        w60_ravel = np.append(w60_ravel, w60_scaled[kw60].ravel())
        u120_ravel = np.append(u120_ravel, u120_scaled[ku120].ravel())
        v120_ravel = np.append(v120_ravel, v120_scaled[kv120].ravel())
        w120_ravel = np.append(w120_ravel, w120_scaled[kw120].ravel())
        delta_u_ravel = np.append(delta_u_ravel, delta_u_scaled[ku120].ravel())
        delta_v_ravel = np.append(delta_v_ravel, delta_v_scaled[kv120].ravel())
        delta_w_ravel = np.append(delta_w_ravel, delta_w_scaled[kw120].ravel())

    y_60 = np.ones(tauwall_ravel.shape[0]) * y60 / 0.5 
    y_120 = np.ones(tauwall_ravel.shape[0]) * y120 / 0.5
    X = np.zeros((tauwall_ravel.shape[0], 9))
    X[:,0] = u60_ravel
    X[:,1] = w60_ravel
    X[:,2] = v60_ravel
    X[:,3] = u120_ravel
    X[:,4] = w120_ravel
    X[:,5] = v120_ravel
    X[:,6] = delta_u_ravel
    X[:,7] = delta_w_ravel
    X[:,8] = delta_v_ravel
    
    np.save('{}/features_{}.npy'.format(path_outputdir,name), X)
    np.save('{}/features_{}.npy'.format(path_outputdir,name), tauwall_ravel)
    X = torch.tensor(X, dtype=torch.float32)
    target = torch.tensor(tauwall_ravel, dtype=torch.float32)

    dataset = TensorDataset(X, target)

    if name == 'train':
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    print('u_allmatch e labels salvati in {}'.format(str(Path(path_outputdir))))
    return dataloader


# In[10]:


#scrivo funzione main (input gli args che sarebbero i valori delle variabili settati da riga di comando)
def main(args):
    #costruisco percorso della cartella dei log come sottocartella di output_dir
    cartella_log = os.path.join(Path(args.output_dir) / args.logging_dir) #Path() per forzare sottocartella

    #creo oggetto che contiene metadati che indicano dove salvare file, log etc.
    configurazione_accelerate_project = ProjectConfiguration(project_dir=args.output_dir, logging_dir = cartella_log)

    #if accelerator.is_main_process:
        #if args.create_dataset:   # flag che decidi tu
    train_dataloader = create_dataset(args.train_data_dir, name='train')   # funzione che salva u_allmatch in npz

    valid_dataloader = create_dataset(args.train_data_dir, name='valid')   # funzione che salva u_allmatch in npz
    test_dataloader = create_dataset(args.train_data_dir, name='test')   # funzione che salva u_allmatch in npz

    batch_size = 128
    num_workers = 0
    model = Cfd_mlp(batch_size, 1e-4,[9, 512,256,128], freq_interp = freq_interp)


    # In[10]:



    # In[11]:


    # trainer
    logger = TensorBoardLogger('{}'.format(cartella_log), name='tauwall_prediction', version='tauwall_512_256_128_MSE_weighted') #logger per monitoraggio allenamento
    checkpoint = ModelCheckpoint(
        dirpath="checkpoints_512_256_128_OncycleLR_MSE_weighted",
        filename="{epoch:02d}-{val_loss:.7f}_",
        monitor='val_loss',
        save_top_k=3,
        mode='min') #permette di salvare lo stato del modello e riprendere l'allenamento i  seguito
    earlystop = EarlyStopping(monitor='val_loss', patience=40, mode='min')
    trainer = pl.Trainer(accelerator="gpu",
                         devices=4,
                         strategy="ddp",
                         max_epochs=500,
                         logger=logger,
                         deterministic=True,
                         callbacks=[checkpoint,earlystop],
                         log_every_n_steps=1,
                         #flush_logs_every_n_steps=2,
                         #auto_lr_find=True,
                         # overfit_batches=10,
                        )
    trainer.fit(model, train_dataloader, valid_dataloader)

y_train = np.load('dataset_dict/tauwall_scaled_train_ravel.npy')
hist, bin_edges = np.histogram(y_train, bins=100, density=True)
bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2
from scipy.interpolate import interp1d
freq_interp = interp1d(bin_center, hist, bounds_error=False, fill_value='extrapolate')


if __name__ == "__main__":
    
    #os.environ["ACCELERATE_DISABLE_RNG_TRACKING"] = "1"

#per debug, guardo come pytorch vede le GPUs e i rank
   # print("=== PROCESS DIAGNOSTIC START ===")
   # print("HOST:", socket.gethostname())
   # print("ENV WORLD_SIZE:", os.environ.get("WORLD_SIZE"))
   # print("ENV RANK:", os.environ.get("RANK"))
   # print("ENV LOCAL_RANK:", os.environ.get("LOCAL_RANK"))
   # print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
   # print("torch.cuda.is_available():", torch.cuda.is_available())
   # print("torch.cuda.device_count():", torch.cuda.device_count())
   # try:
   #     local_rank = int(os.environ.get("LOCAL_RANK", 0))
   # except:
   #     local_rank = 0
   # print("LOCAL_RANK (int):", local_rank)
   # # prova a impostare device esplicito (vedi parte C)
   # print("PID:", os.getpid())
   # sys.stdout.flush()
   # print("=== PROCESS DIAGNOSTIC END ===")
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    args = parse_args()

    main(args)


# In[ ]:




