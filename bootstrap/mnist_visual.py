import os
import click
import traceback
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from .lib import utils
from .lib.options import Options
import argparse
import importlib
from tqdm import tqdm
import pandas as pd
import numpy as np


class LayerOutNet(nn.Module):
    def __init__(self, model, layer):
        super(LayerOutNet, self).__init__()
        self.model = model
        self.layer = layer

        def save_output(module, input, output):
            self.buffer = output
        self.layer.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer



def gen_output_features(model, device, val_loader):
    # tq = tqdm(val_loader, desc='{} E{:03d}'.format('val ', epoch), ncols=100)
    model.eval()
    out_features = np.zeros((len(val_loader),model.layer.weight.shape[0]))
    out_target = np.zeros((len(val_loader,)))

    print("dataset len:", len(val_loader))
    outputs=[]
    targets=[]
    with torch.no_grad():
        i=j=0
        for batch_idx, item in enumerate(val_loader):
            data = item['data']
            target = item['class_id'].squeeze(1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            outputs.append(output.cpu())
            targets.append(target.cpu())
    print(len(outputs))
    out_features = torch.cat(outputs, dim=0)
    out_targets = torch.cat(targets, dim=0)
    print("out_features:",out_features.shape)
    print("out_targets:", out_targets.shape)
    return out_features.numpy(), out_targets.numpy()



def run(path_opts=None):
    # first call to Options() load the options yaml file from --path_opts command line argument if path_opts=None
    options=Options(path_opts)
    # initialiaze seeds to be able to reproduce experiment on reload
    utils.set_random_seed(options['misc']['seed'])
    print(options)

    # dataset
    dataset_module = importlib.import_module(options['dataset']['import'])
    val_dataset = dataset_module.dataset(options, split='val')
    val_loader = val_dataset.make_batch_loader()

    # model
    ## network
    net_module = importlib.import_module(options['model']['network']['import'])
    net = net_module.Net(options)
    device = torch.device("cuda")
    model = net.to(device)


    ## resume init
    if options['exp'].get('resume', False):
        model_path = os.path.join(options['exp']['dir'], 'ckpt_best_accuracy_top1_model.pth')
        model_dict = torch.load(model_path)
        model.load_state_dict(model_dict)



    layeroutmodel = LayerOutNet(model,model.fc2)
    out_features, out_targets = gen_output_features(model=layeroutmodel, device=device, val_loader=val_loader)
    out_features_path = os.path.join(options['exp']['dir'], 'fc2.npy')
    np.save(out_features_path,out_features)
    out_targets_path = os.path.join(options['exp']['dir'], 'targets.npy')
    np.save(out_targets_path, out_targets)





def get_parser():
    optfile_parser = argparse.ArgumentParser(add_help=False)
    optfile_parser.add_argument('-o', '--path_opts', type=str, required=True)
    #optfile_parser.add_argument('-lr', '--optimizer.lr', type=float, required=False)
    return optfile_parser


if __name__ == '__main__':
    parser = get_parser()
    args_dict = vars(parser.parse_args())
    path_yaml = args_dict['path_opts']
    run(path_yaml)




