import os
import click
import traceback
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F

from .lib import utils
from .lib.options import Options
import argparse
import importlib
from tqdm import tqdm
import pandas as pd


def init_experiment_directory(options):
    exp_dir = options['exp']['dir']
    resume = options['exp']['resume']
    # create the experiment directory
    if not os.path.isdir(exp_dir):
        os.system('mkdir -p ' + exp_dir)
    else:
        if resume is None:
            if click.confirm('Exp directory already exists in {}. Erase?'
                    .format(exp_dir, default=False)):
                os.system('rm -r ' + exp_dir)
                os.system('mkdir -p ' + exp_dir)
            else:
                os._exit(1)

    path_yaml = os.path.join(exp_dir, 'options.yaml')
    #  create the options.yaml file
    if not os.path.isfile(path_yaml):
        options.save(path_yaml)


def train( model, device, train_loader, optimizer, lossfunc, epoch):
    model.train()
    tq = tqdm(train_loader, desc='{} E{:03d}'.format('train', epoch), ncols=100)
    for batch_idx ,item in enumerate(tq):
        data = item['data']
        target = item['class_id'].squeeze(1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossfunc(output, target)
        loss.backward()
        optimizer.step()

        tq.set_postfix(loss='{:.4f}'.format(loss.item()), comp='{}'.format(batch_idx * len(data)))




def val(model, device, val_loader, lossfunc, epoch, topk=[1]):
    tq = tqdm(val_loader, desc='{} E{:03d}'.format('val ', epoch), ncols=100)
    model.eval()
    test_loss = 0
    correct = {k:0 for k in topk}
    maxk= max(topk)
    with torch.no_grad():
        for batch_idx, item in enumerate(tq):
            data = item['data']
            target = item['class_id'].squeeze(1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = lossfunc(output, target)
            test_loss += loss.item()
            _, pred = output.topk(maxk, 1, True, True)
            batch_correct = pred.eq(target.view(-1, 1).expand_as(pred))
            for k in topk:
                correct[k] += batch_correct[:,:k].sum().item()


    test_loss /= len(val_loader.dataset)
    result = {'epoch':epoch ,'loss':test_loss}
    for k in topk:
        keyname = "accuracy_top{}".format(k)
        result[keyname]=100. * correct[k] / len(val_loader.dataset)
        print('\nVal set: Average loss: {:.4f}, Top-{} Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, k, correct[k], len(val_loader.dataset),
            100. * correct[k] / len(val_loader.dataset)))

    return result



def run(path_opts=None):
    # first call to Options() load the options yaml file from --path_opts command line argument if path_opts=None
    options=Options(path_opts)
    # initialiaze seeds to be able to reproduce experiment on reload
    utils.set_random_seed(options['misc']['seed'])
    init_experiment_directory(options)

    print(options)

    # dataset
    dataset_module = importlib.import_module(options['dataset']['import'])
    train_dataset = dataset_module.dataset(options, split='train')
    train_loader = train_dataset.make_batch_loader()
    val_dataset = dataset_module.dataset(options, split='val')
    val_loader = val_dataset.make_batch_loader()

    # model
    ## network
    net_module = importlib.import_module(options['model']['network']['import'])
    net = net_module.Net(options)
    device = torch.device("cuda")
    model = net.to(device)



    ## criterion/loss_function
    if options['model']['criterion'].get('import', False):
        criterion_module = importlib.import_module(options['optimizer']['import'])
        criterion = criterion_module(options)
    elif options['model']['criterion']['name']=='nll':
        criterion=F.nll_loss
    elif options['model']['criterion']['name'] == 'cross_entropy':
        criterion = F.cross_entropy
    elif options['model']['criterion']['name'] == 'BCEWithLogitsLoss':
        criterion = F.binary_cross_entropy_with_logits

    ## metrics
    topk=options['model']['metric']['topk']



    # optimizer
    if options['optimizer'].get('import', False):
        optimizer_module = importlib.import_module(options['optimizer']['import'])
        optimizer = optimizer_module(options)
    elif options['optimizer']['name']=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=options['optimizer']['lr'], momentum=options['optimizer']['momentum'])
    elif options['optimizer']['name']=='adam':
        optimizer = optim.Adam(model.parameters(), lr=options['optimizer']['lr'])

    # train
    ## train init
    results = []
    start_epoch = 0
    best_accuracy_top1 = 0

    ## resume init
    if options['exp'].get('resume', False):
        model_path = os.path.join(options['exp']['dir'], 'ckpt_best_accuracy_top1_model.pth')
        engine_path = os.path.join(options['exp']['dir'], 'ckpt_best_accuracy_top1_engine.pth')
        optimizer_path = os.path.join(options['exp']['dir'], 'ckpt_best_accuracy_top1_optimizer.pth')
        model_dict = torch.load(model_path)
        model.load_state_dict(model_dict)
        optimizer_dict = torch.load(optimizer_path)
        optimizer.load_state_dict(optimizer_dict)
        engine_dict = torch.load(engine_path)
        start_epoch = engine_dict['epoch']
        best_accuracy_top1 = engine_dict['accuracy_top1']


    for epoch in range(start_epoch + 1, options['engine']['nb_epochs'] + 1):
        train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, lossfunc=criterion, epoch=epoch)
        result = val(model=model, device=device, val_loader=val_loader, lossfunc=criterion, epoch=epoch, topk=topk)
        results.append(result)


    ## save best checkpoints
        if result['accuracy_top1'] > best_accuracy_top1:
            best_accuracy_top1 = result['accuracy_top1']
            model_path = os.path.join(options['exp']['dir'], 'ckpt_best_accuracy_top1_model.pth')
            engine_path = os.path.join(options['exp']['dir'], 'ckpt_best_accuracy_top1_engine.pth')
            optimizer_path = os.path.join(options['exp']['dir'], 'ckpt_best_accuracy_top1_optimizer.pth')
            engine_dict = {'epoch':epoch, 'accuracy_top1':best_accuracy_top1}
            torch.save(model.state_dict(), model_path)
            torch.save(engine_dict, engine_path)
            torch.save(optimizer.state_dict(), optimizer_path)



    ## save evaluate history
    if start_epoch < options['engine']['nb_epochs']:
        result_path = os.path.join(options['exp']['dir'], 'result.csv')
        results_columns = list(results[0].keys())
        data=[[r[k] for k in results_columns] for r in results]
        result_df = pd.DataFrame(data,columns=results_columns)
        if options['exp'].get('resume', False):
            pre_result_df = pd.read_csv(result_path)
            result_df = pd.concat([pre_result_df,result_df], ignore_index=True)
        print(result_df)
        result_df.to_csv(result_path, index=False)
    else:
        result = val(model=model, device=device, val_loader=val_loader, lossfunc=criterion, epoch=start_epoch, topk=topk)




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




