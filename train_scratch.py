# train a baseline model from scratch

import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from tqdm import tqdm
import argparse
import os
import logging
import numpy as np

from utils.utils import RunningAverage, set_logger, Params
from model import *
from data_loader import fetch_dataloader
from cam_Vis import *


# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--id', default='XXXX', help='name')
parser.add_argument('--save_path', default='experiments/CIFAR10/baseline/resnet18', type=str)
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
# Model parameters
parser.add_argument('--model_name', default='resnet18', type=str, help='model name')
parser.add_argument('--resume', default='', metavar='Name/path', help='path to latest checkpoint (default: none)')
# Training parameters
parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
parser.add_argument('--schedule', default=[80, 120], type=int, nargs='+', help='schedule')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--num_epochs', default=160, type=int, help='number of epochs')
parser.add_argument('--num_workers', default=15, type=int, help='number of workers')
parser.add_argument('--augmentation', default=1, type=int, help='augmentation')
parser.add_argument('--seed', default=0, type=int, help='seed')
# Dataset parameters
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')

parser.add_argument('--use_posion_data', action='store_true', help='dataset mode')
parser.add_argument('--pData_path', default='XXXX', type=str, help='dataset')
args = parser.parse_args()


# ************************** random seed **************************


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device_ids = args.gpu_id
torch.cuda.set_device(device_ids[0])


# ************************** training function **************************
def train_epoch(model, optim, loss_fn, data_loader, args):
    model.train()
    loss_avg = RunningAverage()

    with tqdm(total=len(data_loader)) as t:  # Use tqdm for progress bar
        for i, (train_batch, labels_batch) in enumerate(data_loader):
            if args.cuda:
                train_batch = train_batch.cuda()        # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)

            # compute model output and loss
            output_batch = model(train_batch)           # logit without softmax
            loss = loss_fn(output_batch, labels_batch)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # update the average loss
            loss_avg.update(loss.item())

            # tqdm setting
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    return loss_avg()


def evaluate(model, loss_fn, data_loader, args):
    model.eval()
    # summary for current eval loop
    summ = []

    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in data_loader:
            if args.cuda:
                data_batch = data_batch.cuda()          # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)
            # # **************************please give the code for ploting nasty teacher logits distbution **************************
            # plot_logitsDistri(output_batch, labels_batch)
            # vis_featureMaps(t_model, train_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            # calculate accuracy
            output_batch = np.argmax(output_batch, axis=1)
            acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])

            summary_batch = {'acc': acc, 'loss': loss.item()}
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    return metrics_mean


def train_and_eval(model, optim, loss_fn, train_loader, dev_loader, args):
    best_val_acc = -1
    best_epo = -1
    lr = args.learning_rate

    for epoch in range(args.num_epochs):
        # LR schedule *****************
        lr = adjust_learning_rate(optim, epoch, lr, args)

        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        logging.info('Learning Rate {}'.format(lr))

        # ********************* one full pass over the training set *********************
        train_loss = train_epoch(model, optim, loss_fn, train_loader, args)
        # train_loss = -1
        
        logging.info("- Train loss : {:05.3f}".format(train_loss))

        # ********************* Evaluate for one epoch on validation set *********************
        val_metrics = evaluate(model, loss_fn, dev_loader, args)     # {'acc':acc, 'loss':loss}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
        logging.info("- Eval metrics : " + metrics_string)

        # save last epoch model
        save_name = os.path.join(args.save_path, 'last_model.tar')
        torch.save({
            'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim.state_dict()},
            save_name)

        # ********************* get the best validation accuracy *********************
        val_acc = val_metrics['acc']
        if val_acc >= best_val_acc:
            best_epo = epoch + 1
            best_val_acc = val_acc
            logging.info('- New best model ')
            # save best model
            save_name = os.path.join(args.save_path, 'best_model.tar')
            torch.save({
                'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim.state_dict()},
                save_name)

        logging.info('- So far best epoch: {}, best acc: {:05.3f}'.format(best_epo, best_val_acc))


def adjust_learning_rate(opt, epoch, lr, args):
    if epoch in args.schedule:
        lr = lr * args.gamma
        for param_group in opt.param_groups:
            param_group['lr'] = lr
    return lr


if __name__ == "__main__":
    # ************************** set log **************************
    set_logger(os.path.join(args.save_path, 'training.log'))

    # #################### Load the parameters from json file #####################################
    # json_path = os.path.join(args.save_path, 'params.json')
    # assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    # params = Params(json_path)

    args.cuda = torch.cuda.is_available() # use GPU if available
    # torch.set_default_dtype(torch.float64)

    for k, v in args.__dict__.items():
        logging.info('{}:{}'.format(k, v))

    # ########################################## Dataset ##########################################
    if args.use_posion_data:
        trainloader, devloader = fetch_dataloader('posion_data', args)
        logging.info('--use_posion_data!')
    else:
        trainloader, devloader = fetch_dataloader('clean_data', args)
        logging.info('use clean dataset!')
    # ############################################ Model ############################################
    if args.dataset == 'cifar10':
        num_class = 10
    elif args.dataset == 'cifar100':
        num_class = 100
    elif args.dataset == 'tiny_imagenet':
        num_class = 200
    else:
        num_class = 10

    logging.info('Number of class: ' + str(num_class))
    logging.info('Create Model --- ' + args.model_name)

    # ResNet 18 / 34 / 50 ****************************************
    if args.model_name == 'resnet18':
        model = ResNet18(num_class=num_class)
    elif args.model_name == 'resnet34':
        model = ResNet34(num_class=num_class)
    elif args.model_name == 'resnet50':
        model = ResNet50(num_class=num_class)

    # PreResNet(ResNet for CIFAR-10)  20/32/56/110 ***************
    elif args.model_name.startswith('preresnet20'):
        model = PreResNet(depth=20, num_classes=num_class)
    elif args.model_name.startswith('preresnet32'):
        model = PreResNet(depth=32, num_classes=num_class)
    elif args.model_name.startswith('preresnet44'):
        model = PreResNet(depth=44, num_classes=num_class)
    elif args.model_name.startswith('preresnet56'):
        model = PreResNet(depth=56, num_classes=num_class)
    elif args.model_name.startswith('preresnet110'):
        model = PreResNet(depth=110, num_classes=num_class)

    # DenseNet *********************************************
    elif args.model_name == 'densenet121':
        model = densenet121(num_class=num_class)
    elif args.model_name == 'densenet161':
        model = densenet161(num_class=num_class)
    elif args.model_name == 'densenet169':
        model = densenet169(num_class=num_class)

    # ResNeXt *********************************************
    elif args.model_name == 'resnext29':
        model = CifarResNeXt(cardinality=8, depth=29, num_classes=num_class)

    elif args.model_name == 'mobilenetv2':
        model = MobileNetV2(class_num=num_class)

    elif args.model_name == 'shufflenetv2':
        model = shufflenetv2(class_num=num_class)

    # Basic neural network ********************************
    elif args.model_name == 'net':
        model = Net(num_class, args)

    elif args.model_name == 'mlp':
        model = MLP(num_class=num_class)

    else:
        model = None
        print('Not support for model ' + str(args.model_name))
        exit()

    if args.cuda:
        model = model.cuda()

    if len(args.gpu_id) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    # checkpoint ********************************
    if args.resume:
        logging.info('- Load checkpoint model from {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logging.info('- Train from scratch ')

    # ############################### Optimizer ###############################
    if args.model_name == 'net' or args.model_name == 'mlp':
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        logging.info('Optimizer: Adam')
    else:
        optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
        logging.info('Optimizer: SGD')

    # ************************** LOSS **************************
    criterion = nn.CrossEntropyLoss()

    # ################################# train and evaluate #################################
    train_and_eval(model, optimizer, criterion, trainloader, devloader, args)


