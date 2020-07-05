import argparse

from data import HistoDataNorm
from resnet import BasicResNet
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score

import random
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import util
from tqdm import tqdm

import torch
import torch.utils.data as data_utils
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional

import os

import numpy as np
import shutil


def main(args, seed):
    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    # args.batch_size *= max(1, len(args.gpu_ids))
    # print(args.batch_size)


    experiment_folder = args.experiment_folder + str(seed) + '/'
    print(experiment_folder)
    
    if args.full:
        domain_list_train=[args.missing]
    else:
        domain_list_train = []
        domain_list_train.append(args.missing + '_m')
    print(len(domain_list_train))

    testset = HistoDataNorm('missing_' +args.type, domain_list=domain_list_train, augmentation=False)

    num_classes = 2

    print(len(testset), "testing on")
      
    testloader = data_utils.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=args.num_workers,  worker_init_fn=np.random.seed(seed) )

    print("num_classes", num_classes)
    # Model
    print('Building model..')
    net = BasicResNet(num_classes=num_classes, network='resnet50', pretrained=True)
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        #cudnn.benchmark = False

    print('Loading best model')
    # assert os.path.isdir('HSM/ckpts'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(experiment_folder + args.mode  + "_" + str(args.temperature) + "_" + 'ckpts/best.pth.tar')
    # if args.mode=='label':
    #
    # else:
    #     checkpoint = torch.load('domain_classification/ckpts/best.pth.tar')
    net.load_state_dict(checkpoint['net'])

    loss_fn = nn.CrossEntropyLoss()
    accuracy_from_best_model, auc = test(0, net, testloader, device, loss_fn, args.batch_size, args.mode,
                                           experiment_folder=experiment_folder, full=args.full)
    if args.full:
        record = "AUC " + str(auc) + " Test accuracy " + str(accuracy_from_best_model)
        print(record)
        file = open(experiment_folder + args.mode + '_AUC_' + str(args.temperature) + "_" +  "classification_test_results.txt", "w+")
        file.write(record)
        file.close()
    else:
        record = " Test accuracy " + str(accuracy_from_best_model)
    
        print(record)
        file = open(experiment_folder + args.mode + 'NOT'+args.type[:-1]+ "_" + str(args.temperature) + "_" +  "classification_test_results.txt", "w+")
        file.write(record)
        file.close()


@torch.no_grad()
def test(epoch, net, testloader, device, loss_fn, num_samples, mode, experiment_folder, full):
    net.eval()
    loss_meter = util.AverageMeter()
    correct = 0
    total_0 = 0
    total_1 = 0
    correct_0 = 0
    correct_1 = 0
    auc = None

    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, y, d, yd in testloader:
            x, y, d, yd = x.to(device), y.to(device), d.to(device), yd.to(device)
            z = net(x)

            loss = loss_fn(z, y.argmax(dim=1))
            loss_meter.update(loss.item(), x.size(0))
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))
            values, pred = torch.max(z, 1)


            correct += pred.eq(y.argmax(dim=1)).sum().item()
            if full:
                auc = roc_auc_score(y.argmax(dim=1).cpu().numpy(), pred.cpu().numpy(), labels=[0, 1])
            for img in range(y.size(0)):
                if pred.eq(y.argmax(dim=1))[img].item():
                    if y.argmax(dim=1)[img].item() == 0:
                        correct_0 += 1
                    else:
                        correct_1 += 1

    print("total", len(testloader.dataset))
    print(correct)
    accuracy_from_best_model = correct * 100. / len(testloader.dataset)
    print(accuracy_from_best_model)

    print(correct_0)
    print(correct_1)
    total_per_class = len(testloader.dataset) / 2
    print("Uninfected accuracy", correct_0 * 100. / total_per_class)
    print("Parasitized accuracy", correct_1 * 100. / total_per_class)
    print("testing on", len(testloader.dataset))
    return accuracy_from_best_model, auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conditional Glow on Malaria Dataset')


    def str2bool(s):
        return s.lower().startswith('t')


    parser.add_argument('--batch_size', default=2000, type=int, help='Batch size per GPU')
    parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
    parser.add_argument('--num_channels', '-C', default=128, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=8, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--seed', type=str, default="0,1,2", help='seed for reproducibility')
    parser.add_argument('--warm_up', default=500000, type=int, help='Number of steps for lr warm-up')
    parser.add_argument('--mode', default="original", choices=['original', 'w_generations', 'no_balance', '100_generations', 'filtered'])
    parser.add_argument('--experiment_folder', default="new")
    parser.add_argument('--missing', type=str)
    parser.add_argument('--type', type=str)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--full', action='store_true')

    args = parser.parse_args()
    seed_list = [int(item) for item in args.seed.split(',')]
    print(seed_list)
    for seed in seed_list:
    
        print("seed", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True    
        
        best_loss = float('inf')
        global_step = 0

        main(args, seed)