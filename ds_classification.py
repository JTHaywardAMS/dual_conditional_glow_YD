import argparse

from data import HistoDataNorm
from resnet import BasicResNet

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
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

    experiment_folder = args.experiment_folder  + str(seed) + '/'
    os.makedirs(experiment_folder, exist_ok=True)
    print(experiment_folder)

    # Set random seeds
    
    if args.mode=='no_balance':
        print('no_balance')
        domain_list_train = ["C116P77ThinF", "C132P93ThinF", "C137P98ThinF", "C180P141NThinF", "C182P143NThinF", \
                             "C184P145ThinF", "C39P4thinF", 'C59P20thinF', "C68P29N", "C99P60ThinF"]                     
        domain_list_train.remove(args.missing)
        domain_list_train.append(args.missing +'_m')
        domain_list_train.sort()
        dataset = HistoDataNorm('missing_'+args.type, domain_list=domain_list_train, augmentation=True)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    
    else:
        domain_list_train_1 = ["C116P77ThinF", "C132P93ThinF", "C137P98ThinF", "C180P141NThinF", "C182P143NThinF", \
                                 "C184P145ThinF", "C39P4thinF", 'C59P20thinF', "C68P29N", "C99P60ThinF"]                     
        domain_list_train_1.remove(args.missing)
        domain_list_train_1.append(args.missing +'_empty')
        domain_list_train_1.sort()
        dataset_1 = HistoDataNorm('missing_'+args.type, domain_list=domain_list_train_1, augmentation=True)
        targets = dataset_1.train_labels
        train_idx, valid_idx= train_test_split(np.arange(len(targets)), test_size=0.2, random_state=seed, shuffle=True, stratify=targets)

        train_dataset_1 = torch.utils.data.Subset(dataset_1, train_idx)
        test_dataset_1 = torch.utils.data.Subset(dataset_1, valid_idx)
        #train_dataset_1, test_dataset_1 = torch.utils.data.random_split(dataset_1, [train_size_1, test_size_1])
            
        domain_list_train_2 = ["C116P77ThinF_empty", "C132P93ThinF_empty", "C137P98ThinF_empty", "C180P141NThinF_empty", "C182P143NThinF_empty", \
                                 "C184P145ThinF_empty", "C39P4thinF_empty", 'C59P20thinF_empty', "C68P29N_empty", "C99P60ThinF_empty"]   
        domain_list_train_2.remove(args.missing+'_empty')
        domain_list_train_2.append(args.missing +'_m')
        domain_list_train_2.sort()
            
        dataset_2 = HistoDataNorm('missing_'+args.type, domain_list=domain_list_train_2, augmentation=True)
        
        targets = dataset_2.train_labels
        train_idx, valid_idx= train_test_split(np.arange(len(targets)), test_size=0.2, random_state=seed, shuffle=True, stratify=targets)
        print("val idx", seed, valid_idx[:15])
        train_dataset_2 = torch.utils.data.Subset(dataset_2, train_idx)
        test_dataset_2 = torch.utils.data.Subset(dataset_2, valid_idx)
        
        print(len(train_dataset_1))
        print(len(train_dataset_2))
        print(len(test_dataset_1))
        print(len(test_dataset_2))
        
        
        if 'generations' in  args.mode or 'filtered' in args.mode:
            print('using generations')
            shutil.rmtree('missing_' + args.type + args.missing + '_empty/' + args.type)
            shutil.copytree(experiment_folder + 'generations/'+ args.missing + '/'+ args.type, 'missing_' + args.type + args.missing + '_empty/' + args.type)
            print("after copying", len(os.listdir('missing_' + args.type + args.missing + '_empty/' + args.type)))
            domain_list_train_3 = ["C116P77ThinF_empty", "C132P93ThinF_empty", "C137P98ThinF_empty", "C180P141NThinF_empty", "C182P143NThinF_empty", \
                                 "C184P145ThinF_empty", "C39P4thinF_empty", 'C59P20thinF_empty', "C68P29N_empty", "C99P60ThinF_empty"]   

            
            dataset_3 = HistoDataNorm('missing_'+args.type, domain_list=domain_list_train_3, augmentation=True)
            
            targets = dataset_3.train_labels
            train_idx, valid_idx= train_test_split(np.arange(len(targets)), test_size=0.2, random_state=seed, shuffle=True, stratify=targets)
    
            train_dataset_3 = torch.utils.data.Subset(dataset_3, train_idx)
            test_dataset_3 = torch.utils.data.Subset(dataset_3, valid_idx)
            train_dataset_2 = torch.utils.data.ConcatDataset([train_dataset_3,train_dataset_2])
            test_dataset_2 = torch.utils.data.ConcatDataset([test_dataset_3,test_dataset_2])
            print(len(train_dataset_2))
            print(len(test_dataset_2))


        #train_size_2 = int(0.8 * len(dataset_2))
        #test_size_2 = len(dataset_2) - train_size_2
        #train_dataset_2, test_dataset_2 = torch.utils.data.random_split(dataset_2, [train_size_2, test_size_2])

            
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_1,train_dataset_2])
        test_dataset = torch.utils.data.ConcatDataset([test_dataset_1,test_dataset_2])

    num_classes =2
    print("training on", len(train_dataset))
    print("validating on", len(test_dataset))

    trainloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, worker_init_fn=np.random.seed(seed))

    testloader = data_utils.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,
                                       num_workers=args.num_workers, worker_init_fn=np.random.seed(seed) )
    print("before", len(os.listdir('missing_' + args.type + args.missing + '_empty/' + args.type)))
    shutil.rmtree('missing_' + args.type + args.missing + '_empty/' + args.type)
    os.makedirs('missing_' + args.type + args.missing + '_empty/' + args.type)
    print("after", len(os.listdir('missing_' + args.type + args.missing + '_empty/' + args.type)))

    print("after val on", len(testloader.dataset))
    print("num_classes", num_classes)
    # Model
    print('Building model..')
    net = BasicResNet(num_classes=num_classes, network='resnet50', pretrained=True)
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        #cudnn.benchmark = False

    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir(experiment_folder + 'ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(experiment_folder + 'ckpts/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        global best_loss
        global global_step
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']
        global_step = start_epoch * len(domain_list_train)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))

    train_accuracy = 0.0
    test_accuracy = 0.0

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        new_train_accuracy = train(epoch, net, trainloader, device, optimizer, scheduler,
                                   loss_fn, args.max_grad_norm, args.mode)
        new_test_accuracy, update= test(epoch, net, testloader, device, loss_fn, args.mode,
                                                    experiment_folder=experiment_folder)

        if update:
            test_accuracy = new_test_accuracy
            train_accuracy = new_train_accuracy

    record = " Validation accuracy " + str(test_accuracy) + ". Train accuracy " + str(train_accuracy)
    print(record)
    file = open(experiment_folder + args.mode + "_" + str(args.temperature) + "_" +  "classification_validation_results.txt", "w+")
    file.write(record)
    file.close()


@torch.enable_grad()
def train(epoch, net, trainloader, device, optimizer, scheduler, loss_fn, max_grad_norm, mode):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    correct = 0

    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, y, d, yd in trainloader:
            x, y, d, yd = x.to(device), y.to(device), d.to(device), yd.to(device)
            optimizer.zero_grad()
            z = net(x)
            loss = loss_fn(z, y.argmax(dim=1))
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            if max_grad_norm > 0:
                util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            scheduler.step(global_step)

            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg),
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            global_step += x.size(0)
            values, pred = torch.max(z, 1)


            correct += pred.eq(y.argmax(dim=1)).sum().item()

    accuracy = correct * 100. / len(trainloader.dataset)

    print('train accuracy', accuracy)

    return accuracy


@torch.no_grad()
def test(epoch, net, testloader, device, loss_fn, mode, experiment_folder):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    correct = 0
    correct_0 = 0
    correct_1 = 0
    correct_array = torch.zeros(10)
    wrong_array = torch.zeros(10)
    update = False

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
            for img in range(y.size(0)):
                if pred.eq(y.argmax(dim=1))[img].item():
                    if y.argmax(dim=1)[img].item() == 0:
                        correct_0 += 1
                    else:
                        correct_1 += 1


    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs(experiment_folder + args.mode + "_" + str(args.temperature) + "_" +  'ckpts', exist_ok=True)
        torch.save(state, experiment_folder + args.mode + "_" + str(args.temperature) + "_" + 'ckpts/best.pth.tar')
        best_loss = loss_meter.avg
        update = True

    accuracy = correct * 100. / len(testloader.dataset)
    print(correct)
    print('val accuracy', accuracy)
    print(correct_array)
    print(wrong_array)

    return accuracy, update


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conditional Glow on Malaria Dataset')


    def str2bool(s):
        return s.lower().startswith('t')


    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    #parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
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
    parser.add_argument('--num_domains', type=int, default=10)
    parser.add_argument('--missing', type=str)
    parser.add_argument('--type', type=str)
    parser.add_argument('--temperature', type=float, default=1.0)


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
        
        # domain_list_train = ["C116P77ThinF", "C132P93ThinF", "C137P98ThinF", "C180P141NThinF", "C182P143NThinF", \
        #                     "C184P145ThinF", "C39P4thinF", 'C59P20thinF', "C68P29N", "C99P60ThinF"]
        # for testing in domain_list_train:
        # best_loss = float('inf')
        # global_step = 0
        # training = list(domain_list_train)
        # training.remove(testing)
        # print(testing)
        # print(training)

        # main(args, seed, str(testing), training)