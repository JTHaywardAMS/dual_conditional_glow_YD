import json
import os
from PIL import Image
import argparse
import torch
from torchvision.utils import make_grid
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import numpy as np
from model import Glow

def main(args):
    seed_list = [int(item) for item in args.seed.split(',')]

    for seed in seed_list:

        device = torch.device("cuda")

        experiment_folder = args.experiment_folder + '/' + str(seed) +'/'
        print(experiment_folder)
        #model_name = 'glow_checkpoint_'+ str(args.chk)+'.pth'
        for thing in os.listdir(experiment_folder):
            if 'best' in thing:
                model_name = thing
        print(model_name)

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        with open(experiment_folder + 'hparams.json') as json_file:
            hparams = json.load(json_file)

        image_shape = (32, 32, 3)
        if hparams['y_condition']:
            num_classes = 2
            num_domains = 0
        elif hparams['d_condition']:
            num_classes=10
            num_domains=0
        elif hparams['yd_condition']:
            num_classes=2
            num_domains=10
        else:
            num_classes=2
            num_domains=0

        model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                     hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes, num_domains,
                     hparams['learn_top'], hparams['y_condition'], hparams['extra_condition'], hparams['sp_condition'], hparams['d_condition'], hparams['yd_condition'])
        print('loading model')
        model.load_state_dict(torch.load(experiment_folder + model_name))
        model.set_actnorm_init()

        model = model.to(device)

        model = model.eval()
        
        if hparams['y_condition']:
            print('y_condition')
            def sample(model, temp=args.temperature):
                with torch.no_grad():
                    if hparams['y_condition']:
                        print("extra", hparams['extra_condition'])
                        y = torch.eye(num_classes)
                        y = torch.cat(1000*[y])
                        print(y.size())
                        y_0 = y[::2, :].to(device) # number hardcoded in model for now
                        y_1 = y[1::2, :].to(device)
                        print(y_0.size())
                        print(y_0)
                        print(y_1)
                        print(y_1.size())
                        images0 = model(z=None,y_onehot=y_0, temperature=temp, reverse=True, batch_size=1000)
                        images1 = model(z=None, y_onehot=y_1, temperature=temp, reverse=True, batch_size=1000)
                return images0, images1

            images0, images1 = sample(model)

            os.makedirs(experiment_folder + 'generations/Uninfected', exist_ok=True)
            os.makedirs(experiment_folder + 'generations/Parasitized', exist_ok=True)
            for i in range(images0.size(0)):
                torchvision.utils.save_image(images0[i, :, :, :], experiment_folder + 'generations/Uninfected/sample_{}.png'.format(i))
                torchvision.utils.save_image(images1[i, :, :, :], experiment_folder + 'generations/Parasitized/sample_{}.png'.format(i))
            images_concat0 = torchvision.utils.make_grid(images0[:64,:,:,:], nrow=int(64 ** 0.5), padding=2, pad_value=255)
            torchvision.utils.save_image(images_concat0, experiment_folder + '/uninfected.png')
            images_concat1 = torchvision.utils.make_grid(images1[:64,:,:,:], nrow=int(64 ** 0.5), padding=2, pad_value=255)
            torchvision.utils.save_image(images_concat1, experiment_folder + '/parasitized.png')

        elif hparams['d_condition']:
            print('d_cond')
            def sample_d(model, idx, batch_size=1000, temp=args.temperature):
                with torch.no_grad():
                    if hparams['d_condition']:

                        y_0 = torch.zeros([batch_size,10], device='cuda:0')
                        y_0[:,idx] = torch.ones(batch_size)
                        y_0.to(device)
                        print(y_0)

                        # y_1 = torch.zeros([batch_size, 201], device='cuda:0')
                        # y_1[:, 157] = torch.ones(batch_size)
                        # y_1.to(device)
                        # y = torch.eye(num_classes)
                        # y = torch.cat(1000 * [y])
                        # print(y.size())
                        # y_0 = y[::2, :].to(device)  # number hardcoded in model for now
                        # y_1 = y[1::2, :].to(device)
                        # print(y_0.size())
                        # print(y_0)
                        # print(y_1)
                        # print(y_1.size())

                        images0 = model(z=None,y_onehot=y_0, temperature=temp, reverse=True, batch_size=1000)
                        # images1 = model(z=None, y_onehot=y_1, temperature=1.0, reverse=True, batch_size=1000)
                return images0

            for idx, dom in enumerate(["C116P77ThinF", "C132P93ThinF", "C137P98ThinF", "C180P141NThinF", "C182P143NThinF", \
                             "C184P145ThinF", "C39P4thinF", 'C59P20thinF', "C68P29N", "C99P60ThinF"]):

                images0 = sample_d(model, idx)

                os.makedirs(experiment_folder + 'generations/'+dom+ '/Uninfected/', exist_ok=True)
                os.makedirs(experiment_folder + 'generations/'+dom+ '/Parasitized/', exist_ok=True)
                # os.makedirs(experiment_folder + 'generations/C59P20thinF/Uninfected/', exist_ok=True)
                # os.makedirs(experiment_folder + 'generations/C59P20thinF/Parasitized/', exist_ok=True)
                for i in range(images0.size(0)):
                    torchvision.utils.save_image(images0[i, :, :, :], experiment_folder + 'generations/' + dom + '/Uninfected/sample_{}.png'.format(i))
                    #torchvision.utils.save_image(images1[i, :, :, :], experiment_folder + 'generations/C59P20thinF/Parasitized/sample_{}.png'.format(i))
                images_concat0 = torchvision.utils.make_grid(images0[:25,:,:,:], nrow=int(25** 0.5), padding=2, pad_value=255)
                torchvision.utils.save_image(images_concat0, experiment_folder + dom + '.png')
                # images_concat1 = torchvision.utils.make_grid(images1[:64,:,:,:], nrow=int(64 ** 0.5), padding=2, pad_value=255)
                # torchvision.utils.save_image(images_concat1, experiment_folder + 'C59P20thinF.png')

        elif hparams['yd_condition']:

            def sample_YD(model, idx, batch_size=1000, temp=args.temperature):
                with torch.no_grad():
                    if hparams['yd_condition']:
                        y_0 = torch.zeros([batch_size, 12], device='cuda:0')
                        y_0[:, 0] = torch.ones(batch_size)
                        y_0[:, idx+2] = torch.ones(batch_size)
                        y_0.to(device)
                        print(y_0)

                        y_1 = torch.zeros([batch_size, 12], device='cuda:0')
                        y_1[:, 1] = torch.ones(batch_size)
                        y_1[:, idx+2] = torch.ones(batch_size)
                        y_1.to(device)
                        print(y_1)

                        images0 = model(z=None, y_onehot=y_0, temperature=temp, reverse=True, batch_size=1000)
                        images1 = model(z=None, y_onehot=y_1, temperature=temp, reverse=True, batch_size=1000)
                return images0, images1

            def sample_DD(model, idx, batch_size=1000, temp=args.temperature):
                with torch.no_grad():
                    if hparams['yd_condition']:
                        y_1 = torch.zeros([batch_size, 20], device='cuda:0')
                        y_1[:, idx] = torch.ones(batch_size)
                        y_1.to(device)
                        print(y_1)

                        y_0 = torch.zeros([batch_size, 20], device='cuda:0')
                        y_0[:, idx+10] = torch.ones(batch_size)
                        y_0.to(device)
                        print(y_0)

                        images0 = model(z=None, y_onehot=y_0, temperature=temp, reverse=True, batch_size=1000)
                        images1 = model(z=None, y_onehot=y_1, temperature=temp, reverse=True, batch_size=1000)
                return images0, images1

            for idx, dom in enumerate(
                    ["C116P77ThinF", "C132P93ThinF", "C137P98ThinF", "C180P141NThinF", "C182P143NThinF", \
                     "C184P145ThinF", "C39P4thinF", 'C59P20thinF', "C68P29N", "C99P60ThinF"]):

                images0, images1 = sample_YD(model, idx)

                os.makedirs(experiment_folder + 'generations/' + dom + '/Uninfected/', exist_ok=True)
                os.makedirs(experiment_folder + 'generations/' + dom + '/Parasitized/', exist_ok=True)
                for i in range(images0.size(0)):
                    torchvision.utils.save_image(images0[i, :, :, :],
                                                 experiment_folder + 'generations/' + dom + '/Uninfected/sample_{}.png'.format(
                                                     i))
                    torchvision.utils.save_image(images1[i, :, :, :],
                                                 experiment_folder + 'generations/' + dom + '/Parasitized/sample_{}.png'.format(
                                                     i))
                images_concat0 = torchvision.utils.make_grid(images0[:64, :, :, :], nrow=int(64 ** 0.5), padding=2,
                                                             pad_value=255)
                torchvision.utils.save_image(images_concat0, experiment_folder +  dom + str(args.temperature)+ '_uninfected.png')
                images_concat1 = torchvision.utils.make_grid(images1[:64,:,:,:], nrow=int(64** 0.5), padding=2, pad_value=255)
                torchvision.utils.save_image(images_concat1, experiment_folder + dom + str(args.temperature)+'_parasitized.png')


        else:
            def sample(model, temp=args.temperature):
                with torch.no_grad():
                    images = model(z=None,y_onehot=None, temperature=temp, reverse=True, batch_size=1000)

                return images

            images = sample(model)

            os.makedirs('unconditioned/' + str(seed) +'/generations/' + experiment_folder[:-3] , exist_ok=True)
            for i in range(images.size(0)):
                torchvision.utils.save_image(images[i, :, :, :],
                                             'unconditioned/' + str(seed) +'/generations/' + experiment_folder[:-2] + 'sample_{}.png'.format(
                                                 i))

            images_concat = torchvision.utils.make_grid(images[:64, :, :, :], nrow=int(64 ** 0.5), padding=2, pad_value=255)
            torchvision.utils.save_image(images_concat, 'unconditioned/' +str(seed) +'/'+ experiment_folder[:-3] + '.png')

        # os.makedirs('d_unconditioned/generations/' + experiment_folder + 'Uninfected', exist_ok=True)
            # os.makedirs('d_unconditioned/generations/' + experiment_folder + 'Parasitized', exist_ok=True)
            # for i in range(images.size(0)):
            #     torchvision.utils.save_image(images[i, :, :, :], 'd_unconditioned/' + 'generations/' + experiment_folder+ 'Uninfected/sample_{}.png'.format(i))
            #
            # images_concat = torchvision.utils.make_grid(images[:64,:,:,:], nrow=int(64 ** 0.5), padding=2, pad_value=255)
            # torchvision.utils.save_image(images_concat, 'd_unconditioned/' + experiment_folder[:-1] + '.png')
            #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample Conditional Glow on Malaria Dataset')


    parser.add_argument('--seed', type=str, default="0,1,2", help='Random seed for reproducibility')
    parser.add_argument('--experiment_folder', default="new")
    parser.add_argument('--chk', type=int, default=34400)
    parser.add_argument('--temperature', type=float, default=1.0)




    main(parser.parse_args())