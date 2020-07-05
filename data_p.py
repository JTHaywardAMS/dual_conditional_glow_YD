import numpy as np
import os
import torch
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def normalize(image, target=None):
    """Normalizing function we got from the cedars-sinai medical center"""
    if target is None:
        target = np.array([[148.60, 41.56], [169.30, 9.01], [105.97, 6.67]])

    M, N = image.shape[:2]

    whitemask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    whitemask = whitemask > 215 ## TODO: Hard code threshold; replace with Otsu

    imagelab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    imageL, imageA, imageB = cv2.split(imagelab)

    # mask is valid when true
    imageLM = np.ma.MaskedArray(imageL, whitemask)
    imageAM = np.ma.MaskedArray(imageA, whitemask)
    imageBM = np.ma.MaskedArray(imageB, whitemask)

    ## Sometimes STD is near 0, or 0; add epsilon to avoid div by 0 -NI
    epsilon = 1e-11

    imageLMean = imageLM.mean()
    imageLSTD = imageLM.std() + epsilon

    imageAMean = imageAM.mean()
    imageASTD = imageAM.std() + epsilon

    imageBMean = imageBM.mean()
    imageBSTD = imageBM.std() + epsilon

    # normalization in lab
    imageL = (imageL - imageLMean) / imageLSTD * target[0][1] + target[0][0]
    imageA = (imageA - imageAMean) / imageASTD * target[1][1] + target[1][0]
    imageB = (imageB - imageBMean) / imageBSTD * target[2][1] + target[2][0]

    imagelab = cv2.merge((imageL, imageA, imageB))
    imagelab = np.clip(imagelab, 0, 255)
    imagelab = imagelab.astype(np.uint8)

    # Back to RGB space
    returnimage = cv2.cvtColor(imagelab, cv2.COLOR_LAB2RGB)
    # Replace white pixels
    returnimage[whitemask] = image[whitemask]

    return returnimage

class HistoNormalize(object):
    """Normalizes the given PIL.Image"""

    def __call__(self, img):
        img_arr = np.array(img)
        img_norm = normalize(img_arr)
        img = Image.fromarray(img_norm, img.mode)
        return img

class RandomRotate(object):
    """Rotate the given PIL.Image by either 0, 90, 180, 270."""

    def __call__(self, img):
        random_rotation = np.random.randint(4, size=1)
        if random_rotation == 0:
            pass
        else:
            img = torchvision.transforms.functional.rotate(img, random_rotation*90)
        return img

class HistoDataNorm(data_utils.Dataset):
    def __init__(self, path, domain_list=[], augmentation=False, normalize=False):
        self.path = path
        self.domain_list = domain_list
        self.augmentation = augmentation
        self.normalize = normalize


        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Compose([transforms.Resize(32),
                                          transforms.CenterCrop(32)])
        self.hflip = transforms.RandomHorizontalFlip()
        self.vflip = transforms.RandomVerticalFlip()
        self.rrotate = RandomRotate()
        self.to_pil = transforms.ToPILImage()
        self.color_norm = HistoNormalize()

        self.train_data, self.train_labels, self.train_domains, self.train_yd = self.get_data()

    def get_imgs_and_labels(self, domain_path, type, label):
        class_path = domain_path + '/' + type
        all_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        # print("allfiles", len(all_files))
        all_files.sort()
        # if len(all_files)>0:
        #   all_files.pop()

        if len(all_files) > 0:
            img_list = []

            for file_name in all_files:
                file_path = class_path + '/' + file_name

                with open(file_path, 'rb') as f:
                    with Image.open(f) as img:
                        img = img.convert('RGB')
                        # print(img)
                if self.normalize:
                  img_list.append(self.to_tensor(self.resize(self.color_norm(img))))
                else:
                  img_list.append(self.to_tensor(self.resize(img)))

            # Concatenate
            imgs = torch.stack(img_list)

            # generate labels
            labels = torch.zeros(imgs.size()[0]) + label

        else:
            imgs, labels = torch.Tensor(), torch.Tensor()

        return imgs, labels.long()


    def get_data(self):
        # for each domain get all classes
        imgs_per_domain_list = []
        labels_per_domain_list = []
        domain_per_domain_list = []

        num_ims=0
        for i, domain in enumerate(self.domain_list):
            domain_path = self.path + domain

            # Extract for each class
            #imgs_uninfected, labels_uninfected = self.get_imgs_and_labels(domain_path, 'Uninfected', 0)
            imgs_parasitized, labels_parasitized = self.get_imgs_and_labels(domain_path, 'Parasitized', 1)
            # imgs_complex, labels_complex = self.get_imgs_and_labels(domain_path, '03_COMPLEX', 2)
            # imgs_lympho, labels_lympho = self.get_imgs_and_labels(domain_path, '04_LYMPHO', 3)
            # imgs_debris, labels_debris = self.get_imgs_and_labels(domain_path, '05_DEBRIS', 4)
            # imgs_mucosa, labels_mucosa = self.get_imgs_and_labels(domain_path, '06_MUCOSA', 5)
            # # imgs_adipose, labels_adipose = self.get_imgs_and_labels(domain_path, 'ADIPOSE', 6)
            # # imgs_empty, labels_empty = self.get_imgs_and_labels(domain_path, 'EMPTY', 7)

            # stack them per domain
            #imgs_per_domain = torch.cat([imgs_uninfected, imgs_parasitized])
            #labels_per_domain = torch.cat([labels_uninfected, labels_parasitized])
            imgs_per_domain = torch.cat([imgs_parasitized])
            labels_per_domain = torch.cat([labels_parasitized])
            domain_per_domain = torch.zeros(labels_per_domain.size()) + i

            # append everything
            imgs_per_domain_list.append(imgs_per_domain)
            labels_per_domain_list.append(labels_per_domain)
            domain_per_domain_list.append(domain_per_domain)

        # One last cat
        train_imgs = torch.cat(imgs_per_domain_list)
        train_labels = torch.cat(labels_per_domain_list)
        train_domains = torch.cat(domain_per_domain_list).long()

        # Convert to onehot

        d = torch.eye(len(self.domain_list))
        train_domains = d[train_domains]



        # yd = torch.eye(len(self.domain_list)*2)
        # train_yd = yd[train_labels*len(self.domain_list) +train_domains]

        y = torch.eye(2)
        train_labels = y[train_labels]

        train_yd = torch.cat([train_labels,train_domains], dim=1)

        return train_imgs, train_labels, train_domains, train_yd
        # return train_imgs, train_labels, train_domains

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):
        x = self.train_data[index]
        y = self.train_labels[index]
        d = self.train_domains[index]
        yd = self.train_yd[index]
      

        if self.augmentation:
            x = self.to_tensor(self.vflip(self.hflip(self.rrotate(self.to_pil(x)))))

        return x, y, d, yd