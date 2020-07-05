import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class BasicResNet(nn.Module):
    def __init__(self, num_classes=100, network='resnet18', pretrained=True, dropout_rate=0.2, num_domain_classes=None):
        super(BasicResNet, self).__init__()

        if network == 'resnet152':
            resnet_model = models.resnet152(pretrained=pretrained)
        elif network == 'resnet101':
            resnet_model = models.resnet101(pretrained=pretrained)
        elif network == 'resnet50':
            resnet_model = models.resnet50(pretrained=pretrained)
        elif network == 'resnet34':
            resnet_model = models.resnet34(pretrained=pretrained)
        elif network == 'resnet18':
            resnet_model = models.resnet18(pretrained=pretrained)
        elif network == 'MobileNetV2':
            resnet_model =  models.mobilenet_v2(pretrained=pretrained)
        elif network == 'inceptionV3':
            resnet_model = models.inception_v3(pretrained=pretrained)
        elif network == 'mnasnet':
            resnet_model = models.mnasnet1_0(pretrained=pretrained)
        elif network == 'VGG':
            resnet_model = models.vgg11(pretrained=pretrained)
        else:
            raise Exception("{} model type not supported".format(network))

        self.num_domain_classes=num_domain_classes
        self.resnet_model = resnet_model


        if 'resnet' in network:
            if num_domain_classes is None:
                self.resnet_model.fc = nn.Sequential(
                    nn.BatchNorm1d(resnet_model.fc.in_features),
                    nn.Dropout(dropout_rate),
                    nn.Linear(resnet_model.fc.in_features, resnet_model.fc.in_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(resnet_model.fc.in_features),
                    nn.Dropout(dropout_rate),
                    nn.Linear(resnet_model.fc.in_features, resnet_model.fc.in_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(resnet_model.fc.in_features),
                    nn.Dropout(dropout_rate),
                    nn.Linear(resnet_model.fc.in_features, num_classes),
                )
            else:
                self.resnet_model.fc = nn.Sequential(
                    nn.BatchNorm1d(resnet_model.fc.in_features),
                    nn.Dropout(dropout_rate),
                    nn.Linear(resnet_model.fc.in_features, resnet_model.fc.in_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(resnet_model.fc.in_features),
                    nn.Dropout(dropout_rate),
                    nn.Linear(resnet_model.fc.in_features, resnet_model.fc.in_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(resnet_model.fc.in_features),
                    nn.Dropout(dropout_rate),
                    nn.Linear(resnet_model.fc.in_features, num_classes + num_domain_classes),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_classes + num_domain_classes),
                    nn.Dropout(dropout_rate)
                )
                self.classes_output = nn.Linear(num_classes + num_domain_classes, num_classes)
                self.domain_output = nn.Linear(num_classes + num_domain_classes, num_domain_classes)
        elif 'VG' in network:
            self.resnet_model.classifier = nn.Sequential(
                nn.Linear(25088, 4096, bias=True),
                nn.ReLU(),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(4096, 4096, bias=True),
                nn.ReLU(),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(4096, num_classes))

        # print(resnet_model)

    def forward(self, x):
        if self.num_domain_classes is None:
            output = self.resnet_model.forward(x)
            return output

        else:
            output = self.resnet_model.forward(x)
            output1 = self.classes_output(output)
            output2 = self.domain_output(output)
            return output1, output2
