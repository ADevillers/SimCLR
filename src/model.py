import torch
import torchvision



class SimCLR(torch.nn.Module):
    def __init__(self, resnet_type='resnet50', z_dim=128, cifar10=False):
        super().__init__()
        resnet_dict = {
            'resnet18': torchvision.models.resnet18,
            'resnet34': torchvision.models.resnet34,
            'resnet50': torchvision.models.resnet50,
            'resnet101': torchvision.models.resnet101,
            'resnet152': torchvision.models.resnet152
        }

        self.resnet = resnet_dict[resnet_type](pretrained=False, zero_init_residual=True)
        self.h_dim = self.resnet.fc.in_features
        self.z_dim = z_dim

        self.resnet.fc = torch.nn.Identity()
        if cifar10:
            self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.resnet.maxpool = torch.nn.Identity()

        self.proj_head_inv = torch.nn.Sequential(
            torch.nn.Linear(self.h_dim, 2048, bias=False),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            # torch.nn.Linear(2048, 2048, bias=False),
            # torch.nn.BatchNorm1d(2048),
            # torch.nn.ReLU(),
            torch.nn.Linear(2048, self.z_dim, bias=False),
            torch.nn.BatchNorm1d(self.z_dim, affine=False)
        )
    
    def forward(self, images):
        h = self.resnet(images)
        z = self.proj_head_inv(h)

        return h, z
