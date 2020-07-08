import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os
import sys
sys.path.append('../')
from torch.utils.tensorboard import SummaryWriter
import readline
from torch.autograd import Variable
import numpy as np

def clip_by_tensor(t, t_min, t_max, p_min, p_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :param p_min: pixel min
    :param p_max: pixel max
    :return: cliped tensor
    """

    result = (t >= p_min).float() * t + (t < p_min).float() * p_min
    result = (result <= p_max).float() * result + (result > p_max).float() * p_max

    result = (result >= t_min).float() * result + (result < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result.float()

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.train_times = 0
        self.max_test_acccuracy = 0

        self.conv = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),  # 16 * 16

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 8 * 8

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 128 * 4 * 4, bias=False),
            nn.ReLU(),
            nn.Linear(128 * 4 * 4, 100, bias=False),
            nn.ReLU()
        )
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        return self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze()

def main():
    # device = input('输入运行的设备，例如“cpu”或“cuda:0”  ')
    # dataset_dir = input('输入保存MNIST数据集的位置，例如“./”  ')
    # class_num = int(input('输入class_num，例如“10”  '))
    # phase = input('输入算法阶段，例如“BIM”  ')

    device = 'cuda:2'
    dataset_dir = '../../dataset/'
    class_num = 10
    phase = 'BIM'

    torch.cuda.empty_cache()

    if phase == 'BIM':
        # model_path = input('输入模型文件路径，例如“./model.pth”  ')
        # iter_num = int(input('输入对抗攻击的迭代次数，例如“25”  '))
        # eta = float(input('输入对抗攻击学习率，例如“0.05”  '))
        # attack_type = input('输入攻击类型，例如“UT/T”  ')
        # clip_eps = float(input('输入截断eps，例如“0.01”  '))

        source_model_path = './models/cifar10_ann_v2.pth'
        target_model_path = './models/cifar10_ann_v1.pth'
        
        iter_num = 50
        eta = 0.002
        attack_type = 'UT'

        clip_eps = 0.05

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(
                root=dataset_dir,
                train=False,
                transform=transform_test,
                download=True),
            batch_size=1,
            shuffle=False,
            drop_last=False)

        p_max = transform_test(np.ones((32, 32, 3))).to(device)
        p_min = transform_test(np.zeros((32, 32, 3))).to(device)

        source_net = Net().to(device)
        source_net.load_state_dict(torch.load(source_model_path))

        target_net = Net().to(device)
        target_net.load_state_dict(torch.load(target_model_path))

        target_net.eval()

        mean_p = 0.0
        test_sum = 0
        source_success_sum = 0
        target_success_sum = 0

        if attack_type == 'UT':
            for X, y in test_data_loader:
                img, label = X.to(device), y.to(device)
                img_ori = torch.rand_like(img).copy_(img)
                img.requires_grad = True

                test_sum += 1

                print('Img %d' % test_sum)

                source_net.train()

                for it in range(iter_num):
                    output = source_net(img).unsqueeze(0)

                    # loss = F.mse_loss(output, F.one_hot(label, class_num).float())
                    loss = F.cross_entropy(output, label)

                    loss.backward()

                    img_grad = torch.sign(img.grad.data)

                    img_adv = clip_by_tensor(img + eta * img_grad, img_ori - clip_eps, img_ori + clip_eps, p_min, p_max)

                    img = Variable(img_adv, requires_grad=True)

                source_net.eval()

                with torch.no_grad():
                    img_diff = img - img_ori
                    
                    l_norm = torch.max(torch.abs(img_diff)).item()
                    print('Perturbation: %f' % l_norm)

                    mean_p += l_norm

                    source_output = source_net(img).unsqueeze(0)
                    target_output = target_net(img).unsqueeze(0)

                    source_attack_flag = (source_output.max(1)[1] != label).float().sum().item()
                    source_success_sum += source_attack_flag

                    target_attack_flag = (target_output.max(1)[1] != label).float().sum().item()
                    target_success_sum += target_attack_flag

                    if source_attack_flag > 0.5:
                        print('Source Attack Success')
                    else:
                        print('Source Attack Failure')

                    if target_attack_flag > 0.5:
                        print('Target Attack Success')
                    else:
                        print('Target Attack Failure')

                if test_sum >= 250:
                    mean_p /= 250
                    break
        else:
            for X, y in test_data_loader:
                for i in range(1, class_num):
                    img, label = X.to(device), y.to(device)
                    img_ori = torch.rand_like(img).copy_(img)
                    img.requires_grad = True
                    
                    target_label = (label + i) % class_num

                    test_sum += 1

                    source_net.train()

                    for it in range(iter_num):
                        output = source_net(img).unsqueeze(0)

                        # loss = F.mse_loss(output, F.one_hot(target_label, class_num).float())
                        loss = F.cross_entropy(output, target_label)

                        loss.backward()

                        img_grad = torch.sign(img.grad.data)

                        img_adv = clip_by_tensor(img - eta * img_grad, img_ori - clip_eps, img_ori + clip_eps, p_min, p_max)

                        img = Variable(img_adv, requires_grad=True)

                    source_net.eval()

                    with torch.no_grad():
                        img_diff = img - img_ori

                        l_norm = torch.max(torch.abs(img_diff)).item()
                        print('Perturbation: %f' % l_norm)

                        mean_p += l_norm

                        source_output = source_net(img).unsqueeze(0)
                        target_output = target_net(img).unsqueeze(0)

                        source_attack_flag = (source_output.max(1)[1] == target_label).float().sum().item()
                        source_success_sum += source_attack_flag

                        target_attack_flag = (target_output.max(1)[1] == target_label).float().sum().item()
                        target_success_sum += target_attack_flag

                        if source_attack_flag > 0.5:
                            print('Source Attack Success')
                        else:
                            print('Source Attack Failure')

                        if target_attack_flag > 0.5:
                            print('Target Attack Success')
                        else:
                            print('Target Attack Failure')

                if test_sum >= 270:
                    mean_p /= 270
                    break
        
        print('Mean Perturbation: %.3f' % mean_p)
        print('source_success_sum: %d' % source_success_sum)
        print('target_success_sum: %d' % target_success_sum)
        print('test_sum: %d' % test_sum)
        print('source_success_rate: %.2f%%' % (100 * source_success_sum / test_sum))
        print('target_success_rate: %.2f%%' % (100 * target_success_sum / test_sum))

if __name__ == '__main__':
    main()