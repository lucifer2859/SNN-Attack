import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import sys
sys.path.append('../')
import SpikingFlow.softbp.neuron as neuron
import SpikingFlow.softbp.layer as layer
import SpikingFlow.encoding as encoding
import SpikingFlow.softbp.soft_pulse_function as soft_pulse_function
from torch.utils.tensorboard import SummaryWriter
import readline
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = torch.clamp(t, 0.0, 1.0)
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

class ANN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 7 * 7

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 7 * 7, 128 * 3 * 3, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128 * 3 * 3, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 100, bias=False),
            nn.ReLU()
        )
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        return self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze()

class SNN_Net(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.PLIFNode(v_threshold=v_threshold, v_reset=v_reset),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),
            nn.MaxPool2d(2, 2)  # 7 * 7

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.7),
            nn.Linear(128 * 7 * 7, 128 * 3 * 3, bias=False),
            neuron.PLIFNode(v_threshold=v_threshold, v_reset=v_reset),
            layer.Dropout(0.7),
            nn.Linear(128 * 3 * 3, 128, bias=False),
            neuron.PLIFNode(v_threshold=v_threshold, v_reset=v_reset),
            nn.Linear(128, 100, bias=False),
            neuron.PLIFNode(v_threshold=v_threshold, v_reset=v_reset)
        )
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        return self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze()

    def reset_(self):
        for item in self.modules():
            if hasattr(item, 'reset'):
                item.reset()

def main():
    # device = input('输入运行的设备，例如“cpu”或“cuda:0”  ')
    # dataset_dir = input('输入保存MNIST数据集的位置，例如“./”  ')
    # class_num = int(input('输入class_num，例如“10”  '))
    # T = int(input('输入仿真时长，例如“50”  ')) 
    # phase = input('输入算法阶段，例如“BIM”  ')

    device = 'cuda:3'
    dataset_dir = '../../dataset/'
    class_num = 10
    T = 50
    phase = 'BIM'

    torch.cuda.empty_cache()

    encoder = encoding.PoissonEncoder()

    if phase == 'BIM':
        # model_path = input('输入模型文件路径，例如“./model.pth”  ')
        # iter_num = int(input('输入对抗攻击的迭代次数，例如“25”  '))
        # eta = float(input('输入对抗攻击学习率，例如“0.05”  '))
        # attack_type = input('输入攻击类型，例如“UT/T”  ')
        # clip_flag = bool(input('输入是否使用截断，例如“True/False”  '))
        # clip_eps = float(input('输入截断eps，例如“0.01”  '))

        source_model_path = './models/mnist_img_v1.pth'
        target_model_path1 = './models/mnist_ann_v1.pth'
        target_model_path2 = './models/mnist_spike_v1.pth'
        
        iter_num = 25
        eta = 0.02
        attack_type = 'T'

        clip_flag = True
        clip_eps = 0.3

        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=1,
            shuffle=False,
            drop_last=False)

        source_net = SNN_Net().to(device)
        source_net.load_state_dict(torch.load(source_model_path))

        target_net1 = ANN_Net().to(device)
        target_net1.load_state_dict(torch.load(target_model_path1))

        target_net2 = SNN_Net().to(device)
        target_net2.load_state_dict(torch.load(target_model_path2))

        target_net1.eval()
        target_net2.eval()

        mean_p = 0.0
        test_sum = 0
        source_success_sum = 0
        target_success_sum1 = 0
        target_success_sum2 = 0

        if attack_type == 'UT':
            for X, y in test_data_loader:
                img, label = X.to(device), y.to(device)
                img_ori = torch.rand_like(img).copy_(img)
                img.requires_grad = True

                test_sum += 1

                print('Img %d' % test_sum)

                source_net.train()

                for it in range(iter_num):
                    for t in range(T):
                        if t == 0:
                            out_spikes_counter = source_net(img).unsqueeze(0)
                        else:
                            out_spikes_counter += source_net(img).unsqueeze(0)

                    out_spikes_counter_frequency = out_spikes_counter / T

                    # loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(label, class_num).float())
                    loss = F.cross_entropy(out_spikes_counter_frequency, label)

                    loss.backward()

                    img_grad = torch.sign(img.grad.data)

                    img_adv = None

                    if clip_flag:
                        img_adv = clip_by_tensor(img + eta * img_grad, img_ori - clip_eps, img_ori + clip_eps)
                    else:
                        img_adv = torch.clamp(img + eta * img_grad, 0.0, 1.0)

                    img = Variable(img_adv, requires_grad=True)

                    source_net.reset_()
                
                source_net.eval()

                with torch.no_grad():
                    img_diff = img - img_ori

                    l2_norm = torch.norm(img_diff.view(img_diff.size()[0], -1), dim=1).item()
                    print('Perturbation: %f' % l2_norm)

                    mean_p += l2_norm

                    target_output1 = target_net1(img).unsqueeze(0)

                    for t in range(T):
                        if t == 0:
                            source_out_spikes_counter = source_net(img).unsqueeze(0)
                            target_out_spikes_counter2 = target_net2(encoder(img).float()).unsqueeze(0)
                        else:
                            source_out_spikes_counter += source_net(img).unsqueeze(0)
                            target_out_spikes_counter2 += target_net2(encoder(img).float()).unsqueeze(0)

                    source_output = source_out_spikes_counter / T
                    target_output2 = target_out_spikes_counter2 / T

                    source_attack_flag = (source_output.max(1)[1] != label).float().sum().item()
                    source_success_sum += source_attack_flag

                    target_attack_flag1 = (target_output1.max(1)[1] != label).float().sum().item()
                    target_success_sum1 += target_attack_flag1

                    target_attack_flag2 = (target_output2.max(1)[1] != label).float().sum().item()
                    target_success_sum2 += target_attack_flag2

                    source_net.reset_()
                    target_net2.reset_()

                    if source_attack_flag > 0.5:
                        print('Source Attack Success')
                    else:
                        print('Source Attack Failure')

                    if target_attack_flag1 > 0.5:
                        print('Target Attack 1 Success')
                    else:
                        print('Target Attack 1 Failure')

                    if target_attack_flag2 > 0.5:
                        print('Target Attack 2 Success')
                    else:
                        print('Target Attack 2 Failure')

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
                        for t in range(T):
                            if t == 0:
                                out_spikes_counter = source_net(img).unsqueeze(0)
                            else:
                                out_spikes_counter += source_net(img).unsqueeze(0)

                        out_spikes_counter_frequency = out_spikes_counter / T

                        # loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(target_label, class_num).float())
                        loss = F.cross_entropy(out_spikes_counter_frequency, target_label)
                        
                        loss.backward()

                        img_grad = torch.sign(img.grad.data)

                        img_adv = None

                        if clip_flag:
                            img_adv = clip_by_tensor(img - eta * img_grad, img_ori - clip_eps, img_ori + clip_eps)
                        else:
                            img_adv = torch.clamp(img - eta * img_grad, 0.0, 1.0)

                        img = Variable(img_adv, requires_grad=True)

                        source_net.reset_()
                    
                    source_net.eval()

                    with torch.no_grad():
                        img_diff = img - img_ori

                        l2_norm = torch.norm(img_diff.view(img_diff.size()[0], -1), dim=1).item()
                        print('Perturbation: %f' % l2_norm)

                        mean_p += l2_norm

                        target_output1 = target_net1(img).unsqueeze(0)

                        for t in range(T):
                            if t == 0:
                                source_out_spikes_counter = source_net(img).unsqueeze(0)
                                target_out_spikes_counter2 = target_net2(encoder(img).float()).unsqueeze(0)
                            else:
                                source_out_spikes_counter += source_net(img).unsqueeze(0)
                                target_out_spikes_counter2 += target_net2(encoder(img).float()).unsqueeze(0)

                        source_output = source_out_spikes_counter / T
                        target_output2 = target_out_spikes_counter2 / T

                        source_attack_flag = (source_output.max(1)[1] == target_label).float().sum().item()
                        source_success_sum += source_attack_flag

                        target_attack_flag1 = (target_output1.max(1)[1] == target_label).float().sum().item()
                        target_success_sum1 += target_attack_flag1

                        target_attack_flag2 = (target_output2.max(1)[1] == target_label).float().sum().item()
                        target_success_sum2 += target_attack_flag2

                        source_net.reset_()
                        target_net2.reset_()

                        if source_attack_flag > 0.5:
                            print('Source Attack Success')
                        else:
                            print('Source Attack Failure')

                        if target_attack_flag1 > 0.5:
                            print('Target Attack 1 Success')
                        else:
                            print('Target Attack 1 Failure')

                        if target_attack_flag2 > 0.5:
                            print('Target Attack 2 Success')
                        else:
                            print('Target Attack 2 Failure')

                if test_sum >= 270:
                    mean_p /= 270
                    break
        
        print('Mean Perturbation: %.2f' % mean_p)
        print('source_success_sum: %d' % source_success_sum)
        print('target_success_1_sum: %d' % target_success_sum1)
        print('target_success_2_sum: %d' % target_success_sum2)
        print('test_sum: %d' % test_sum)
        print('source_success_rate: %.2f%%' % (100 * source_success_sum / test_sum))
        print('target_success_1_rate: %.2f%%' % (100 * target_success_sum1 / test_sum))
        print('target_success_2_rate: %.2f%%' % (100 * target_success_sum2 / test_sum))

if __name__ == '__main__':
    main()