import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import sys
sys.path.append('../')
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

def clamp(n, n_min, n_max):
    if n > n_max:
        return n_min
    if n < n_min:
        return n_min
    return n

class Net(nn.Module):
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

def main():
    # device = input('输入运行的设备，例如“cpu”或“cuda:0”  ')
    # dataset_dir = input('输入保存MNIST数据集的位置，例如“./”  ')
    # class_num = int(input('输入class_num，例如“10”  '))
    # lr = float(input('输入学习率，例如“1e-3”  '))
    # phase = input('输入算法阶段，例如“train/FGSM/BIM”  ')

    device = 'cuda:2'
    dataset_dir = '../../dataset/'
    class_num = 10
    lr = 1e-4
    phase = 'OPA'

    torch.cuda.empty_cache()

    if phase == 'train':
        # model_dir = input('输入保存模型文件的位置，例如“./”  ')
        # batch_size = int(input('输入batch_size，例如“64”  '))
        # train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”  '))
        # log_dir = input('输入保存tensorboard日志文件的位置，例如“./”  ')

        model_dir = './models/'
        batch_size = 64
        train_epoch = 9999999
        log_dir = './logs/'

        writer = SummaryWriter(log_dir)

        train_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)
        
        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False)

        net = Net().to(device)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        
        train_times = 0
        best_epoch = 0
        max_correct_sum = 0

        for epoch in range(1, train_epoch + 1):
            net.train()

            for X, y in train_data_loader:
                img, label = X.to(device), y.to(device)
                
                optimizer.zero_grad()

                output = net(img)

                loss =  F.mse_loss(output, F.one_hot(label, class_num).float())
                # loss = F.cross_entropy(output, label)

                loss.backward()
                optimizer.step()

                correct_rate = (output.max(1)[1] == label).float().mean().item()
                writer.add_scalar('train_correct_rate', correct_rate, train_times)
                # if train_times % 1024 == 0:
                #     print(device, dataset_dir, batch_size, lr, train_epoch, log_dir)
                #     print(sys.argv, 'train_times', train_times, 'train_correct_rate', correct_rate)
                train_times += 1

            net.eval()

            with torch.no_grad():
                test_sum = 0
                correct_sum = 0
                for X, y in test_data_loader:
                    img, label = X.to(device), y.to(device)
                    
                    output = net(img)

                    correct_sum += (output.max(1)[1] == label).float().sum().item()
                    test_sum += label.numel()

                writer.add_scalar('test_correct_rate', correct_sum / test_sum, train_times)

                print('epoch', epoch, 'test_correct_rate', correct_sum / test_sum)

                if correct_sum > max_correct_sum:
                    max_correct_sum = correct_sum
                    torch.save(net.state_dict(), model_dir + 'ann_best_%d.pth' % (epoch))
                    if best_epoch > 0:
                        os.system('rm %sann_best_%d.pth' % (model_dir, best_epoch))
                    best_epoch = epoch

    elif phase == 'BIM':
        # model_path = input('输入模型文件路径，例如“./model.pth”  ')
        # iter_num = int(input('输入对抗攻击的迭代次数，例如“25”  '))
        # eta = float(input('输入对抗攻击学习率，例如“0.05”  '))
        # attack_type = input('输入攻击类型，例如“UT/T”  ')
        # clip_flag = bool(input('输入是否使用截断，例如“True/False”  '))
        # clip_eps = float(input('输入截断eps，例如“0.01”  '))

        model_path = './models/mnist_ann_v1.pth'
        iter_num = 1000
        eta = 0.02
        attack_type = 'T'

        clip_flag = True
        clip_eps = 0.5

        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=1,
            shuffle=False,
            drop_last=False)

        net = Net().to(device)
        net.load_state_dict(torch.load(model_path))

        mean_p = 0.0
        test_sum = 0
        success_sum = 0

        if attack_type == 'UT':
            for X, y in test_data_loader:
                img, label = X.to(device), y.to(device)
                img_ori = torch.rand_like(img).copy_(img)
                img.requires_grad = True

                test_sum += 1

                print('Img %d' % test_sum)

                net.train()

                for it in range(iter_num):
                    output = net(img).unsqueeze(0)

                    # loss = F.mse_loss(output, F.one_hot(label, class_num).float())
                    loss = F.cross_entropy(output, label)

                    loss.backward()

                    img_grad = torch.sign(img.grad.data)

                    img_adv = None

                    if clip_flag:
                        img_adv = clip_by_tensor(img + eta * img_grad, img_ori - clip_eps, img_ori + clip_eps)
                    else:
                        img_adv = torch.clamp(img + eta * img_grad, 0.0, 1.0)

                    img = Variable(img_adv, requires_grad=True)
                
                net.eval()

                with torch.no_grad():
                    img_diff = img - img_ori

                    l2_norm = torch.norm(img_diff.view(img_diff.size()[0], -1), dim=1).item()
                    print('Perturbation: %f' % l2_norm)

                    mean_p += l2_norm

                    output = net(img).unsqueeze(0)

                    attack_flag = (output.max(1)[1] != label).float().sum().item()
                    success_sum += attack_flag

                    if attack_flag > 0.5:
                        print('Attack Success')
                    else:
                        print('Attack Failure')

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

                    net.train()

                    for it in range(iter_num):
                        output = net(img).unsqueeze(0)

                        # loss = F.mse_loss(output, F.one_hot(target_label, class_num).float())
                        loss = F.cross_entropy(output, target_label)

                        loss.backward()

                        img_grad = torch.sign(img.grad.data)

                        img_adv = None

                        if clip_flag:
                            img_adv = clip_by_tensor(img - eta * img_grad, img_ori - clip_eps, img_ori + clip_eps)
                        else:
                            img_adv = torch.clamp(img - eta * img_grad, 0.0, 1.0)

                        img = Variable(img_adv, requires_grad=True)
                    
                    net.eval()

                    with torch.no_grad():
                        img_diff = img - img_ori

                        l2_norm = torch.norm(img_diff.view(img_diff.size()[0], -1), dim=1).item()
                        print('Perturbation: %f' % l2_norm)

                        mean_p += l2_norm

                        output = net(img).unsqueeze(0)

                        attack_flag = (output.max(1)[1] == target_label).float().sum().item()
                        success_sum += attack_flag

                        if attack_flag > 0.5:
                            print('Attack Success')
                        else:
                            print('Attack Failure')

                        '''
                        samples = img.permute(0, 2, 3, 1).data.cpu().numpy()

                        im = np.repeat(samples[0], 3, axis=2)
                        im_path = 'demo/%d_to_%d.png' % (label.item(), target_label.item())
                        print(im_path)
                        print(output)
                        plt.imsave(im_path, im)
                        '''

                if test_sum >= 270:
                    mean_p /= 270
                    break
        
        print('Mean Perturbation: %.2f' % mean_p)
        print('success_sum: %d' % success_sum)
        print('test_sum: %d' % test_sum)
        print('success_rate: %.2f%%' % (100 * success_sum / test_sum))

    elif phase == 'PGD':
        # model_path = input('输入模型文件路径，例如“./model.pth”  ')
        # iter_num = int(input('输入对抗攻击的迭代次数，例如“25”  '))
        # eta = float(input('输入对抗攻击学习率，例如“0.05”  '))
        # max_eps = float(input('输入最大扰动幅度，例如“4.0”  '))

        model_path = './models/mnist_ann_v1.pth'
        iter_num = 100
        max_eps = 0.3
        eta = 0.01
        attack_type = 'T'

        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=1,
            shuffle=False,
            drop_last=False)

        net = Net().to(device)
        net.load_state_dict(torch.load(model_path))

        mean_p2 = 0.0
        mean_pmax = 0.0
        test_sum = 0
        success_sum = 0

        for X, y in test_data_loader:
            for i in range(1, class_num):
                img, label = X.to(device), y.to(device)
                img_ori = torch.rand_like(img).copy_(img)
                img.requires_grad = True
                    
                target_label = (label + i) % class_num

                test_sum += 1

                net.train()

                for it in range(iter_num):
                    output = net(img).unsqueeze(0)

                    # loss = F.mse_loss(output, F.one_hot(target_label, class_num).float())
                    loss = F.cross_entropy(output, target_label)

                    loss.backward()

                    img_grad = torch.sign(img.grad.data)

                    perturbation = img - eta * img_grad - img_ori

                    # l2_norm = torch.norm(perturbation.view(perturbation.size()[0], -1), dim=1).item()
                    lmax_norm = torch.max(torch.abs(perturbation)).item()
                    
                    # if l2_norm > max_eps:
                        # perturbation = perturbation * max_eps / l2_norm

                    if lmax_norm > max_eps:
                        perturbation = perturbation * max_eps / lmax_norm

                    img_adv = torch.clamp(img_ori + perturbation, 0.0, 1.0)

                    img = Variable(img_adv, requires_grad=True)
                    
                net.eval()

                with torch.no_grad():
                    img_diff = img - img_ori

                    l2_norm = torch.norm(img_diff.view(img_diff.size()[0], -1), dim=1).item()
                    lmax_norm = torch.max(torch.abs(img_diff)).item()
                    
                    print('L2 Perturbation: %f' % l2_norm)
                    print('Lmax Perturbation: %f' % lmax_norm)

                    mean_p2 += l2_norm
                    mean_pmax += lmax_norm

                    output = net(img).unsqueeze(0)

                    attack_flag = (output.max(1)[1] == target_label).float().sum().item()
                    success_sum += attack_flag

                    if attack_flag > 0.5:
                        print('Attack Success')
                    else:
                        print('Attack Failure')

                    '''
                    samples = img.permute(0, 2, 3, 1).data.cpu().numpy()

                    im = np.repeat(samples[0], 3, axis=2)
                    im_path = 'demo/%d_to_%d.png' % (label.item(), target_label.item())
                    print(im_path)
                    print(output)
                    plt.imsave(im_path, im)
                    '''

                if test_sum >= 270:
                    mean_p2 /= 270
                    mean_pmax /= 270
                    break
        
        print('Mean L2 Perturbation: %.2f' % mean_p2)
        print('Mean Lmax Perturbation: %.2f' % mean_pmax)
        print('success_sum: %d' % success_sum)
        print('test_sum: %d' % test_sum)
        print('success_rate: %.2f%%' % (100 * success_sum / test_sum))

    elif phase == 'OPA':
        # model_path = input('输入模型文件路径，例如“./model.pth”  ')
        # number = int(input('输入种群的数目，例如“400”  '))
        # iter_num = int(input('输入对抗攻击的迭代次数，例如“100”  '))

        model_path = './models/mnist_ann_v1.pth'
        number = 400
        iter_num = 100
        attack_type = 'T'

        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=1,
            shuffle=False,
            drop_last=False)

        net = Net().to(device)
        net.load_state_dict(torch.load(model_path))

        mean_p = 0.0
        test_sum = 0
        success_sum = 0

        for X, y in test_data_loader:
            for i in range(1, class_num):
                img = X.numpy()
                print(img.shape)
                label = y.to(device)
                    
                target_label = (label + i) % class_num

                test_sum += 1

                net.eval()

                population = []

                for num in range(number):
                    x_val = np.random.randint(0, 28)
                    y_val = np.random.randint(0, 28)

                    grey_val = (np.random.randn() + 0.5) / 2
                    if grey_val > 1.0:
                        grey_val = 1.0
                    elif grey_val < 0.0:
                        grey_val = 0.0

                    img_p = img.copy()
                    img_p[0][0][x_val][y_val] = grey_val

                    img_p = torch.from_numpy(img_p).to(device)

                    output = net(img_p).unsqueeze(0)

                    target_prob = output.tolist()[target_label.item()]

                    population.append([x_val, y_val, grey_val, target_prob])

                for it in range(iter_num):
                    r1 = np.random.randint(0, 400)
                    while True:
                        r2 = np.random.randint(0, 400)
                        if r1 != r2:
                            break
                    while True:
                        r3 = np.random.randint(0, 400)
                        if (r1 != r3) and (r2 != r3):
                            break

                    x_val = round(population[r1][0] + (population[r2][0] - population[r3][0]) / 2)
                    if x_val

if __name__ == '__main__':
    main()