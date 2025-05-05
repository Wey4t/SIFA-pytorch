from utils import UnpairedDataset, parse_config,get_config, set_random
from transforms import Compose, Normalization, Resize
import yaml
import matplotlib.pyplot as plt
from model import SIFA
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib
import os
import configparser
from tqdm import tqdm
import wandb
matplotlib.use('Agg')

# train
def train():
    # load config
    config = "./config/train.cfg"
    config = parse_config(config)
    # load data
    print(config)
    A_path = config['train']['a_path']
    B_path = config['train']['b_path']
    batch_size = config['train']['batch_size']
    

    transform = Compose([
        Normalization(keys=['A',  'B']),
        Resize(size=(256, 256))
    ])
    trainset = UnpairedDataset(A_path, B_path, transform=transform)
    train_loader = DataLoader(trainset, batch_size,
                              shuffle=True, drop_last=True)
    # load exp_name
    exp_name = config['train']['exp_name']

    loss_cycle = []
    loss_seg = []
    # load model
    # try:
    #     wandb.init(project="sifa", name=folder_name, config=vars(args))
    #     wandb.config.update(args)  # Log all hyperparameters
    #     self.wandb_enable = True
    # except Exception as e:
    #     self.wandb_enable = False
    #     self.log(f"[Warning] wandb not initialized: {e}")

    # device = torch.device('cuda:{}'.format(config['train']['gpu']))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print("Using:",torch.cuda.get_device_name(0))
    sifa_model = SIFA(config).to(device)
    sifa_model.train()
    sifa_model.initialize()
    num_epochs = config['train']['num_epochs']
    save_epoch = num_epochs // 20

    for epoch in tqdm(range(num_epochs)):
        for i, (A, A_label, B, _) in enumerate(train_loader):

            A = A.to(device).detach()
            B = B.to(device).detach()
            A_label = A_label.to(device).detach()

            sifa_model.update_GAN(A, B)
            sifa_model.update_seg(A, B, A_label)
            loss_cyclea, loss_cycleb, segloss = sifa_model.print_loss()
            loss_cycle.append(loss_cyclea+loss_cycleb)
            loss_seg.append(segloss)
        # ddfseg_model.update_lr() #no need for changing lr
        if (epoch+1) % save_epoch == 0:
            model_dir = "save_model/" + str(exp_name)
            if(not os.path.exists(model_dir)):
                os.mkdir(model_dir)
            sifa_model.sample_image(epoch, exp_name)
            print('save model to {}/model-{}.pth'.format(model_dir, epoch+1))
            torch.save(sifa_model.state_dict(),
                       '{}/model-{}.pth'.format(model_dir, epoch+1))
            print('save model finished')
        sifa_model.update_lr()

    print('train finished')
    loss_cycle = np.array(loss_cycle)
    loss_seg = np.array(loss_seg)
    np.savez('trainingloss.npz', loss_cycle, loss_seg)
    x = np.arange(0, loss_cycle.shape[0])
    plt.figure(1)
    plt.plot(x, loss_cycle, label='cycle loss of training')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('cycle loss')
    plt.savefig('cycleloss.jpg')
    plt.close()
    plt.figure(2)
    plt.plot(x, loss_seg, label='seg loss of training')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('seg loss')
    plt.savefig('segloss.jpg')
    plt.close()
    print('loss saved')


if __name__ == '__main__':
    set_random()
    train()
