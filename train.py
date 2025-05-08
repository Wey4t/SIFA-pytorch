from utils import UnpairedDataset, parse_config, get_config, set_random
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
    
    # Initialize wandb
    exp_name = config['train']['exp_name']
    wandb.init(project="sifa", name=exp_name, config=config)
    
    # load data
    print(config)
    A_path = config['train']['a_path']
    B_path = config['train']['b_path']
    batch_size = config['train']['batch_size']
    
    transform = Compose([
        Normalization(keys=['A', 'B']),
        Resize(size=(256, 256))
    ])
    trainset = UnpairedDataset(A_path, B_path, transform=transform)
    train_loader = DataLoader(trainset, batch_size,
                              shuffle=True, drop_last=True)
    
    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    
    # model initialization
    sifa_model = SIFA(config).to(device)
    sifa_model.train()
    sifa_model.initialize()
    
    # training parameters
    num_epochs = config['train']['num_epochs']
    save_epoch = num_epochs // 400
    
    # Track loss values for plotting
    loss_cycle = []
    loss_seg = []
    
    for epoch in tqdm(range(num_epochs)):
        epoch_cycle_loss = 0
        epoch_seg_loss = 0
        batch_count = 0
        print(f"Epoch {epoch+1} with {len(train_loader)/batch_size} steps")
        for i, (A, A_label, B, _) in enumerate(train_loader):
            batch_count += 1
            
            A = A.to(device).detach()
            B = B.to(device).detach()
            A_label = A_label.to(device).detach()

            sifa_model.update_GAN(A, B)
            sifa_model.update_seg(A, B, A_label)
            
            loss_cyclea, loss_cycleb, segloss = sifa_model.print_loss()
            current_cycle_loss = loss_cyclea + loss_cycleb
            loss_cycle.append(current_cycle_loss)
            loss_seg.append(segloss)
            
            epoch_cycle_loss += current_cycle_loss
            epoch_seg_loss += segloss
            
            # Log batch-level metrics
            wandb.log({
                "batch/cycle_loss": current_cycle_loss,
                "batch/seg_loss": segloss,
                "batch/cycle_loss_a": loss_cyclea,
                "batch/cycle_loss_b": loss_cycleb,
            })
        
        # Log epoch-level metrics
        wandb.log({
            "epoch": epoch,
            "epoch/cycle_loss": epoch_cycle_loss / batch_count,
            "epoch/seg_loss": epoch_seg_loss / batch_count,
        })
        
        # Save model periodically
        if (epoch+1) % save_epoch == 0:
            model_dir = "save_model/" + str(exp_name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                
            # Save sample images
            sifa_model.sample_image(epoch, exp_name)
            
            # Save model checkpoint
            model_path = '{}/model-{}.pth'.format(model_dir, epoch+1)
            print('save model to', model_path)
            torch.save(sifa_model.state_dict(), model_path)
            
  
            
            # If sample_image saves images to disk, log them to wandb
            # Assuming sample_image saves to a path like "samples/{exp_name}_{epoch}.jpg"
            try:
                sample_path = f"samples/{exp_name}_{epoch}.jpg"
                if os.path.exists(sample_path):
                    wandb.log({"sample_images": wandb.Image(sample_path)})
            except:
                print("Could not log sample images to wandb")
                
            print('save model finished')
            
        # Update learning rate
        sifa_model.update_lr()

    print('train finished')
    
    # Save final loss arrays
    loss_cycle_array = np.array(loss_cycle)
    loss_seg_array = np.array(loss_seg)
    np.savez('trainingloss.npz', loss_cycle_array, loss_seg_array)
    
    # Create and save plots
    x = np.arange(0, len(loss_cycle))
    
    plt.figure(1)
    plt.plot(x, loss_cycle_array, label='cycle loss of training')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('cycle loss')
    plt.savefig('cycleloss.jpg')
    wandb.log({"final_plots/cycle_loss": wandb.Image('cycleloss.jpg')})
    plt.close()
    
    plt.figure(2)
    plt.plot(x, loss_seg_array, label='seg loss of training')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('seg loss')
    plt.savefig('segloss.jpg')
    wandb.log({"final_plots/seg_loss": wandb.Image('segloss.jpg')})
    plt.close()
    
    print('loss saved')
    
    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    set_random()
    train()