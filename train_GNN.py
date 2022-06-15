import torch
import matplotlib.pyplot as plt
import utils
import os
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from glob import glob

def train(model, train_dataloader, val_loader, epochs=5000, lr=1e-5, epochs_til_summary=10,
          epochs_til_checkpoint=100, model_dir='./log', loss_fn=None,
          wandb=None, prefix_model_dir='', gt=None,graph_update_fn=None,
          output_type='v',output_steps=1, lr_schedule=False,decay_factor=0.0001,
          log_iter=None,checkpoints=None,device='cuda:0'):
    
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    
    if not(checkpoints==None):
        PATH = sorted(glob('{}/checkpoints/model_epoch*.pth'.format(checkpoints)))
        print(PATH[-1])
        check = torch.load(PATH[-1],map_location=device)
        model.load_state_dict(check['model_state_dict'])
        optim.load_state_dict(check['optimizer_state_dict'])
        del check
        
    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=np.exp(np.log(decay_factor)/epochs*20), last_epoch=-1)
    model_dir_postfixed = os.path.join(model_dir, prefix_model_dir)
    checkpoints_dir = os.path.join(model_dir_postfixed, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)
    summaries_dir = os.path.join(model_dir_postfixed, 'summaries')
    utils.cond_mkdir(summaries_dir)
    writer = SummaryWriter(summaries_dir)

    total_step = 0
   
    total_steps = 0

    with tqdm(total=(len(train_dataloader)) * epochs) as pbar:
        for epoch in range(epochs):
                            
            if (not epoch % epochs_til_checkpoint) or total_steps%10000 == 0:
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optim.state_dict()}, 
                           os.path.join(checkpoints_dir, 'model_epoch_%010d.pth' % total_steps))
                if wandb:
                    wandb.log({"store_checkpoints": 1},step=total_steps)
                print('storing model...........\n')

            # for j in range(len(train_dataloader)):
            for data in train_dataloader:
                start_time = time.time()
                # data = train_dataloader.ask(j).clone()
                data = data.cuda()
                
                optim.zero_grad()
                train_loss = 0.
                losses = {}

                # calculate model output and loss
                output = model(data,train=True)
                loss_dir = loss_fn(output, data)

                for loss_name, loss in loss_dir.items():
                    single_loss = loss
                    train_loss += single_loss

                    if loss_name in losses:
                        losses[loss_name] += loss
                    else:
                        losses[loss_name] = loss

                # update weights
                train_loss.backward()
                optim.step()
                                                    
                writer.add_scalar("total_train_loss", train_loss, total_steps)
                
                total_steps+=1
                pbar.update(1)
            
            if epoch % epochs_til_summary == 0 or total_steps%30000 == 0:
                tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                model.eval()
                log_iter(model, val_loader, output_steps, graph_update_fn, wandb,
                            writer, total_steps, epoch, output_type, prefix="val")  
                model.train()
                
            if lr_schedule and epoch%20==0 and epoch!= 0:
                scheduler.step()
                print("update_learning_rate")
                if wandb!=None:
                    wandb.log({"lr": scheduler.get_last_lr()},step=total_steps)