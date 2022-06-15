
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
import dataio
import skimage

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def log_iter_2d_irregular(timestep, model, loader, output_steps, graph_update_fn, wandb, writer, total_step, epoch,
             output_type, prefix="train",dim=1):
    if loader==None:
        return
    if not (isinstance(loader,list)):
        loaders = [loader]
    else:
        loaders = loader
    num_traj = len(loaders)
    fig = plt.figure(figsize=[3*5*num_traj,5*8])
    mse_for_each_timestep = plt.figure(figsize=[3*num_traj,3])
    mse = []
    relative_l2 = []
    for k,loader in enumerate(loaders):   
        for i,data in enumerate(loader):
            coords = data.coords[0]
            cell = data.cell[0]

            if i==0: # initialize all variables
                iter = len(loader)
                init_1 = data
                prediction = torch.zeros([iter*output_steps+1,init_1.x_eval.shape[0]])
                grid = torch.zeros([iter*output_steps+1,init_1.x_eval.shape[0]])
                gt_imag = torch.zeros([iter*output_steps+1,init_1.x_eval.shape[0]])
                
                prediction[0,:]=init_1.x_eval[:,0].detach()
                gt_imag[0,:]=init_1.x_eval[:,0].detach()

            # save the current output and generate the next output 
            init_2 = init_1.clone()
            y_old = init_1.x_eval[:,0].clone()
            with torch.no_grad():
                output = model(init_2.cuda())

            # keep track of current step
            i=i*output_steps
            output.eval = output.eval.detach().cpu()
            output.x = output.x.detach().cpu()

            # if output is amplitude
            if output_type=="x":
                prediction[i+1:i+1+output_steps,:] = output.eval.clone().permute(1,0)
                gt_imag[i+1:i+1+output_steps,:] = data.gt.permute(1,0)
                init_1 = graph_update_fn(output,init_1,train=False)

            # elif output is velocity
            elif output_type=="v":
                prediction[i+1:i+1+output_steps,:] = torch.cumsum(y_old.unsqueeze(-1)+output.eval[:,:].clone(), dim=1).permute(1, 0).detach() 
                gt_imag[i+1:i+1+output_steps,:] = data.gt.permute(1,0)
                init_1 = graph_update_fn(output,init_1,train=False)
                #only for single timestep prediction 
        if timestep==1:
            time_list = np.concatenate([[1,2,3,4],np.linspace(5,i,10)])
        else:
            time_list = np.concatenate([[1,2,3,4],np.linspace(5,int(i/timestep)-1,10)])
              
        for index,t in enumerate(time_list):
            t = int(np.floor(t))
            plt.figure(fig.number)
            fig.add_subplot(len(time_list),3*len(loaders),3*k+1+3*len(loaders)*index)
            plt.tricontourf(coords[:, 0], coords[:, 1], cell, (prediction[t,:]), levels=50,cmap='jet')
            plt.triplot(coords[:, 0], coords[:, 1], cell,'-b',linewidth=0.1)
            plt.title('prediction_{}'.format(t))
            plt.colorbar()
            fig.add_subplot(len(time_list),3*len(loaders),3*k+2+3*len(loaders)*index)
            plt.tricontourf(coords[:, 0], coords[:, 1], cell, (gt_imag[t,:]), levels=50,cmap='jet')
            plt.triplot(coords[:, 0], coords[:, 1], cell,'-b',linewidth=0.1)
            plt.title('ground_truth_{}'.format(t))
            plt.colorbar()
            fig.add_subplot(len(time_list),3*len(loaders),3*k+3+3*len(loaders)*index)
            plt.tricontourf(coords[:, 0], coords[:, 1], cell, (prediction[t,:]-gt_imag[t,:]), levels=50,cmap='jet')
            plt.triplot(coords[:, 0], coords[:, 1], cell,'-b',linewidth=0.1)
            plt.title('diffV_{}'.format(t))
            plt.colorbar()
            
            plt.figure(mse_for_each_timestep.number)
            mse_for_each_timestep.add_subplot(1,num_traj,k+1)
            plt.plot(torch.mean((prediction-gt_imag)**2,dim=1),"*-")
            plt.title('gt_mse_for_each_timestep')
            plt.yscale('log')
            
            plt.tight_layout()
    
        # mse plots
        plt.figure(mse_for_each_timestep.number)
        mse_for_each_timestep.add_subplot(1,num_traj,k+1)
        plt.plot(torch.mean((prediction-gt_imag)**2,dim=1),"*-",label="prdiction_vs_gt")
        plt.title('mse_for_each_timestep')
        plt.yscale('log')
        plt.legend()

        # residual plot
        # mse and relative l2
        mse.append(((prediction-gt_imag)**2).mean().cpu().numpy())
        relative_l2.append(torch.norm((prediction-gt_imag).view(-1))/torch.norm((gt_imag).view(-1)).cpu().numpy())
        del prediction
        del gt_imag

    average_mse = sum(mse)/len(mse)
    average_l2 = sum(relative_l2)/len(relative_l2)
    if writer:
        writer.add_figure(prefix+'_iter_animiate', fig, global_step=total_step)
        writer.add_scalar(prefix+'_iter_animiate_mse',average_mse,global_step=total_step)
    if wandb:
        wandb.log({prefix+'_iter_animiate': wandb.Image(fig)},step=total_step)
        wandb.log({prefix+'_iter_animiate_mse': average_mse},step=total_step)
        wandb.log({prefix+'_iter_animiate_relative_l2': average_l2},step=total_step)
        wandb.log({prefix+'_iter_mse_for_each_timestep': wandb.Image(mse_for_each_timestep)},step=total_step)


def summary_autodecoder(sample,latent_dim,irregular_mesh, model, model_input, gt, model_output, writer, total_steps, prefix='train_',wandb=None):
    image_resolution = [300,300]
    mask = torch.zeros(image_resolution)
    mask.view(-1)[gt['mask']]=1
    model_input['coords'] = model_input['coords_fine']
    with torch.no_grad():
        model_output = model(model_input)
    pred_img = dataio.lin2img(model_output['model_out'], image_resolution)
    gt_img = dataio.lin2img(gt['img_fine'], image_resolution)
    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)
    psnr = write_psnr(pred_img, gt_img, writer, total_steps, prefix + 'image_')
    if wandb:
        wandb.log({prefix+'psnr': psnr},step=total_steps)
        wandb.log({prefix+'gt_vs_pred': wandb.Image(output_vs_gt)},step=total_steps)
        wandb.log({prefix+'mask': wandb.Image(mask)},step=total_steps)

    if sample:
        latent = (torch.randn(model_output['model_out'].shape[0],latent_dim).cuda())*torch.sqrt(model.varience)+model.mean
        with torch.no_grad():
            model_output = model(model_input,latent=latent)
        pred_img = dataio.lin2img(model_output['model_out'], image_resolution)
        writer.add_image(prefix + 'sampled', make_grid(pred_img, scale_each=False, normalize=True),
                        global_step=total_steps)
        if wandb:
            wandb.log({prefix+'sampled': wandb.Image(pred_img)},step=total_steps)
    if irregular_mesh and total_steps%4000==0:
        for resol in ['high','low']:
            f = np.load('./data/meshes/fish14_{}.npy'.format(resol), allow_pickle=True)[()]
            vert = f['vert']
            vert = vert*2-1 #coords convert to be range from -1 and 1
            tri = f['tri'].astype(np.int32)
            model_input['coords'] = (torch.tensor(vert).unsqueeze(0)).repeat(9,1,1).float().cuda()
            latent = latent[0:9,:]
            with torch.no_grad():
                model_output = model(model_input,latent=latent)
            fig = plt.figure(figsize=[15,15])
            for i in range(3):
                for j in range(3):
                    fig.add_subplot(3,3,i*3+j+1)
                    plt.triplot(vert[:,0],vert[:,1],tri,linewidth=0.1)
                    plt.tricontourf(vert[:,0],vert[:,1],tri,(model_output['model_out'][i*3+j,:,0].cpu().numpy()*2)+1,levels=40)
            if wandb:
                wandb.log({prefix+'irregular_sampled_{}'.format(resol): wandb.Image(fig)},step=total_steps)


def write_psnr(pred_img, gt_img, writer, iter, prefix):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs = list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        trgt = (trgt / 2.) + 0.5

        psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)

        psnrs.append(psnr)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)
    return np.mean(psnrs)
