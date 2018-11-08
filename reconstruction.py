import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
from modules.losses import reconstruction_loss
import numpy as np
import imageio

def reconstruction(config, generator, kp_extractor, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_extractor=kp_extractor)
    else:
        raise AttributeError("Checkpoint should be specified for mode='test'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_list = []
    generator.eval()
    kp_extractor.eval()

    
    for it, x in tqdm(enumerate(dataloader)):
        if it == 1000:
           break
        with torch.no_grad():               
            kp_appearance = kp_extractor(x['video_array'][:, :, :1])                    
            kp_video = []
            out = {'video_prediction': [], 'video_deformed': []}
            for i in range(x['video_array'].shape[2]):
                     kp_target = kp_extractor(x['video_array'][:, :, i:(i+1)])
                     kp_video.append(kp_target)  
                     kp_dict_part = {'kp_video': kp_target, 'kp_appearance': kp_appearance}           
                     out_part = generator(x['video_array'][:, :, :1], **kp_dict_part)
                     out['video_prediction'].append(out_part['video_prediction'])
                     out['video_deformed'].append(out_part['video_deformed'])

            out['video_prediction'] = torch.cat(out['video_prediction'], dim=2)
            out['video_deformed'] =  torch.cat(out['video_deformed'], dim=2)
            out['kp_video'] = {k: torch.cat(list(map(lambda x: x[k], kp_video)), dim=1) for k in kp_appearance.keys()}
            out['kp_appearance'] = kp_appearance

            x['appearance_array'] = x['video_array'][:, :, :1]
            
            #Store to .png for evaluation
            out_video_batch = out['video_prediction'].data.cpu().numpy()
            out_video_batch = np.concatenate(np.transpose(out_video_batch, [0, 2, 3, 4, 1])[0], axis=1) 
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * out_video_batch).astype(np.uint8))  
        
            image = Visualizer().visualize_reconstruction(x, out)
            image_name = x['name'][0] + config['transfer_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), image)

            loss = reconstruction_loss(out['video_prediction'].cpu(), x['video_array'].cpu(),
                                       config['loss_weights']['reconstruction'][0])
            loss_list.append(loss.data.cpu().numpy())
            del x, kp_video, kp_appearance, out, loss

    print ("Reconstruction loss: %s" % np.mean(loss_list))
