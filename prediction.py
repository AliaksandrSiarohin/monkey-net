import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from logger import Logger, Visualizer
import numpy as np
import imageio

from modules.prediction_module import PredictionModule
from augmentation import SelectRandomFrames, VideoToTensor
from tqdm import trange
from frames_dataset import FramesDataset
from sync_batchnorm import DataParallelWithCallback


class KPDataset(Dataset):
    """Dataset of detected keypoints"""

    def __init__(self, keypoints_array, num_frames):
        self.keypoints_array = keypoints_array
        self.transform = SelectRandomFrames(consequent=True, select_appearance_frame=False)
        self.transform.number_of_frames = num_frames

    def __len__(self):
        return len(self.keypoints_array)

    def __getitem__(self, idx):
        keypoints = self.keypoints_array[idx]
        selected = self.transform(keypoints)
        selected = {k: np.concatenate([v[k][0] for v in selected], axis=0) for k in selected[0].keys()}
        return selected


def prediction(config, generator, kp_detector, checkpoint, log_dir):
    dataset = FramesDataset(is_train=True, transform=VideoToTensor(), **config['dataset_params'])
    log_dir = os.path.join(log_dir, 'prediction')
    png_dir = os.path.join(log_dir, 'png')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='prediction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    print("Extracting keypoints...")

    kp_detector.eval()
    generator.eval()

    keypoints_array = []

    prediction_params = config['prediction_params']

    for it, x in tqdm(enumerate(dataloader)):
        if it == prediction_params['train_size']:
            break
        with torch.no_grad():
            keypoints = []
            for i in range(x['video_array'].shape[2]):
                kp = kp_detector(x['video_array'][:, :, i:(i + 1)])
                kp = {k: v.data.cpu().numpy() for k, v in kp.items()}
                keypoints.append(kp)
            keypoints_array.append(keypoints)

    predictor = PredictionModule(num_kp=config['model_params']['common_params']['num_kp'],
                                 kp_variance=config['model_params']['common_params']['kp_variance'],
                                 **prediction_params['rnn_params']).cuda()

    num_epochs = prediction_params['num_epochs']
    lr = prediction_params['lr']
    bs = prediction_params['batch_size']
    num_frames = prediction_params['num_frames']
    init_frames = prediction_params['init_frames']

    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=50)

    kp_dataset = KPDataset(keypoints_array, num_frames=num_frames)

    kp_dataloader = DataLoader(kp_dataset, batch_size=bs)

    print("Training prediction...")
    for _ in trange(num_epochs):
        loss_list = []
        for x in kp_dataloader:
            x = {k: v.cuda() for k, v in x.items()}
            gt = {k: v.clone() for k, v in x.items()}
            for k in x:
                x[k][:, init_frames:] = 0
            prediction = predictor(x)

            loss = sum([torch.abs(gt[k][:, init_frames:] - prediction[k][:, init_frames:]).mean() for k in x])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.detach().data.cpu().numpy())

        loss = np.mean(loss_list)
        scheduler.step(loss)

    dataset = FramesDataset(is_train=False, transform=VideoToTensor(), **config['dataset_params'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    print("Make predictions...")
    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            x['video_array'] = x['video_array'][:, :, :num_frames]
            kp_init = kp_detector(x['video_array'])
            for k in kp_init:
                kp_init[k][:, init_frames:] = 0

            kp_appearance = kp_detector(x['video_array'][:, :, :1])

            kp_video = predictor(kp_init)
            for k in kp_video:
                kp_video[k][:, :init_frames] = kp_init[k][:, :init_frames]
            if 'var' in kp_video:
                kp_video['var'] = kp_init['var'][:, (init_frames - 1):init_frames].repeat(1, kp_video['var'].shape[1],
                                                                                          1, 1, 1)
            out = {'video_prediction': [], 'video_deformed': []}
            for i in range(x['video_array'].shape[2]):
                kp_target = {k: v[:, i:(i + 1)] for k, v in kp_video.items()}
                kp_dict_part = {'kp_video': kp_target, 'kp_appearance': kp_appearance}
                out_part = generator(x['video_array'][:, :, :1], **kp_dict_part)
                out['video_prediction'].append(out_part['video_prediction'])
                out['video_deformed'].append(out_part['video_deformed'])

            out['video_prediction'] = torch.cat(out['video_prediction'], dim=2)
            out['video_deformed'] = torch.cat(out['video_deformed'], dim=2)
            out['kp_video'] = kp_video
            out['kp_appearance'] = kp_appearance

            x['appearance_array'] = x['video_array'][:, :, :1]

            out_video_batch = out['video_prediction'].data.cpu().numpy()
            out_video_batch = np.concatenate(np.transpose(out_video_batch, [0, 2, 3, 4, 1])[0], axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * out_video_batch).astype(np.uint8))

            image = Visualizer(**config['visualizer_params']).visualize_reconstruction(x, out)
            image_name = x['name'][0] + prediction_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), image)

            del x, kp_video, kp_appearance, out
