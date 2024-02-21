# Import necessary libraries for data processing and model training
import glob
import pandas as pd
import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from python_json_config import ConfigBuilder
from transformations import Rescale, Normalize, ToTensor
from models import CNN

class VideoModelTrainer:
    """
    A class to handle loading of video data, training a CNN model, and evaluating its performance.
    """
    def __init__(self, video_path, ann_path, config_path):
        """
        Initializes the VideoModelTrainer with paths for video data, annotations, and configuration.

        Parameters:
        - video_path: Path to the directory containing input videos.
        - ann_path: Path to the annotations Excel file.
        - config_path: Path to the JSON configuration file.
        """
        self.video_path = video_path
        self.ann_path = ann_path
        self.config = ConfigBuilder().parse_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN(config=self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.val_losses = []

    def load_data(self):
        """
        Loads video data and annotations, applies transformations, and splits into training, validation, and test sets.
        """
        df = pd.read_excel(self.ann_path).dropna().replace('x', 3)
        videos = glob.glob(self.video_path + '*.mp4')
        vid_names = [video.split('/')[-1][:-4] for video in videos]
        
        # Define subsets for validation and test data
        val_data_vids = ('v23', 'v24', 'v25', 'v26', 'v27')
        test_data_vids = ('v28', 'v29', 'v30', 'v31', 'v32')

        self.train_data, self.val_data, self.test_data = [], [], []
        transform = transforms.Compose([Rescale(224), Normalize(), ToTensor()])
        
        # Process each video file
        for i, row in df.iterrows():
            if (i % 50) == 0:
                print(f'Videos loaded {i+1} out of {len(df)}')
            vid_name, v, o, t, e = row['Videos'], row['V'], row['O'], row['T'], row['E']
            imgs = self.load_video_frames(self.video_path + vid_name + '.mp4')
            
            sample = transform({'name': vid_name, 'imgs': imgs, 'v': v, 'o': o, 't': t, 'e': e})
            
            if sample['name'].startswith(val_data_vids):
                self.val_data.append(sample)
            elif sample['name'].startswith(test_data_vids):
                self.test_data.append(sample)
            else:
                self.train_data.append(sample)

        self.train_loader = DataLoader(self.train_data, batch_size=8, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.val_data, batch_size=8, drop_last=True)
        self.test_loader = DataLoader(self.test_data, batch_size=8, drop_last=True)

    def load_video_frames(self, vid_path):
        """
        Loads and extracts frames from a given video path.

        Parameters:
        - vid_path: Path to the video file.

        Returns:
        A list of loaded video frames.
        """
        vidcap = cv2.VideoCapture(vid_path)
        success, image = vidcap.read()
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        est_tot_frames = 25  # Arbitrary upper bound of frames
        desired_frames = np.arange(0, est_tot_frames * fps, fps // 5)
        imgs = [image]
        for frame_number in desired_frames[1:]:
            vidcap.set(1, frame_number)
            success, image = vidcap.read()
            if not success:
                break
            imgs.append(image)
        vidcap.release()
        return imgs

    def train(self, epochs):
        """
        Trains the CNN model using the training set and evaluates on the validation set.

        Parameters:
        - epochs: Number of training epochs.
        """
        for epoch in range(epochs):
            self.run_epoch(self.train_loader, training=True)
            val_mean_loss = self.run_epoch(self.val_loader, training=False)
            print(f'Validation loss: {val_mean_loss}')
            if not any(x < val_mean_loss for x in self.val_losses):
                self.save_checkpoint(epoch)
            self.val_losses.append(val_mean_loss)

    def run_epoch(self, loader, training=True):
        """
        Runs a single epoch of training or validation.

        Parameters:
        - loader: DataLoader for the current set (training or validation).
        - training: Boolean indicating if the model is being trained or validated.

        Returns:
        The mean loss for the epoch.
        """
        if training:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        for i, batch in enumerate(loader, 0):
            loss = self.process_batch(batch, training)
            running_loss += loss.item()
        return running_loss / (i + 1)

    def process_batch(self, batch, training):
        """
        Processes a single batch - computes the model outputs and loss, and performs backpropagation if training.

        Parameters:
        - batch: The current batch of data.
        - training: Boolean indicating if backpropagation should be performed.

        Returns:
        The loss for the batch.
        """
        v, o, t, e = batch['v'], batch['o'], batch['t'], batch['e']
        v_pred, o_pred, t_pred, e_pred = self.model(batch, 25)  # Assuming a fixed time_step value for simplicity
        mid_time_step = 25 // 2 + 1
        loss = sum([self.model.loss1(v_pred[mid_time_step, :, :]), self.model.loss2(o_pred[mid_time_step, :, :]),
                    self.model.loss3(t_pred[mid_time_step, :, :]), self.model.loss4(e_pred[mid_time_step, :, :])])

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss

    def save_checkpoint(self, epoch):
        """
        Saves the model checkpoint.

        Parameters:
        - epoch: The current epoch number.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.config.checkpoint_dir + f'checkpoint_epoch_{epoch}.tar')

# Example usage
if __name__ == "__main__":
    video_model_trainer = VideoModelTrainer(
        video_path='D:/Glostrup DISE/Input Videos/',
        ann_path='C:/Users/umaer/OneDrive/Documents/Annotations.xlsx',
        config_path='C:/Users/umaer/OneDrive/Documents/PhD/Code/sleepEndoscopy/config.json'
    )
    video_model_trainer.load_data()
    video_model_trainer.train(epochs=10)
