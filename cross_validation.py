# Import necessary libraries
import glob
import pandas as pd
import numpy as np
import cv2
import torch
import random
from sklearn.metrics import f1_score
from transformations import Rescale, Normalize, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from python_json_config import ConfigBuilder
from models import CNN

class CrossValidationTrainer:
    """
    A class for performing cross-validation training, validation, and testing for video data.
    """
    def __init__(self, video_paths, ann_paths, config_path):
        """
        Initializes the trainer with paths to video directories, annotation files, and a configuration file.
        """
        self.video_paths = video_paths
        self.ann_paths = ann_paths
        self.config = ConfigBuilder().parse_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = []
        self.load_data()
        self.prepare_cross_validation()

    def load_data(self):
        """
        Loads video data and annotations from the specified paths, applies transformations,
        and combines them into a single dataset.
        """
        videos = []
        dfs = []
        for path in self.video_paths:
            videos.extend(glob.glob(path + '*.mp4'))
        for path in self.ann_paths:
            dfs.append(pd.read_excel(path))
        df = pd.concat(dfs).dropna().replace('x', 3).iloc[:1000]  # Assuming data cleanup as per original script
        
        transform = transforms.Compose([Rescale(224), Normalize(), ToTensor()])
        for i, row in df.iterrows():
            if (i % 50) == 0:
                print(f'Videos loaded {i+1} out of {len(df)}')
            self.process_video(row, transform)

    def process_video(self, row, transform):
        """
        Processes a single video: loads frames, applies transformations, and adds to the dataset.
        """
        vid_name = row['Videos']
        vid_path = self.get_video_path(vid_name)    
        imgs = self.load_video_frames(vid_path, vid_name)
        
        sample = transform({'name': vid_name, 'imgs': imgs, 'v': row['V'], 'o': row['O'], 't': row['T'], 'e': row['E']})
        self.data.append(sample)

    def get_video_path(self, vid_name):
        """
        Determines the correct video path based on the video name suffix.
        """
        if vid_name[-1] == 's':
            return self.video_paths[1] + vid_name + '.mp4'
        elif vid_name[-1] == 'c':
            return self.video_paths[2] + vid_name + '.mp4'
        else:
            return self.video_paths[0] + vid_name + '.mp4'

    def load_video_frames(self, vid_path, vid_name):
        """
        Loads specified frames from a video file.
        """
        vidcap = cv2.VideoCapture(vid_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        est_tot_frames = 25  # Estimated total frames to extract
        n = 6 if vid_name[-1] in ['s', 'c'] else 5  # Frame extraction interval
        desired_frames = n * np.arange(est_tot_frames)
        imgs = []
        for j in desired_frames:
            vidcap.set(1, j-1)
            success, image = vidcap.read(1)
            if not success:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            imgs.append(image)
        vidcap.release()
        return imgs

    def prepare_cross_validation(self):
        """
        Splits the dataset into folds for cross-validation.
        """
        self.data = split_frames(self.data)
        random.seed(42)
        random.shuffle(self.data)
        n_folds = 5
        fold_samples = round(len(self.data) / n_folds)
        self.folds = [self.data[i:i + fold_samples] for i in range(0, len(self.data), fold_samples)]

    def train_and_evaluate(self):
        """
        Trains and evaluates the model across all cross-validation folds.
        """
        accs, f1s = [], []
        all_targets_v, all_targets_o, all_targets_t, all_targets_e = [], [], [], []
        all_predictions_v, all_predictions_o, all_predictions_t, all_predictions_e = [], [], [], []
        all_subjects, all_probs_v, all_probs_o, all_probs_t, all_probs_e = [], [], [], []
        
        for i, fold in enumerate(self.folds):
            # Prepare data loaders for the current fold
            train_loader, val_loader, test_loader = self.prepare_data_loaders(i)
            
            # Initialize and train the model for the current fold
            model = self.initialize_model(train_loader)
            
            # Evaluate the model on the test set
            self.evaluate_model(model, test_loader, i, accs, f1s, all_targets_v, all_targets_o,
                                all_targets_t, all_targets_e, all_predictions_v, all_predictions_o,
                                all_predictions_t, all_predictions_e, all_subjects, all_probs_v, 
                                all_probs_o, all_probs_t, all_probs_e)
        
        # Save the evaluation results to Excel files
        self.save_results(accs, f1s, all_targets_v, all_predictions_v, all_targets_o, all_predictions_o,
                          all_targets_t, all_predictions_t, all_targets_e, all_predictions_e, all_subjects,
                          all_probs_v, all_probs_o, all_probs_t, all_probs_e)

        def prepare_data_loaders(self, train_folds, val_fold, test_fold):
        """
        Prepares data loaders for training, validation, and testing datasets.
        """
        # Flatten the list of lists into a single list for train, val, and test datasets
        train_folds_flat = [item for sublist in train_folds for item in sublist]
        val_fold_flat = [item for sublist in val_fold for item in sublist]
        test_fold_flat = [item for sublist in test_fold for item in sublist]

        # Create DataLoader instances for each dataset
        train_loader = DataLoader(train_folds_flat, batch_size=8, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_fold_flat, batch_size=8, drop_last=True)
        test_loader = DataLoader(test_fold_flat, batch_size=8, drop_last=True)

        return train_loader, val_loader, test_loader

    def initialize_model(self):
        """
        Initializes the CNN model, its loss functions, and the optimizer.
        """
        # Calculate class weights for each structure based on the training dataset
        v_weights = calc_weights(self.data, 'v')
        o_weights = calc_weights(self.data, 'o')
        t_weights = calc_weights(self.data, 't')
        e_weights = calc_weights(self.data, 'e')

        # Initialize the model with the calculated weights
        model = CNN(config=self.config, v_weights=v_weights, o_weights=o_weights, t_weights=t_weights, e_weights=e_weights)
        model.to(self.device)

        # Define loss functions for each prediction target
        criterion1 = model.loss1
        criterion2 = model.loss2
        criterion3 = model.loss3
        criterion4 = model.loss4

        # Initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        return model, criterion1, criterion2, criterion3, criterion4, optimizer

    def evaluate_model(self, model, test_loader):
        """
        Evaluates the model on the test dataset and calculates accuracy and F1 score.
        """
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            all_targets, all_predictions = [], []
            for batch in test_loader:
                outputs = model(batch, self.config.time_steps)
                outputs = outputs[self.config.output_time_step, :, :]
                
                # Collect targets and predictions
                targets = np.concatenate((batch['v'], batch['o'], batch['t'], batch['e']))
                predictions = np.concatenate((torch.argmax(torch.softmax(outputs['v'], dim=1), 1),
                                              torch.argmax(torch.softmax(outputs['o'], dim=1), 1),
                                              torch.argmax(torch.softmax(outputs['t'], dim=1), 1),
                                              torch.argmax(torch.softmax(outputs['e'], dim=1), 1)))
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

            # Calculate accuracy and F1 score
            accuracy = np.mean(np.array(all_targets) == np.array(all_predictions))
            f1 = f1_score(all_targets, all_predictions, average='weighted')

        return accuracy, f1

    def save_results(self, accuracies, f1_scores):
        """
        Saves the evaluation results to an Excel file.
        """
        # Prepare a DataFrame to store results
        results_df = pd.DataFrame({
            'Fold': range(1, len(accuracies) + 1),
            'Accuracy': accuracies,
            'F1 Score': f1_scores
        })

        # Save the results to an Excel file
        results_df.to_excel('evaluation_results.xlsx', index=False)


# Example usage
if __name__ == "__main__":
    video_paths = ['D:/Glostrup DISE/Input Videos/', 'D:/Stanley DISE/Input Videos/', 'D:/Capasso DISE/Input Videos/']
    ann_paths = ['C:/Users/umaer/OneDrive/Documents/Annotations.xlsx', 
                 'C:/Users/umaer/OneDrive/Documents/Annotations Stanley.xlsx', 
                 'C:/Users/umaer/OneDrive/Documents/Annotations Capasso.xlsx']
    config_path = 'C:/Users/umaer/OneDrive/Documents/PhD/Code/sleepEndoscopy/config.json'
    
    trainer = CrossValidationTrainer(video_paths, ann_paths, config_path)
    trainer.train_and_evaluate()
