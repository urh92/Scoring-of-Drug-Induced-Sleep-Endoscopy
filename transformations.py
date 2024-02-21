# Import necessary libraries
import cv2
import torch
import numpy as np

# Define a Rescale transformation class
class Rescale(object):
    """
    A transformation class to rescale images to a specified output size.
    """
    def __init__(self, output_size):
        # Ensure the output size is either an integer or a tuple
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size  # Set the desired output size
        
    def __call__(self, sample):
        # Extract data from the sample
        name, imgs, v, o, t, e = sample['name'], sample['imgs'], sample['v'], sample['o'], sample['t'], sample['e']
        imgs_scaled = []  # Initialize a list to hold scaled images
        
        # Scale each image in the sample
        for img in imgs:
            # If output_size is an integer, resize image to square of that size
            if isinstance(self.output_size, int):  
                imgs_scaled.append(cv2.resize(img, (self.output_size, self.output_size)))
            else: 
                # If output_size is a tuple, resize image to that size (width, height)
                imgs_scaled.append(cv2.resize(img, self.output_size))
        
        # Return the transformed sample with scaled images and original data
        return {'name': name, 'imgs': imgs_scaled, 'v': v, 'o': o, 't': t, 'e': e}
        

# Define a Normalize transformation class
class Normalize(object):
    """
    A transformation class to normalize image pixel values to the range [0, 1].
    """
    def __call__(self, sample):
        # Extract data from the sample
        name, imgs, v, o, t, e = sample['name'], sample['imgs'], sample['v'], sample['o'], sample['t'], sample['e']
        imgs_normalized = []  # Initialize a list to hold normalized images
        
        # Normalize each image in the sample
        for img in imgs:
            imgs_normalized.append(img / 255)  # Divide pixel values by 255
        
        # Return the transformed sample with normalized images and original data
        return {'name': name, 'imgs': imgs_normalized, 'v': v, 'o': o, 't': t, 'e': e}

# Define a ToTensor transformation class
class ToTensor(object):
    """
    A transformation class to convert images and labels to PyTorch tensors.
    """
    def __call__(self, sample):
        # Extract data from the sample
        name, imgs, v, o, t, e = sample['name'], sample['imgs'], sample['v'], sample['o'], sample['t'], sample['e']
        imgs_tensor = []  # Initialize a list to hold tensor images
        
        # Convert each image in the sample to a tensor
        for img in imgs:
            # If the image is grayscale (2D), add an extra dimension to make it 3D
            if len(img.shape) < 3:
                img = np.expand_dims(img, 2)
            # Transpose the image dimensions (HWC to CHW) and convert to a tensor
            imgs_tensor.append(torch.from_numpy(img.transpose((2, 0, 1))))
        
        # Convert labels to tensors and specify the tensor type for use with CUDA if available
        return {'name': name, 'imgs': imgs_tensor, 'v': torch.tensor(v).type(torch.cuda.LongTensor), 
                'o': torch.tensor(o).type(torch.cuda.LongTensor),
                't': torch.tensor(t).type(torch.cuda.LongTensor), 
                'e': torch.tensor(e).type(torch.cuda.LongTensor)}
