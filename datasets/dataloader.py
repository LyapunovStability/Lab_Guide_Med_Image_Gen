import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle

# Findings list for reference/compatibility
FINGDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
             'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
             'Lung Opacity', 'Pleural Effusion', 
             'Pleural Other', 'Pneumonia', 'Pneumothorax']

class GeneratorTrainDataset(Dataset):
    """
    Dataset for Generator Training (i.e., Stage 1 & Stage 2 Training; Pair-wise Strategy) and Validation.
    Input: Data dicts containing (Ref Image, Target Image, Interval Lab Data).
    """
    def __init__(self, data_pkl_path, resize=512, crop=512):
        self.data_pkl_path = data_pkl_path
        self.resize = resize
        self.crop = crop
        
        with open(self.data_pkl_path, 'rb') as f:
            self.data = pickle.load(f) # Dictionary: {patient_id: patient_data_dict}
            
        self.patient_ids = list(self.data.keys())

    def transform(self, img):
        return transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        pid = self.patient_ids[index]
        sample = self.data[pid]
        
        # 1. Lab Test Data (Variable Length Interval)
        # Shape: (L, D) -> need to convert to FloatTensor
        lab_test = torch.FloatTensor(sample['lab_test_data'])
        lab_times = torch.FloatTensor(sample['lab_test_time'])
        lab_mask = torch.FloatTensor(sample['lab_test_mask'])
        
        # 2. Reference Image
        ref_img_path = sample['reference_image_path']
        ref_img = Image.open(ref_img_path).convert('RGB')
        ref_img = self.transform(ref_img)
        ref_time = sample['reference_image_time']
        ref_abnormality = torch.FloatTensor(sample['reference_image_abnormality'])
        
        # 3. Target Image
        target_img_path = sample['target_image_path']
        target_img = Image.open(target_img_path).convert('RGB')
        target_img = self.transform(target_img)
        target_time = sample['target_image_time']
        target_abnormality = torch.FloatTensor(sample['target_image_abnormality'])
        
        return {
            "patient_id": pid,
            "lab_test": lab_test,
            "lab_times": lab_times,
            "lab_mask": lab_mask,
            "ref_img": ref_img,
            "ref_time": ref_time,
            "ref_abnormality": ref_abnormality,
            "target_img": target_img,
            "target_time": target_time,
            "target_abnormality": target_abnormality
        }

class InferenceDataset(Dataset):
    """
    Dataset for Generator Inference (i.e., Generator Inference for downstream prediction model).
    Input: Raw Test Data or Test Data with Target Points.
    """
    def __init__(self, data_pkl_path, resize=512, crop=512):
        self.data_pkl_path = data_pkl_path
        self.resize = resize
        self.crop = crop
        
        with open(self.data_pkl_path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.patient_ids = list(self.data.keys())

    def transform(self, img):
        return transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        pid = self.patient_ids[index]
        sample = self.data[pid]
        
        # 1. Full History Lab Data
        lab_test = torch.FloatTensor(sample['lab_test_data'])
        lab_times = torch.FloatTensor(sample['lab_test_time'])
        lab_mask = torch.FloatTensor(sample['lab_test_mask'])
        
        # 2. Reference Image
        ref_img_path = sample['reference_image_path']
        ref_img = Image.open(ref_img_path).convert('RGB')
        ref_img = self.transform(ref_img)
        ref_time = sample['reference_image_time']
        
        # 3. Target Time Points (Optional, might not exist in Version 0)
        target_times = sample.get('target_image_time_list', None)
        if target_times is not None:
            target_times = torch.FloatTensor(target_times)
            
        return {
            "patient_id": pid,
            "lab_test": lab_test,
            "lab_times": lab_times,
            "lab_mask": lab_mask,
            "ref_img": ref_img,
            "ref_time": ref_time,
            "target_times": target_times # Can be None
        }

class CollateFunc:
    def __init__(self, dim=0):
        self.dim = dim

    def pad_tensor(self, vec, pad, dim):
        """
        Pads a tensor to length 'pad' along dimension 'dim'.
        Returns padded_vec and mask.
        """
        if isinstance(vec, np.ndarray):
            vec = torch.from_numpy(vec)
        
        original_shape = vec.shape
        pad_size = list(original_shape)
        pad_size[dim] = pad - vec.size(dim)
        
        # Pad with zeros
        padded_vec = torch.cat([vec, torch.zeros(*pad_size, dtype=vec.dtype, device=vec.device)], dim=dim)
        
        # Create mask (1 for real data, 0 for padded)
        # We assume we just need to know which time steps are padded.
        
        return padded_vec

    def __call__(self, batch):
        # batch is a list of dicts from __getitem__
        
        # 1. Pad Lab Tests
        # Find max length in this batch
        max_len = max([x['lab_test'].shape[0] for x in batch])
        
        lab_tests_padded = []
        lab_masks_padded = [] # Mask for original missingness + padding
        lab_times_padded = []
        lab_lengths = [] # Keep track of original sequence lengths
        
        for x in batch:
            # Pad Lab Data
            curr_len = x['lab_test'].shape[0]
            lab_lengths.append(curr_len)
            pad_len = max_len - curr_len
            
            # Pad features (L, D)
            padded_lab = torch.cat([x['lab_test'], torch.zeros(pad_len, x['lab_test'].shape[1])], dim=0)
            lab_tests_padded.append(padded_lab)
            
            # Pad Mask (L, D) - 0 means missing or padded
            # Original mask: 1=present, 0=missing
            # Padding: 0
            padded_mask = torch.cat([x['lab_mask'], torch.zeros(pad_len, x['lab_mask'].shape[1])], dim=0)
            lab_masks_padded.append(padded_mask)
            
            # Pad Times (L,)
            # Pad with -1 or similar to indicate invalid time? Or just 0?
            padded_time = torch.cat([x['lab_times'], torch.zeros(pad_len)], dim=0)
            lab_times_padded.append(padded_time)
            
        # Stack
        batch_lab_test = torch.stack(lab_tests_padded)
        batch_lab_mask = torch.stack(lab_masks_padded)
        batch_lab_times = torch.stack(lab_times_padded)
        batch_lab_lengths = torch.tensor(lab_lengths, dtype=torch.long)
        
        # 2. Stack other fields
        patient_ids = [x['patient_id'] for x in batch]
        batch_ref_img = torch.stack([x['ref_img'] for x in batch])
        batch_ref_time = torch.tensor([x['ref_time'] for x in batch])
        
        output = {
            "patient_id": patient_ids,
            "lab_test": batch_lab_test,
            "lab_mask": batch_lab_mask,
            "lab_times": batch_lab_times,
            "lab_lengths": batch_lab_lengths, # Explicit length for masking in model
            "ref_img": batch_ref_img,
            "ref_time": batch_ref_time,
        }
        
        # 3. Handle Target Fields (Only for Training)
        if 'target_img' in batch[0]:
            batch_target_img = torch.stack([x['target_img'] for x in batch])
            batch_target_time = torch.tensor([x['target_time'] for x in batch])
            batch_target_abnormality = torch.stack([x['target_abnormality'] for x in batch])
            batch_ref_abnormality = torch.stack([x['ref_abnormality'] for x in batch])
            
            output.update({
                "target_img": batch_target_img,
                "target_time": batch_target_time,
                "target_abnormality": batch_target_abnormality,
                "ref_abnormality": batch_ref_abnormality
            })
            
        # 4. Handle Target Time List (Only for Inference)
        if 'target_times' in batch[0] and batch[0]['target_times'] is not None:
             # Since target_times might be variable length list, we might keep as list
             # Or pad if we need a tensor. Keeping as list of tensors is safer for inference.
             # NOTE: We intentionally keep this as a list of tensors (not stacked) because
             # different patients may have different numbers of target time points during inference.
             # Downstream inference scripts should handle this list structure (e.g. by iterating or flattening).
             batch_target_times = [x['target_times'] for x in batch]
             output.update({"target_times": batch_target_times})

        return output
