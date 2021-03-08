"""
Pytorch dataloaders for example


"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils.evaluation import load_nifti_itk

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class Oral_dataset_offline(Dataset):
    def __init__(self, root_dir, list_path, offline=False, transform=None, train=True):
        """
        Give patch file path, corresponding surv time and status
        :param list_path:
        """

        self.list_path = list_path
        self.random = train
        self.transform = transform
        self.root_dir = root_dir
        self.train = train
        self.offline = offline

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, idx):

        pid_run_name = self.list_path[idx]

        pid_name = pid_run_name.split('_')[0]


        # print(pid_name)

        full_path = str(os.path.join(self.root_dir, pid_name+'.nii.gz'))
        full_mask_path = full_path.replace('img', 'mask')
        full_dist_path = full_path.replace('img', 'dist_map')


        # Must make sure the load_nifit_itk is the same before
        img_data, spacing = load_nifti_itk(full_path)
        mask_data, spacing = load_nifti_itk(full_mask_path)
        dist_data, spacing = load_nifti_itk(full_dist_path)

        img_data[img_data > 30.0] = 30.0
        mask_data[img_data < 2.5] = 0.0
        dist_data /= 121.0
        img_data = img_data / 30.0


        img_data = np.asarray(img_data, dtype=np.float32)
        mask_data = np.asarray(mask_data, dtype=np.float32)
        dist_data = np.asarray(dist_data, dtype=np.float32)

        sample = {'image': img_data, 'mask': [mask_data, dist_data], 'surv_label':  pid_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Oral_dataloader_offline():
    def __init__(self, root_dir, data_path, batch_size, test_mode='T'):


        testdataset = Oral_dataset_offline(root_dir, data_path, train=False,
                                           transform=transforms.Compose([
                                               # TumorCenterCrop(output_size=(64, 64, 96)),
                                               ToTensor(test_mode)]))
        testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                        worker_init_fn=worker_init_fn)

        self.dataloader = testloader

    def get_loader(self):
        return self.dataloader


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W

    def __init__(self, test_mode):
        self.test_mode = test_mode

    def __call__(self, sample):
        image, mask_com = sample['image'], sample['mask']
        mask, dist_lym_to_tumor = mask_com

        tumor_mask = mask.copy()
        lypho_mask = dist_lym_to_tumor.copy()

        tumor_mask[mask == 2.0] = 0.0
        tumor_mask[mask > 2.0] = 0.0

        lypho_mask[lypho_mask > 0.0] = 1.0

        if np.ndim(image) > 2:
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            tumor_mask = tumor_mask.reshape(1, tumor_mask.shape[0], tumor_mask.shape[1], tumor_mask.shape[2])
            dist_lym_to_tumor = dist_lym_to_tumor.reshape(1, dist_lym_to_tumor.shape[0], dist_lym_to_tumor.shape[1], dist_lym_to_tumor.shape[2])
            if self.test_mode == 'T':
                concat_img_tumor = np.concatenate((image, tumor_mask), axis=0)
            else:
                concat_img_tumor = np.concatenate((image, tumor_mask, dist_lym_to_tumor), axis=0)
        return {'feat': torch.from_numpy(concat_img_tumor)
                }