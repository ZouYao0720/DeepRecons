import os

import imageio
import numpy as np
import imgaug.augmenters as iaa
from ISR.utils.logger import get_logger
#iaa.seed(1)

class MedicalImageHandler:
    def __init__(self, lr_dir, hr_dir, patch_size, scale,  resize = False, factor = 1, n_validation_samples=None):
        self.folders = {'hr': hr_dir, 'lr': lr_dir}  # image folders
        self.extensions = ('.raw')  # admissible extension
        self.img_list = {}  # list of file names
        self.width = 512
        self.height = 512
        self.resize = resize
        self.factor = factor
        self.n_validation_samples = n_validation_samples
        self.patch_size = patch_size
        self.scale = scale
        self.max_value = 65535.0
        self.patch_size = {'lr': patch_size, 'hr': patch_size * self.scale}
        self.logger = get_logger(__name__)
        self._make_img_list()
        self._check_dataset()
    

    def _make_img_list(self):
        """ Creates a dictionary of lists of the acceptable images contained in lr_dir and hr_dir. """
        
        for res in ['hr', 'lr']:
            file_names = os.listdir(self.folders[res])
            file_names = [file for file in file_names if file.endswith(self.extensions)]
            self.img_list[res] = np.sort(file_names)
        
        if self.n_validation_samples:
            samples = np.random.choice(
                range(len(self.img_list['hr'])), self.n_validation_samples, replace=False
            )
            for res in ['hr', 'lr']:
                self.img_list[res] = self.img_list[res][samples]
    
    def _check_dataset(self):
        """ Sanity check for dataset. """
        
        # the order of these asserts is important for testing
        assert len(self.img_list['hr']) == self.img_list['hr'].shape[0], 'UnevenDatasets'
        #assert self._matching_datasets(), 'Input/LabelsMismatch'
    
    def _matching_datasets(self):
        """ Rough file name matching between lr and hr directories. """
        # LR_name.png = HR_name+x+scale.png
        # or
        # LR_name.png = HR_name.png
        LR_name_root = [x.split('.')[0].rsplit('x', 1)[0] for x in self.img_list['lr']]
        HR_name_root = [x.split('.')[0] for x in self.img_list['hr']]
        return np.all(HR_name_root == LR_name_root)
    
    def _not_flat(self, patch, flatness):
        """
        Determines whether the patch is complex, or not-flat enough.
        Threshold set by flatness.
        """
        
        if max(np.std(patch, axis=0).mean(), np.std(patch, axis=1).mean()) < flatness:
            return False
        else:
            return True
    
    def _crop_imgs(self, imgs, batch_size, flatness):
        """
        Get random top left corners coordinates in LR space, multiply by scale to
        get HR coordinates.
        Gets batch_size + n possible coordinates.
        Accepts the batch only if the standard deviation of pixel intensities is above a given threshold, OR
        no patches can be further discarded (n have been discarded already).
        Square crops of size patch_size are taken from the selected
        top left corners.
        """
        
        slices = {}
        crops = {}
        crops['lr'] = []
        crops['hr'] = []
        accepted_slices = {}
        accepted_slices['lr'] = []
        top_left = {'x': {}, 'y': {}}
        n = 50 * batch_size
        for i, axis in enumerate(['x', 'y']):
            top_left[axis]['lr'] = np.random.randint(
                0, imgs['lr'].shape[i] - self.patch_size['lr'] + 1, batch_size + n
            )
            top_left[axis]['hr'] = top_left[axis]['lr'] * self.scale
        for res in ['lr', 'hr']:
            slices[res] = np.array(
                [
                    {'x': (x, x + self.patch_size[res]), 'y': (y, y + self.patch_size[res])}
                    for x, y in zip(top_left['x'][res], top_left['y'][res])
                ]
            )
        
        for slice_index, s in enumerate(slices['lr']):
            candidate_crop = imgs['lr'][s['x'][0]: s['x'][1], s['y'][0]: s['y'][1], slice(None)]
            if self._not_flat(candidate_crop, flatness) or n == 0:
                crops['lr'].append(candidate_crop)
                accepted_slices['lr'].append(slice_index)
            else:
                n -= 1
            if len(crops['lr']) == batch_size:
                break
        
        accepted_slices['hr'] = slices['hr'][accepted_slices['lr']]
        
        for s in accepted_slices['hr']:
            candidate_crop = imgs['hr'][s['x'][0]: s['x'][1], s['y'][0]: s['y'][1], slice(None)]
            crops['hr'].append(candidate_crop)
        
        crops['lr'] = np.array(crops['lr'])
        crops['hr'] = np.array(crops['hr'])
        return crops
    
    def _apply_transform(self, img, transform_selection):
        """ Rotates and flips input image according to transform_selection. """
        
        rotate = {
            0: lambda x: x,
            1: lambda x: np.rot90(x, k=1, axes=(1, 0)),  # rotate right
            2: lambda x: np.rot90(x, k=1, axes=(0, 1)),  # rotate left
        }
        
        flip = {
            0: lambda x: x,
            1: lambda x: np.flip(x, 0),  # flip along horizontal axis
            2: lambda x: np.flip(x, 1),  # flip along vertical axis
        }
        
        rot_direction = transform_selection[0]
        flip_axis = transform_selection[1]
        
        img = rotate[rot_direction](img)
        img = flip[flip_axis](img)
        
        return img
    
    def _augment_batch(self, batch, augmentations):
        """ Transforms each individual image of the batch independently (due to randomness in the augmentator). """
        
        augment_result = [self._apply_data_augmentation(img, augmentations) for i, img in enumerate(batch)]
        
        lr_aug_t_batch = np.array(
            [t[0] for t in augment_result]
        )
        hr_aug_t_batch = np.array(
            [t[1] for t in augment_result]
        )

        return lr_aug_t_batch, hr_aug_t_batch
    
    def _apply_data_augmentation(self, batch, augmentor):
        """ Augment images according to augmentor. """
        
        # the transform applied simutaneous to lr and hr images
        lr_aug, hr_aug = augmentor(image=batch['lr'], heatmaps=batch['hr'])
        return lr_aug, hr_aug
    
    def _transform_batch(self, batch, transforms):
        """ Transforms each individual image of the batch independently. """
        
        t_batch = np.array(
            [self._apply_transform(img, transforms[i]) for i, img in enumerate(batch)]
        )
        return t_batch

    def get_batch(self, batch_size, idx=None, flatness=0.0):
        """
        Returns a dictionary with keys ('lr', 'hr') containing training batches
        of Low Res and High Res image patches.

        Args:
            batch_size: integer.
            flatness: float in [0,1], is the patch "flatness" threshold.
                Determines what level of detail the patches need to meet. 0 means any patch is accepted.
        """

        if not idx:
            # randomly select one image. idx is given at validation time.
            idx = np.random.choice(range(len(self.img_list['hr'])))
        img = {}
        medical_img = {}
        img_path = os.path.join(self.folders['lr'], self.img_list['lr'][idx])
        medical_img['lr'] = self._read_raw_medical(img_path)
        img_path = os.path.join(self.folders['hr'], self.img_list['hr'][idx])
        medical_img['hr'] = self._read_raw_medical(img_path)

        ### medical_img with size slices * self.width* self.height
        batches_lr = []
        batches_hr = []
        
        ### iterate every slice in a 3D medical raw image to make fully useage of the data
        for slice_idx in range(self.slices):
            for res in ['lr', 'hr']:
                img[res] = medical_img[res][slice_idx,:,:].reshape(self.width, self.height, 1) ## for cv data it is a RGB 3 channel image, reshape to a grep image with one channel
            
            # batch['lr']: np.array: batch size, patch size * patch size
            # batch['hr']: np.array: batch size, patch size * patch size
            batch = self._crop_imgs(img, batch_size, flatness)
            # random transform the images, we can also run this multiple times and use it to augment the data
            transforms = np.random.randint(0, 3, (batch_size, 2))
            batch['lr_affine'] = self._transform_batch(batch['lr'], transforms)
            batch['hr_affine'] = self._transform_batch(batch['hr'], transforms)
            
            # combine all the results into the bigger batch
            print (batch['lr'].shape)
            print (batch['lr_affine'].shape)
            batches_lr.append(batch['lr'])
            batches_lr.append(batch['lr_affine'])
            batches_hr.append(batch['hr'])
            batches_hr.append(batch['hr_affine'])
            
            # get all pre-define augmentions
            augmentions = self._get_valid_augmentions()
            for augment_idx in range(len(augmentions)):
                batch['lr_aug_%d'%augment_idx], batch['hr_aug_%d'%augment_idx] = self._augment_batch(batch, augmentions[augment_idx])
                print (batch['lr_aug_%d'%augment_idx].shape)
                batches_lr.append(batch['lr_aug_%d'%augment_idx])
                batches_hr.append(batch['hr_aug_%d'%augment_idx])

        # final_batch * self.width* self.height
        # final_batch = slices * (2*batch+1)
        final_batches = {'lr': np.asarray(batches_lr),
                        'hr': np.asarray(batches_hr)
                        }
        
        return final_batches
    
    def get_validation_batches(self, batch_size):
        """ Returns a batch for each image in the validation set. """
        
        if self.n_validation_samples:
            batches = []
            for idx in range(self.n_validation_samples):
                batches = batches + (self.get_batch(batch_size, idx, flatness=0.0))
            return batches
        else:
            self.logger.error(
                'No validation set size specified. (not operating in a validation set?)'
            )
            raise ValueError(
                'No validation set size specified. (not operating in a validation set?)'
            )
    
    def get_validation_set(self, batch_size):
        """
        Returns a batch for each image in the validation set.
        Flattens and splits them to feed it to Keras's model.evaluate.
        """
        
        if self.n_validation_samples:
            batches = self.get_validation_batches(batch_size)
            valid_set = {'lr': [], 'hr': []}
            for batch in batches:
                for res in ('lr', 'hr'):
                    valid_set[res].extend(batch[res])
            for res in ('lr', 'hr'):
                valid_set[res] = np.array(valid_set[res])
            return valid_set
        else:
            self.logger.error(
                'No validation set size specified. (not operating in a validation set?)'
            )
            raise ValueError(
                'No validation set size specified. (not operating in a validation set?)'
            )


    def _get_valid_augmentions(self):
        """ Generate pre_defined data augmentation templates (use the python imgaug lib). """

        seq1 = iaa.Sequential([
                iaa.Dropout(0.2),
                iaa.Affine(rotate=(-45, 45))
        ], random_order=True) # apply augmenters in random order

        seq2 = iaa.Sequential([
                iaa.Fliplr(0.5), # horizontal flips
                iaa.Crop(percent=(0, 0.1)), # random crops
                iaa.Sometimes(
                    0.5,
                    iaa.GaussianBlur(sigma=(0, 0.5))
                )
        ], random_order=True) # apply augmenters in random order

        seq3 = iaa.Sequential([
                iaa.Fliplr(0.5), # horizontal flips
                iaa.Crop(percent=(0, 0.1)), # random crops
                iaa.Sometimes(
                    0.5,
                    iaa.GaussianBlur(sigma=(0, 0.5))
                ),
                
                iaa.LinearContrast((0.75, 1.5)),
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-25, 25),
                    shear=(-8, 8)
                )
        
        ], random_order=True) # apply augmenters in random order

        seq4 = iaa.Sequential([
                iaa.Fliplr(0.5), # horizontal flips
                iaa.Crop(percent=(0, 0.1)), # random crops
                iaa.Sometimes(
                    0.5,
                    iaa.GaussianBlur(sigma=(0, 0.5))
                ),
                
                iaa.LinearContrast((0.75, 1.5)),
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-25, 25),
                    shear=(-8, 8)
                )
        
        ], random_order=True) # apply augmenters in random order

        seq5 = iaa.Sequential([
                iaa.Fliplr(0.5), # horizontal flips
                iaa.Crop(percent=(0, 0.1)), # random crops
                iaa.Sometimes(
                    0.5,
                    iaa.GaussianBlur(sigma=(0, 0.5))
                ),
                
                iaa.LinearContrast((0.75, 1.5)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-25, 25),
                    shear=(-8, 8)
                )
        
        ], random_order=True) # apply augmenters in random order

        augmentions = [seq1, seq2, seq3, seq4, seq5]

        return augmentions

    def _read_raw_medical(self, img_path):
        """ Read the .raw and return the n gray image as a list of np.array. """
        f = open(img_path, 'rb')
        img_str = f.read()
        img_arr = np.frombuffer(img_str, np.uint16)
        pages = int(len(img_arr)/(self.width*self.height))
        self.slices = pages
        img_arr = np.reshape(img_arr,(pages,self.width,self.height))/self.max_value
        f.close()
        return img_arr
