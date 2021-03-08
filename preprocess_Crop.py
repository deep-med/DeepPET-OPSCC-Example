import os
from utils.evaluation import load_nifti_itk, convert_num_to_nii, bbox2_3D, check_crop_boundary
import scipy.ndimage as ndimage
import numpy as np
import SimpleITK

def resample_fit_spacing(image, mask, prev_spacing, ref_spacing):
    """
    Resize images to ref spacing

    """

    zoom_ratio = (float(prev_spacing[0]) / float(ref_spacing[0]),
                  float(prev_spacing[1]) / float(ref_spacing[1]),
                  float(prev_spacing[2]) / float(ref_spacing[2]))

    image_npy = ndimage.zoom(image, zoom_ratio)

    label = range(1, 3)

    for each_label in label:
        cur_mask = mask.copy()
        cur_mask[cur_mask!= each_label] = 0.0
        cur_mask[cur_mask == each_label] = 1.0

        resample_npy = ndimage.zoom(cur_mask, zoom_ratio, order=1)

        resample_npy[resample_npy < 0.5] = 0.0
        resample_npy[resample_npy >= 0.5] = 1.0
        resample_npy[resample_npy > 1.0] = 1.0

        if each_label == 1:
            final_mask = resample_npy.copy()
        else:
            final_mask += resample_npy * each_label

    final_mask[final_mask>2.0] = 1.0

    return image_npy, final_mask


def crop_tumor_lym_each_patient(patient_lists, img_src_path, seg_src_path ):

    if patient_lists is None:
        return

    ref_spacing = [2, 2, 2]

    save_dir = './sample_data/'

    for cohort_path in patient_lists:
        nifit_name = cohort_path.split('/')[-1].split('.nii.gz')[0]
        print(nifit_name)

        list_folder = cohort_path.split('/')[:-1]
        nifit_img_folder = '/'.join(list_folder)
        nifit_mask_folder = nifit_img_folder.replace(img_src_path, seg_src_path)

        nifit_src_name = nifit_name + '.nii.gz'
        nii_seg_name = nifit_name + '.nii.gz'


        nii_mask_path = nifit_mask_folder + '/' + nii_seg_name
        nii_img_path = nifit_img_folder + '/' + nifit_src_name

        nii_imgsave_path = save_dir + '/img/{}.nii.gz'.format(nifit_name)

        if os.path.exists(nii_imgsave_path):
            print('exist continue')
            continue


        volumes, vol_info = load_nifti_itk(nii_img_path)
        masks, mask_info = load_nifti_itk(nii_mask_path)
        vol_spacing = mask_info[0]

        # determin if need resampling
        if abs(vol_spacing[0] - ref_spacing[0]) > 0.5:
            resampled_volumes, resampled_masks = resample_fit_spacing(volumes, masks, vol_spacing, ref_spacing)
            re_spacing = ref_spacing
        else:
            resampled_volumes, resampled_masks = volumes, masks
            re_spacing = vol_spacing

        re_info = [re_spacing, mask_info[1], mask_info[2]]

        #
        try:
            rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(resampled_masks)
        except:
            print('{} has issue, no mask'.format(nifit_name))
            continue


        w1 = rmin + int(round((rmax - rmin) / 2.))
        h1 = cmin + int(round((cmax - cmin) / 2.))
        d1 = zmin + int(round((zmax - zmin) / 2.))

        crop_size = [64, 64, 96]

        (w, h, d) = resampled_volumes.shape

        wL, wH = check_crop_boundary(w1, w, crop_size[0])
        hL, hH = check_crop_boundary(h1, h, crop_size[1])
        dL, dH = check_crop_boundary(d1, d, crop_size[2])

        crop_mask = resampled_masks[wL:wH, hL:hH, dL:dH]
        crop_tumor = resampled_volumes[wL:wH, hL:hH, dL:dH]


        test_mask = crop_mask[:]

        if np.max(test_mask) == 0:
            print('no mask extracted')

        try:
            nii_save_mask_path = os.path.join(save_dir+'/mask/', nifit_name + '.nii.gz')
            array_mask = convert_num_to_nii(crop_mask, re_info)

            SimpleITK.WriteImage(array_mask, nii_save_mask_path)
        except:
            print('crop mask too small than [64, 64, 96]')
            continue


        nii_img_save_path = os.path.join(save_dir+'/img/', nifit_name + '.nii.gz')
        array_img = convert_num_to_nii(crop_tumor, re_info)

        SimpleITK.WriteImage(array_img, nii_img_save_path)
