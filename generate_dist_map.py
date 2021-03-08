import glob
from utils.Mash2Mesh import Mask2Mesh
import numpy as np
import SimpleITK as sitk
import torch
from kaolin.metrics.pointcloud import sided_distance
from skimage import measure
from skimage import morphology
import SimpleITK

def load_nifti_mask_itk(file_name):


    reader = SimpleITK.ImageFileReader()
    reader.SetFileName(file_name)
    image = reader.Execute()

    spacing = np.array(list(image.GetSpacing()))

    origin = image.GetOrigin()
    direction = image.GetDirection()

    nda = SimpleITK.GetArrayFromImage(image)
    img_array = nda.transpose((2, 1, 0))

    return img_array, [spacing, origin, direction]

def convert_to_dist_map(image, info):
    spacing, origin, direction = info
    image_array = np.swapaxes(image, 2, 0)
    imgitk = SimpleITK.GetImageFromArray(image_array)
    imgitk.SetSpacing((spacing[0], spacing[1], spacing[2]))
    imgitk.SetOrigin(origin)
    return imgitk

def remove_fp_lymphnodes(ndarray, minArea=4):
    ndarray_output = ndarray
    label = measure.label((ndarray==2).astype(np.int8))
    mask = morphology.remove_small_objects(label, minArea)
    ndarray_output[mask == 0] = 0
    return ndarray_output

def generate_distmap(mask_dir, save_dir):

    all_mask_paths = glob.glob(mask_dir + '*.nii.gz')

    max_dist = []

    for each_mask_path in all_mask_paths:
        masks, mask_info = load_nifti_mask_itk(each_mask_path)

        masks = remove_fp_lymphnodes(masks)

        nii_name = each_mask_path.split('/')[-1].split('.nii.gz')[0]


        m2m_tumor = Mask2Mesh(each_mask_path, target=1)
        m2m_tumor.getMesh(target=1)  # set GT label, 1 for liver
        tumor_pts = np.asarray(m2m_tumor.points, dtype=np.float32)  # original dense points

        print(len(tumor_pts))

        if len(tumor_pts) < 50:
            print('no tumor mask {}'.format(nii_name))

            dist_map = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2]))

            nii_img_path = save_dir + nii_name + '_dist_map.nii.gz'
            # print(nii_img_path)
            array_img = convert_to_dist_map(dist_map, mask_info)

            sitk.WriteImage(array_img, nii_img_path)

            continue

        for target in [2]:
            m2m_vein = Mask2Mesh(each_mask_path, target=2)
            m2m_vein.getMesh(target=2)  # set GT label, 1 for liver
            dense_pts = np.asarray(m2m_vein.points, dtype=np.float32)  # original dense points
            print(len(dense_pts))

        masks[masks == 2] = 20
        masks[masks != 20] = 0.0

        dist_map = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2]))

        position = np.where(masks == 20)

        points = [np.expand_dims(position[0], axis=1),
                  np.expand_dims(position[1], axis=1),
                  np.expand_dims(position[2], axis=1)]

        points = np.concatenate(points, axis=1)
        points = np.asarray(points, dtype=np.float32)

        xyz1 = torch.from_numpy(np.expand_dims(tumor_pts, axis=0)).cuda()
        target_xyz = torch.from_numpy(np.expand_dims(points*2.0, axis=0)).cuda()

        dist2, idx = sided_distance(target_xyz, xyz1)

        np_dist = dist2.data.cpu().numpy()
        np_dist = np.squeeze(np_dist)

        posi = np.asarray(points, dtype=np.int8)
        for idd in range(len(points)):
            x, y, z = posi[idd]
            result = np_dist[idd]
            cur_result = np.sqrt(result)
            dist_map[x:x + 1, y:y + 1,
            z:z + 1] = cur_result  # /max_dist# mm #1 / (1 + np.exp(result / 10000.)) #9 - np.log(result)

        max_dist.append(np.max(dist_map[:]))


        nii_img_path = save_dir + nii_name + '.nii.gz'
        # print(nii_img_path)
        array_img = convert_to_dist_map(dist_map, mask_info)

        sitk.WriteImage(array_img, nii_img_path)
    print('max dist is', np.max(max_dist))

