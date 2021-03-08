'''

Include codes to evaluate models

'''

import torch
from tqdm import tqdm
import SimpleITK
import numpy as np

def load_nifti_itk(file_name):


    reader = SimpleITK.ImageFileReader()
    reader.SetFileName(file_name)
    image = reader.Execute()

    spacing = np.array(list(image.GetSpacing()))

    origin = image.GetOrigin()
    direction = image.GetDirection()

    nda = SimpleITK.GetArrayFromImage(image)
    img_array = nda.transpose((2, 1, 0))[::-1, ::-1, :]

    return img_array, [spacing, origin, direction]


def convert_num_to_nii(image, info):
    spacing, origin, direction = info
    change_image = image[::-1, ::-1, :]
    image_array = np.swapaxes(change_image, 2, 0)
    imgitk = SimpleITK.GetImageFromArray(image_array)
    imgitk.SetSpacing((spacing[0], spacing[1], spacing[2]))
    imgitk.SetOrigin(origin)
    return imgitk

def bbox2_3D(volume):
    """
    Get 3D bounding box from mask
    :param volume: 3D mask
    :return: Bounding box
    """
    r = np.any(volume, axis=(1, 2))
    c = np.any(volume, axis=(0, 2))
    z = np.any(volume, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax, zmin, zmax]


def check_crop_boundary(center, bound, output_size):
    """
    check center if the crop data will exceed boundary

    :param center:
    :param bound:
    :return:
    """

    # center = 21, bound = 84, output = 64
    offset = 0

    dL = int(center - (output_size / 2 - offset))
    dH = int(center + (output_size / 2 + offset))

    if center < (output_size / 2):
        offset = output_size/2 - center
        dL = int(center - (output_size / 2 - offset))
        dH = int(center + (output_size / 2 + offset))

    if center + (output_size / 2) > bound - 1:
        offset = center + (output_size / 2) - (bound - 1)
        dL = int(center - (output_size / 2 + offset))
        dH = int(center + (output_size / 2 - offset))

    return dL, dH



def evaluation(model, testloader):
    """
    Evaluate model on validation/testing sets, set testing to control if it is test or validation
    :param testing:
    :return:
    """

    model.eval()


    pred_all = None


    with torch.no_grad():
        tbar = tqdm(testloader, desc='\r')

        for i_batch, sampled_batch in enumerate(tbar):

            X = sampled_batch['feat']


        # ===================forward=====================

            X = X.cuda()
            pred = model(X)
            final = pred

            if i_batch == 0:
                pred_all = final
            else:
                pred_all = torch.cat([pred_all, final])

    pred_risks = pred_all.cpu().numpy().reshape(-1)

    return pred_risks
