###
# Inference code for DeepPETOPSCC-score
# "Deep learning for fully-automated prediction of overall survival in patients with
# oropharyngeal cancer using FDG PET imaging: an international retrospective study", Under Review
#  PAII & Chang Gung Memorial Hospital (CGMH), Sept 2020
# More examples will be available after acceptance of the work
##
import numpy as np
import argparse
import os
import torch
from models.networks import DeepConvSurv_Cox_Origin
from dataloaders.Oral_oncology_offline import Oral_dataloader_offline
from utils.evaluation import evaluation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def evaluate_test_data(root_dir, test_id, model_test, test_mode='T'):

    # Each time run validation sample


    TestData = Oral_dataloader_offline(root_dir, test_id,
                                  batch_size=1,
                                  test_mode=test_mode)
    Testloader = TestData.get_loader()

    test_val_risks = evaluation(model_test, Testloader)


    return test_val_risks

def inference_with_trained_DeepPET_OPSCC(test_mode='T'):

    """
    load parser parameters

    """

    if test_mode == 'T_N':
        use_lym = True
    else:
        use_lym = False

    print('test mode', test_mode)

    """
    load original survival label
    """

    patch_path = './sample_data/'

    """
    load original survival label
    """

    PID_cli = ['example']

    if test_mode == 'T':

        best_val_model_path = [
        'T_model_1.pth.tar',
        ]

    else:

        best_val_model_path = [
        'TN_model_1.pth.tar',
        ]


    for each_fold in range(1):

        if use_lym:
            model_path = './checkpoints/TN_model/' + best_val_model_path[each_fold]
        else:
            model_path = './checkpoints/T_model/' + best_val_model_path[each_fold]

        model_test = DeepConvSurv_Cox_Origin(use_lymph=use_lym).cuda()
        model_test.load_state_dict(torch.load(model_path))

        pred_test_risks = evaluate_test_data(patch_path,
                                                 PID_cli, model_test,
                                                 test_mode=test_mode)

        print('The risk score from model {} is {}'.format(test_mode, np.squeeze(pred_test_risks)))





if __name__ == '__main__':

    ##########################################################################################
    # The first step is to apply proposed segmentation models for Tumor and Lymph Nodes segmentation
    #############################################################################################

    # Codes will be available after the acceptance of the work

    ##########################################################################################
    # The second step is to crop VOI and save in data_saved based on Segmentation results
    #############################################################################################

    ##################################################################################
    # The third step is to generate dist map from lymph Nodes to Tumor
    ###################################################################################

    ##########################################################
    # Final step to inference, put one as example
    ##########################################################

    inference_with_trained_DeepPET_OPSCC(test_mode='T')
    inference_with_trained_DeepPET_OPSCC(test_mode='T_N')
