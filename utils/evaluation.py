'''

Include codes to evaluate models

'''

import torch
from tqdm import tqdm


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
