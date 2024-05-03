import torch
from torch.nn import functional as F
#  this loss function takes into account both object and background pixels in the image,
#  weighting the importance of each based on their respective binary cross-entropy losses,
#  to compute a final loss that guides the training of the neural network.
def calc_loss( pred, target,gtsh):

    # The function first extracts the ground truth object ID
    # and material from the ground truth Munsell chip representation (gtsh).
    gtObjId=gtsh[:, 0, :, :]
    gtMaterial = gtsh[:, 1, :, :]

    # Iterates over the predictions (pred) for each channel
    # (representing value, hue, and chroma).
    # Within each iteration, the function filters the
    # predictions and targets based on the ground truth object ID and material.
    for i in  range(pred.shape[1]):
        # separates predictions and targets into two groups
        # one for object pixels (where gtObjId is less than or equal to 2)
        # and one for background pixels (where gtObjId is greater than 2).
        # ensures that the material is not equal to 0.
        tmp1=pred[:, i, :, :]
        tmp1=tmp1[(gtObjId <= 2)& (gtMaterial != 0)]

        tmp2=target[:, i, :, :]
        tmp2=tmp2[(gtObjId <= 2)& (gtMaterial != 0)]

        tmp3 = pred[:, i, :, :]
        tmp3 = tmp3[(gtObjId >2)& (gtMaterial != 0)]

        tmp4 = target[:, i, :, :]
        tmp4 = tmp4[(gtObjId >2)& (gtMaterial != 0)]

        if i==0:
            predObj=tmp1
            targetObj=tmp2
            predBG=tmp3
            targetBG=tmp4

        else:
            predObj=torch.cat((predObj,tmp1))
            targetObj=torch.cat((targetObj,tmp2))
            predBG=torch.cat((predBG,tmp3))
            targetBG=torch.cat((targetBG,tmp4))

    # For both object and background pixels, the function calculates the binary cross-entropy loss
    # between the predictions and targets
    # This loss function is commonly used for binary classification tasks,
    # such as this one where each pixel is classified as part of the object or background.
    bceObjects=F.binary_cross_entropy_with_logits(predObj,targetObj,reduction ='mean')
    bceBG=F.binary_cross_entropy_with_logits(predBG,targetBG,reduction ='mean')

    # The final loss (bca) is computed as a weighted combination of the object and background
    # binary cross-entropy losses (bceObjects and bceBG) using a coefficient m.
    # This combination aims to give more importance to the object pixels (bceObjects)
    # while still considering the background pixels (bceBG).
    m=0.85
    bca=(m*bceObjects)+((1-m)*bceBG)
    # returns the total loss (bca) along with the individual losses
    # for object pixels (bceObjects) and background pixels (bceBG)
    return bca,bceObjects,bceBG




