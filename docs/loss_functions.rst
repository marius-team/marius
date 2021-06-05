.. _loss_functions

Loss Functions
********************

Marius offers multiple choices for computing loss

==================  =============
   Name             Description
------------------  -------------
SoftMaxLoss         A loss function which gives the cross entropy loss after softmax output
RankingLoss         A loss function which gives the margin ranking loss
BCEAfterSigmoid     A loss function which applies a sigmoid layer before binary cross entropy loss. This is more numerically unstable than BCEWithLogitsLoss.
BCEWithLogitsLoss   A loss function which combines a sigmoid layer and the binary cross entropy loss
MSELoss             A loss function which gives the mean square error loss
SoftPlusLoss        A loss function which gives the softplus loss
==================  =============