


import torchvision
import torch
import numpy as np
import torch.nn as nn

import torchvision.datasets as vDatasets
import numpy as np
vDatasets.Cityscapes

%matplotlib inline
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Download the dataset after registering in the https://www.cityscapes-dataset.com unzip and place the contents inside the folder Cityscapes after unZipping.
  Cityscapes----/gtCoarse
            |---/gtFine
            |---/leftImg8bit_trainvaltest
            
            
dataset = torchvision.datasets.cityscapes.Cityscapes('/home/Cityscapes',split='train', mode='fine',target_type='semantic')


img1, smnt = dataset[2] # call the second element of the dataset(Or any other element within the dataset)


print(img1) # print the size

#Display the second element in the dataset
imgplot0 = plt.imshow(img1) 

#Display the second elements  segemntation image in the dataset
imgplot1 = plt.imshow(smnt)

# after viewing you can convert it to tensor by the following command which you can use to give input to your FCNN

img1toTensor = ToTensor()(img1)




