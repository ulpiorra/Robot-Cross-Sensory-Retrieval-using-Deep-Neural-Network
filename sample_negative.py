import torch
import random

# sample modality image from another object as the negative example
def negative_example(modality, filename):
    negative = None
    for i in filename:
        # Get the original object filename
        name = i[:4]
        for index in range(len(filename)):
            # random choose a object file
            random_num = random.randint(0, len(filename)-1)
            # use filename check whether it is same for the original object
            if name not in filename[random_num]:
                if negative == None:
                    # Store negative object
                    negative = torch.unsqueeze(modality[random_num], dim=0)
                    break
                else:
                    tensor2 = torch.unsqueeze(modality[random_num], dim=0)
                    # Store negative object
                    negative = torch.cat((negative, tensor2), dim=0)
                    break
    return negative
