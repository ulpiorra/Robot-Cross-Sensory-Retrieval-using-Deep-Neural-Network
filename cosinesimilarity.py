import numpy as np
import torch
from torch.nn.functional import normalize

# calculate cosine similarity
def calculate_cosine_similarity(image1, image2):

    similarity_list_local = []
    batch = len(image1)
    image1_flat = image1.view(batch,-1)
    image2_flat = image2.view(batch,-1)

    image1_norm = normalize(image1_flat)
    image2_norm = normalize(image2_flat)

    for indexA in range(len(image1_norm)):

        single_image_similarity = image1_norm[indexA]*image2_norm
        cosine_similarity = torch.sum(single_image_similarity, dim=1)
        cosine_similarity.tolist()
        similarity_list_local.append(cosine_similarity)

    return similarity_list_local
