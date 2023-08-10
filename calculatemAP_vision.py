from cosinesimilarity import calculate_cosine_similarity

# Calculate vision modality retrieve touch modality mAP
def calculatemAP(val_bar, image_vision, image_touch_name, net, device):
    # Create a list which store the cosine similarity value
    similarity_list = [[] for _ in range(len(image_vision))]
    # Create a list which store the object name used to calculate cosine similarity value
    similarity_name_list = []
    # Create a list which store the mAP
    average_precision_list = [[] for _ in range(len(image_vision))]

    for val_data_2 in val_bar:
        # Get the vision and touch modalities data
        image_test_touch, image_vision_touch, image_test_touch_name, _ = val_data_2
        # Input the data into CrossNet and get output
        output_test_touch, _ = net(image_test_touch.to(device), image_vision_touch.to(device))
        # Calculate the cosine similarity between vision modality CrossNet output and touch modality CrossNet output
        similarity = calculate_cosine_similarity(image_vision, output_test_touch)
        for i in range(len(similarity)):
            # add the similarity values into the similarity_list
            similarity_list[i].extend(similarity[i])
        image_test_touch_name = list(image_test_touch_name)
        # add the names which used to calculate cosine similarity into the similarity_name_list
        similarity_name_list += image_test_touch_name


    for similarity_i in range(len(similarity_list)):
        # Set a variable which used to calculate mAP
        average_precision_local = 0.0
        # Zip the similarity_list and similarity_name_list together
        pairs = list(zip(similarity_list[similarity_i], similarity_name_list))
        # Sorted the zip pairs
        sorted_data = sorted(pairs, key=lambda x: x[0], reverse=True)
        # Get the sorted similarity names
        sorted_datanames = [pair[1] for pair in sorted_data]
        original_image_name = image_touch_name[similarity_i][:4]
        # Consider the ranking information to calculate mAP
        index_list = [index + 1 for index, x in enumerate(sorted_datanames) if original_image_name in x]
        # Calculate mAP
        for num, index_l in enumerate(index_list):
            precision = (num + 1) / index_l
            average_precision_local += precision
        if len(index_list) == 0:
            average_precision_local = 0
        else:
            average_precision_local /= len(index_list)
        # Store mAP into average_precision_list
        average_precision_list[similarity_i].append(average_precision_local)
    return average_precision_list
