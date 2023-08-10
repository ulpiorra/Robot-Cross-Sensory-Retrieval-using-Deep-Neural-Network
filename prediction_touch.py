import os
import sys
import torch
from torchvision import transforms
from tqdm import tqdm
from custom_dataset_reader import CustomImageDataset
from crossnet import CrossNet
from calculatemAP_touch import calculatemAP

# Predict touch modality retrieve vision modality
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("using {} device.".format(device))
    # Data Processing
    data_transform = {
        "val": transforms.Compose([
                                   transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path = 'test_dataset'
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    print('Using {} dataloader workers every process'.format(nw))
    # Data processing the validate dataset
    validate_dataset = CustomImageDataset(root1=os.path.join(image_path, "val", "touch"),
                                          root2=os.path.join(image_path, "val", "vision"),
                                            transform=data_transform["val"])
    # Load validate dataset
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, drop_last=True)

    # Initial CrossNet
    net = CrossNet()
    net = net.to(device)
    # Load weight
    net.load_state_dict(torch.load('CrossNetWeight.pth'))


    # Validate
    mAP_list=[]

    val_bar = tqdm(validate_loader, file=sys.stdout)
    for i, val_data in enumerate(val_bar):
        # Get the vision and touch modalities validate data
        val_touch, val_vision, val_touch_name, val_vision_name = val_data
        # Input the data into CrossNet and get output
        output_val_touch, _ = net(val_touch.to(device), val_vision.to(device))
        # Calculate the mAP for touch modality retrieve vision modality
        average_precision = calculatemAP(val_bar, output_val_touch, val_vision_name, net, device)

        for AP in average_precision:
            mAP_list.append(AP[0])


    mAP = sum(mAP_list) / 80
    print("The mAP is")
    print(mAP)


    print('Finished validation')


if __name__ == '__main__':
    main()
