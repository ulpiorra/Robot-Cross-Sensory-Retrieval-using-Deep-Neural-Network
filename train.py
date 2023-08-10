import sys

import torch
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from custom_dataset_reader import CustomImageDataset
from sample_negative import negative_example
from calculatemAP_vision import calculatemAP
import os
from crossnet import CrossNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # Data Processing
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path = 'test_dataset'
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # Data processing the train dataset
    train_dataset = CustomImageDataset(root1=os.path.join(image_path, "train", "touch"),
                                       root2=os.path.join(image_path, "train", "vision"),
                                       transform=data_transform["train"])
    train_num = len(train_dataset)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    # Load train dataset
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, drop_last=True)
    # Data processing the validate dataset
    validate_dataset = CustomImageDataset(root1=os.path.join(image_path, "val", "touch"),
                                          root2=os.path.join(image_path, "val", "vision"),
                                          transform=data_transform["val"])
    val_num = len(validate_dataset)
    # Load validate dataset
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, drop_last=True)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # Initial CrossNet
    net = CrossNet()
    net = net.to(device)

    # Define loss function
    loss_function = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')

    # Construct an optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.01)

    epochs = 300
    best_acc = 0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # Train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            # Get the vision and touch modalities train data
            image_touch, image_vision, filename_touch, filename_vision = data
            net.zero_grad()
            # Input the data into CrossNet and get output
            output_touch, output_vision = net(image_touch.to(device), image_vision.to(device))
            # Sample an tactile image from another object as the negative example
            negative_touch = negative_example(image_touch, filename_touch)
            # Sample an vision image from another object as the negative example
            negative_vision = negative_example(image_vision, filename_vision)
            # Input the negative example vision image and tactile image into the CrossNet and get the output
            negative_output_touch, _ = net(negative_touch.to(device), negative_vision.to(device))
            # Put the vision modality output and touch modality output
            # and touch modality negative example output into triplet loss
            loss = loss_function(output_vision, output_touch, negative_output_touch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # Validate
        net.eval()
        mAP_list = []
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for i, val_data in enumerate(val_bar):
                # Get the vision and touch modalities validate data
                val_touch, val_vision, val_touch_name, val_vision_name = val_data
                # Input the data into CrossNet and get output
                _, output_val_vision = net(val_touch.to(device), val_vision.to(device))

                # Calculate the mAP for vision modality retrieve touch modality
                average_precision = calculatemAP(val_bar, output_val_vision, val_touch_name, net, device)

                for AP in average_precision:
                    mAP_list.append(AP[0])

                validate_loader.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                                   epochs)
            # mAP is calculate from total value divide the number of objects
            mAP = sum(mAP_list) / 80
            print(f'[epoch %d] train_loss: %.3f  val_mAP: %.3f best_mAP: %.3f' %
                  (epoch + 1, running_loss / train_steps, mAP, best_acc))

            if mAP > best_acc:
                best_acc = mAP

    print('Finished Training')

if __name__ == '__main__':
    main()
