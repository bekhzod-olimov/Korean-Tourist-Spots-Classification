# Import libraries
import os, torch, cv2, random, numpy as np
from collections import OrderedDict as OD
from glob import glob
from time import time; from tqdm import tqdm; from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM; from pytorch_grad_cam.utils.image import show_cam_on_image


def get_fm(fm):
        
        """
        
        This function gets feature map with size (bs, fm_shape, 7, 7)
        applies average pooling and returns feature map with shape (bs, fm_shape).
        
        Parameter:
        
            fm - feature map, tensor.
        
        Output:
        
            fm - reshaped feature map, tensor.
        
        """
        
        pool = torch.nn.AvgPool2d((fm.shape[2],fm.shape[3]))
        
        return torch.reshape(pool(fm), (-1, fm.shape[1]))
    
def get_state_dict(checkpoint_path):

    """
    
    This function gets a path to the trained model checkpoint and return new state dictionary.

    Parameter:
    
        checkpoint_path       - a path tho the checkpoint with the trained model, torch object.

    Output:

        new state dictionary  - a new state dictionary where parameter names are compatible with torch load.
    
    """
    
    # Get checkpoint that is trained on pytorch lighting
    checkpoint = torch.load(checkpoint_path)
    # Create a new dictionary
    new_state_dict = OD()
    
    # Go through the checkpoint dictionary items
    for k, v in checkpoint["state_dict"].items():
        # Remove "model."
        name = k.replace("model.", "") 
        new_state_dict[name] = v
    
    return new_state_dict

# A function to convert tensor to numpy array 
def tn2np(t, inv_fn = None): return (inv_fn(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8) if inv_fn is not None else (t * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def get_preds(model, test_dl, device):

    """
    
    This function gets several parameters and gets predictions based on them.

    Parameters:

        model     - a trained model, timm model object;
        test_dl   - test dataloader, torch dataloader object;
        device    - gpu name, str.

    Output:

        all_ims   - images from the test dataloader, list;
        all_preds - predictions of the model, list;
        all_gts   - ground truth labels, list.        
    
    """
    
    print("Start inference...")
    
    # Set the lists to track the metrics and initial accuracy score
    all_ims, all_preds, all_gts, acc = [], [], [], 0
    # Start time for the inference
    start_time = time()    
    # Go through the test dataloader
    for idx, batch in tqdm(enumerate(test_dl)):
        # Get images and their corresponding labels
        ims, gts = batch
        # Add them to the list
        all_ims.extend(ims); all_gts.extend(gts);        
        # Get predictions
        preds = model(ims.to(device))
        # Get predicted classes
        pred_clss = torch.argmax(preds.data, dim = 1)
        # Add the predicted classes to the list
        all_preds.extend(pred_clss)
        # Compute accuracy score for the batch
        acc += (pred_clss == gts.to(device)).sum().item()
        
    # Verbose
    print(f"Inference is completed in {(time() - start_time):.3f} secs!")
    print(f"Accuracy of the model is {acc / len(test_dl.dataset) * 100:.3f}%")
    
    return all_ims, all_preds, all_gts
    
def visualize(all_ims, all_preds, all_gts, num_ims, rows, cls_names, save_path, save_folder, save_name):
    
    """

    This function gets several parameters and visualizes the results of the inference.

    Parameters:

        all_ims          - images from the dataloader, tensor;
        all_preds        - predicted classes from the dataloader, tensor;
        all_gts          - labels from the dataloader, tensor;
        num_ims          - number of images to be visualized, int;
        rows             - number of rows in the plot, int;
        cls_names        - class names of the dataset, list;
        save_path        - path to save the visualization result, str;
        save_name        - prefix of the name to be saved, str.
    
    """
    
    print("Start visualization...")
    # Set the figure size
    plt.figure(figsize = (20, 20))
    # Get random indices from the dataloader based on the number of imags
    indices = [random.randint(0, len(all_ims) - 1) for _ in range(num_ims)]
    
    # Go through random indices list
    for idx, ind in enumerate(indices):
        
        # Get an image
        im = all_ims[ind]
        # Get a label
        gt = all_gts[ind].item()
        # Get a predicted label
        pr = all_preds[ind].item()
        # Create a subplot
        plt.subplot(rows, num_ims // rows, idx + 1)
        # Show the input image
        plt.imshow(tn2np(im))
        # Turn off the axis
        plt.axis("off")
        # Set the title
        plt.title(f"GT: {cls_names[gt]} | Pred: {cls_names[pr]}")
    
    # Save the figure
    os.makedirs(f"{save_path}/{save_folder}", exist_ok = True)
    files = sorted(glob(f"{save_path}/{save_folder}/*"))
    save_num = 0 if len(files) == 0 else int(files[-1].split(".")[0].split("_")[-1])
    plt.savefig(f"{save_path}/{save_folder}/{save_name}_preds_{save_num + 1}.png")
    print(f"The visualization can be seen in {save_path}/{save_folder} directory.")
    
    
    
def grad_cam(model, all_ims, num_ims, rows, save_path, save_folder, save_name):

    """
    
    This function gets several parameters and visualizes GradCAM results.

    Parameters:

        model           - a trained model, torch model object;
        all_ims         - images from the dataloader;
        num_ims         - number of images to be visualized, int;
        rows            - number of rows in the plot, int;
        save_path       - path to save the visualization result, str;
        save_name       - prefix of the name to be saved, str.
    
    """
    
    print("\nStart GradCAM visualization...")
    # Set the figure size
    plt.figure(figsize = (20, 20))
    # Get random indices from the dataloader based on the number of imags
    indices = [random.randint(0, len(all_ims) - 1) for _ in range(num_ims)]
    
    # Go through random indices list
    for idx, ind in enumerate(indices):
        # Get an image
        im = all_ims[ind]
        # Convert the tensor image to numpy array
        ori_cam = tn2np((im)) / 255
        # Initialize gradcam
        cam = GradCAM(model = model, target_layers = [model.features[-1]], use_cuda = True)
        # Get grayscale gradcam result
        grayscale_cam = cam(input_tensor = im.unsqueeze(0))[0, :]
        # Get combination of the input and gradcam result image
        vis = show_cam_on_image(ori_cam, grayscale_cam, image_weight = 0.6, use_rgb = True)
        
        # Create a subplot
        plt.subplot(rows, num_ims // rows, idx + 1)
        # Show the image
        plt.imshow(vis)
        # Turn off the axis
        plt.axis("off")
        # Set the title
        plt.title("GradCAM Visualization")
        
    # Save the visualization result
    os.makedirs(f"{save_path}/{save_folder}", exist_ok = True)
    files = sorted(glob(f"{save_path}/{save_folder}/*"))
    save_num = 0 if len(files) == 0 else int(files[-1].split(".")[0].split("_")[-1])
    plt.savefig(f"{save_path}/{save_folder}/{save_name}_gradcam_{save_num + 1}.png")
    print(f"The GradCAM visualization can be seen in {save_path}/{save_folder} directory.")
