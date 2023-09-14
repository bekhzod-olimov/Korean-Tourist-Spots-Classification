# Import libraries
import torch, wandb, argparse, yaml, os, pickle, pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger; from time import time
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from datasets import CustomDataloader, get_dl; from transformations import get_tfs
from model import TripletModel, TripletImagePredictionLogger, Model, ImagePredictionLogger

def run(args):
    
    """
    
    This function runs the main script based on the arguments.
    
    Parameter:
    
        args - parsed arguments.
        
    Output:
    
        train process.
    
    """
    
    # Get train arguments 
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    print(f"\nTraining Arguments:\n\n{argstr}")

    # Wandb login
    os.system("wandb login --relogin 3204eaa1400fed115e40f43c7c6a5d62a0867ed1")
    # Create directories
    os.makedirs(args.dls_dir, exist_ok = True); os.makedirs(args.stats_dir, exist_ok = True)
    
    # Get the dataloaders
    if args.dataset_name in ["cars", "ghim", "default_sketch", "con_sketch", "sketch", "places", "korean_landmarks"]:
        # Get transformations to be applied
        tfs = get_tfs(args.dataset_name, args.inp_im_size)[1]
        # Get the train dataloader
        dl = CustomDataloader(transformations = tfs, bs = args.batch_size)
        # Split the dataloader to train, validation, and test dataloaders
        tr_dl, val_dl, test_dl = dl.get_dls()
        cls_names, n_cls = dl.tr_ds.get_info()
    elif args.dataset_name in ["cifar10", "mnist"]:
        # Get transformations to be applied
        tr_tfs, val_tfs = get_tfs(args.dataset_name, args.inp_im_size)
        # Split the dataloader to train, validation, and test dataloaders
        tr_dl, val_dl, test_dl, cls_names, n_cls = get_dl(ds_name = args.dataset_name, tr_tfs = tr_tfs, val_tfs = val_tfs, bs = args.batch_size)
    
    torch.save(test_dl, f"{args.dls_dir}/{args.dataset_name}_test_dl")
    
    # Create a class names file name
    cls_names_file = f"{args.dls_dir}/{args.dataset_name}_cls_names.pkl"
    # Save the class names
    with open(f"{cls_names_file}", "wb") as f: pickle.dump(cls_names, f)

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(val_dl))

    # Get the model to be trained
    # model = TripletModel(input_shape = args.inp_im_size, model_name = args.model_name, num_classes = n_cls, lr = args.learning_rate, margin = args.margin, ds_name = args.dataset_name)
    model = Model(input_shape = args.inp_im_size, model_name = args.model_name, num_classes = n_cls, lr = args.learning_rate)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project = f"{args.dataset_name}", job_type = "train", name = f"{args.dataset_name}_{args.model_name}_{args.batch_size}")

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs = args.epochs, accelerator = "gpu", devices = args.devices, strategy = "ddp", logger = wandb_logger, fast_dev_run = False,
                         callbacks = [EarlyStopping(monitor = "valid_loss", mode = "min", patience = 5), 
                                      # ImagePredictionLogger(val_samples, cls_names = cls_names, ds_name = args.dataset_name), 
                                      ModelCheckpoint(monitor = "valid_loss", dirpath = args.save_model_path, filename = f"{args.model_name}_{args.dataset_name}_best")])

    # Set the training process start time
    start_time = time()
    # Start training
    trainer.fit(model, tr_dl, val_dl)
    # Get train and validation times lists
    # train_times, valid_times = model.get_stats()
    # # Save the stats
    # torch.save(train_times, f"{args.stats_dir}/pl_train_times_{args.devices}_gpu"); torch.save(valid_times[1:], f"{args.stats_dir}/pl_valid_times_{args.devices}_gpu")
    # Close wandb run
    wandb.finish()
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = "Image Classification Training Arguments")
    
    # Add arguments to the parser
    parser.add_argument("-bs", "--batch_size", type = int, default = 64, help = "Mini-batch size")
    parser.add_argument("-is", "--inp_im_size", type = tuple, default = (224, 224), help = "Input image size")
    parser.add_argument("-dn", "--dataset_name", type = str, default = "korean_landmarks", help = "Dataset name for training")
    parser.add_argument("-mn", "--model_name", type = str, default = "rexnet_150", help = "Model name for backbone")
    # parser.add_argument("-mn", "--model_name", type = str, default = 'vit_base_patch16_224', help = "Model name for backbone")
    # parser.add_argument("-mn", "--model_name", type = str, default = 'vgg16_bn', help = "Model name for backbone")
    parser.add_argument("-d", "--devices", type = int, default = 4, help = "Number of GPUs for training")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-3, help = "Learning rate value")
    parser.add_argument("-mr", "--margin", type = float, default = 0.3, help = "Loss margin value")
    parser.add_argument("-e", "--epochs", type = int, default = 100, help = "Train epochs number")
    parser.add_argument("-sm", "--save_model_path", type = str, default = "saved_models", help = "Path to the directory to save a trained model")
    parser.add_argument("-sd", "--stats_dir", type = str, default = "stats", help = "Path to dir to save train statistics")
    parser.add_argument("-dl", "--dls_dir", type = str, default = "saved_dls", help = "Path to dir to save dataloaders")    
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
