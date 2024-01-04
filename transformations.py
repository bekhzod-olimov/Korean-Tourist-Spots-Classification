# Import library
from torchvision import transforms as T

def get_tfs(ds_name, im_dims = (224, 224)): 

    """

    This function gets several parameters and returns transformations to be applied for the dataset.

    Parameters:

       ds_name      - name of the dataset, str;
       im_dims      - image dimensions, tuple.

    Output:

       tfs          - transformations to be applied, list -> transforms object.

    """

    return [T.Compose([T.Resize(im_dims), T.RandomHorizontalFlip(), T.ToTensor()]), T.Compose([T.Resize(im_dims), T.ToTensor()])]
