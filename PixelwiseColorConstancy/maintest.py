import os
import time
import util
import torch
import transforms as t
import dataset58 as dataset
import model.model as model
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# This file tests the model. This function takes the DNN model (model) and a save_path as inputs.
# It evaluates the model on a testing dataset. Sets the model to evaluation mode model.eval()
#  Then iterates over batches in the dataloader provided, containing testing data.
def test_model(model, save_path):
    since = time.time()

    # The model is set to evaluation mode (model.eval()) to disable dropout
    # and batch normalization layers, which behave differently during training
    model.eval()

    with torch.no_grad():  # disabling gradient calculations (not training)
        start_time = time.time()
        # For each batch, the input data (inputs) and ground truth labels (labels)
        # are loaded onto the appropriate device.
        for batch_idx, data in enumerate(dataloader):
            indexes, filenames, inputs, labels, gtsh = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # The forward pass is performed through the model to obtain the predicted outputs (outputs).
            outputs = model(inputs)
            # The util.PlotHCV function is called to visualize and
            # save the predicted outputs along with the input and ground truth labels.
            util.PlotHCV(save_path + "/", 'val' + '_' + str(batch_idx) + '_' + os.path.basename(filenames[0])[:-4],
                         inputs[0], labels[0], outputs[0])
            print(filenames[0])
    # Once evaluation completes for all batches,
    # the total time taken for evaluation (time_elapsed) is calculated and printed
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    batch_size = 1
    num_class = 58

    save_path = util.setup_logging_from_args()
    writer = SummaryWriter(save_path + "/logs")
    # Defines transformations for validation data (transVal)
    transVal = t.Compose([
        t.ToTensorOneHot(),
    ])
    # Creates a validation dataset (val_set)
    # using the ValDataset class from the dataset58 module and applies the defined transformations
    val_set = dataset.ValDataset("data/test/", transform=transVal)
    # Creates a DataLoader (dataloader) for the validation dataset.
    # The DataLoader handles data loading, batching, and parallelization
    dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    # Determines the device to use (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # Loads CNN
    model = model.UNetWithResnet50Encoder(num_class).to(device)
    # loads its optimal weights from a saved file (model_500.pth)
    weights = torch.load('model_500.pth', map_location='cuda:0')
    model.load_state_dict(weights)
    # Calls the test_model function with the loaded model and save path
    model = test_model(model, save_path)
    #  Prints 'Finish!' after testing is complete.
    print('Finish!')
