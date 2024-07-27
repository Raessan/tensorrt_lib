import torch
from dummy_model import DummyModel
import os

PATH_LOAD_WEIGHTS = "../../models/single_io/weights.pt"
PATH_TEST_DATA = "../../test_data/single_io"
BATCH_SIZE = 3
SEED = 42

def save_data(seed = 42):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Instantiate the model and load initial weights
    model = DummyModel()
    model.eval()
    model.load_state_dict(torch.load(PATH_LOAD_WEIGHTS))

    
    inputs = torch.randn(BATCH_SIZE, 3, 100, 100)
    # Increase the variability among batches:
    for i in range(BATCH_SIZE):
        inputs[i,:,:,:] *= 10*i

    # Pass the inputs through the model to get the outputs
    outputs = model(inputs)

    # Save input data to a file
    with open(os.path.join(PATH_TEST_DATA, "x.txt"), "w") as f:
        for i in inputs:
            f.write(','.join(map(str, i.flatten().tolist())) + "\n")

    # Save output data to a file
    with open(os.path.join(PATH_TEST_DATA,"y.txt"), "w") as f:
        for o in outputs:
            f.write(','.join(map(str, o.tolist())) + "\n")

if __name__ == "__main__":
    save_data(SEED)