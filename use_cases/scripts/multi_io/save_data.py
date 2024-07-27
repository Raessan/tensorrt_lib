import torch
from dummy_model import DummyModel
import os

PATH_LOAD_WEIGHTS = "../../models/multi_io/weights.pt"  # Update path as necessary
PATH_TEST_DATA = "../../test_data/multi_io"
BATCH_SIZE = 3
SEED = 42

def save_data(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Instantiate the model and load initial weights
    model = DummyModel()
    model.eval()
    model.load_state_dict(torch.load(PATH_LOAD_WEIGHTS))

    # Create dummy input tensors for the three inputs
    inputs1 = torch.randn(BATCH_SIZE, 3, 100, 100)
    inputs2 = torch.randn(BATCH_SIZE, 2, 50)
    inputs3 = torch.randn(BATCH_SIZE, 25)

    # Increase the variability among batches:
    for i in range(BATCH_SIZE):
        inputs1[i] *= 10 * i
        inputs2[i] *= 10 * i
        inputs3[i] *= 10 * i

    # Pass the inputs through the model to get the outputs
    outputs1, outputs2 = model(inputs1, inputs2, inputs3)

    # Ensure output directory exists
    os.makedirs(PATH_TEST_DATA, exist_ok=True)

    # Save input data to files
    def save_tensor_to_file(tensor, filename):
        with open(os.path.join(PATH_TEST_DATA, filename), "w") as f:
            for batch in tensor:
                f.write(','.join(map(str, batch.flatten().tolist())) + "\n")

    save_tensor_to_file(inputs1, "x1.txt")
    save_tensor_to_file(inputs2, "x2.txt")
    save_tensor_to_file(inputs3, "x3.txt")

    # Save output data to files
    save_tensor_to_file(outputs1, "y1.txt")
    save_tensor_to_file(outputs2, "y2.txt")

if __name__ == "__main__":
    save_data(SEED)