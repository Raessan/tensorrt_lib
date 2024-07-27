import torch
import time
from dummy_model import DummyModel

PATH_LOAD_WEIGHTS = "../../models/single_io/weights.pt"
BATCH_SIZE = 5

def measure_inference_time():
    # Set the device to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate and load the model
    model = DummyModel().to(device)
    model.eval()
    model.load_state_dict(torch.load(PATH_LOAD_WEIGHTS))

    # Create a dummy input tensor and move it to the device
    dummy_input = torch.randn(BATCH_SIZE,  3, 100, 100).to(device)

    # Warm up the model (optional but recommended for accurate timing)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):  # Running multiple iterations for better averaging
            _ = model(dummy_input)
    end_time = time.time()

    avg_inference_time = (end_time - start_time)*1000.0 / 100
    print(f"Average inference time: {avg_inference_time:.6f} miliseconds")

if __name__ == "__main__":
    measure_inference_time()
