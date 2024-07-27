import torch
from dummy_model import DummyModel

PATH_LOAD_WEIGHTS = "../../models/multi_io/weights.pt"
PATH_SAVE_ONNX = "../../models/multi_io/model_multi_batch.onnx"

def export_model():
    # Instantiate the model and load initial weights
    model = DummyModel()
    model.eval()
    model.load_state_dict(torch.load(PATH_LOAD_WEIGHTS))

    # Define the input shapes for each tensor
    dummy_input1 = torch.randn(1, 3, 100, 100)  # Input for (3, 100, 100)
    dummy_input2 = torch.randn(1, 2, 50)        # Input for (2, 50)
    dummy_input3 = torch.randn(1, 25)           # Input for (25)
    
    # Export the model to ONNX with dynamic batch size
    torch.onnx.export(
        model,
        (dummy_input1, dummy_input2, dummy_input3),
        PATH_SAVE_ONNX,
        input_names=["x1", "x2", "x3"],  # Names for each input tensor
        output_names=["y1", "y2"],       # Names for each output tensor
        dynamic_axes={
            "x1": {0: "batch_size"},     # Dynamic batch size for input1
            "x2": {0: "batch_size"},     # Dynamic batch size for input2
            "x3": {0: "batch_size"},     # Dynamic batch size for input3
            "y1": {0: "batch_size"},     # Dynamic batch size for output1
            "y2": {0: "batch_size"}      # Dynamic batch size for output2
        },
        opset_version=16,  # ONNX opset version
        do_constant_folding=True
    )

if __name__ == "__main__":
    export_model()