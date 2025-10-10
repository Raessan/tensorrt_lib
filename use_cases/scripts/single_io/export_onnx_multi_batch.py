import torch
from dummy_model import DummyModel

PATH_LOAD_WEIGHTS = "../../models/single_io/weights.pt"
PATH_SAVE_ONNX = "../../models/single_io/model_multi_batch.onnx"

def export_model():

    # Instantiate the model and load initial weights
    model = DummyModel()
    model.eval()
    model.load_state_dict(torch.load(PATH_LOAD_WEIGHTS))

    # Define the input shape
    dummy_input = torch.randn(1, 3, 100, 100)

    # Export the model to ONNX with dynamic batch size
    torch.onnx.export(
        model,
        dummy_input,
        PATH_SAVE_ONNX,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True
    )

if __name__ == "__main__":
    export_model()