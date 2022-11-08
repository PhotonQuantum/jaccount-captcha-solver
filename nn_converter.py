import argparse
from tempfile import NamedTemporaryFile


def main(input_file: str, output_file: str):
    import onnx
    import torch
    from onnxsim import simplify
    from nn_models import resnet20, PLWrapper

    print("Loading model...")
    cpu = torch.device("cpu")
    model = PLWrapper(resnet20(), -1)
    model.load_from_checkpoint(input_file, map_location=cpu)

    with NamedTemporaryFile() as temp_file:
        print("Tracing model...")
        input_array = torch.ones(1, 1, 40, 110)
        model.to_onnx(temp_file, input_sample=input_array, keep_initializers_as_inputs=True)
        temp_file.seek(0)

        print("Validating model... ", end="")
        onnx_model = onnx.load(temp_file)
        onnx.checker.check_model(onnx_model)
        print("Pass.")

    print("Optimizing model...")
    optimized_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"

    onnx.save(optimized_model, output_file)

    print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PyTorch checkpoint into ONNX format.")
    parser.add_argument("input_file", type=str, help="Checkpoint file.")
    parser.add_argument("output_file", type=str, help="Output ONNX file.")
    args = vars(parser.parse_args())
    main(**args)
