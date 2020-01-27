import argparse
from tempfile import NamedTemporaryFile


def main(input_file: str, output_file: str):
    import onnx
    import torch.onnx
    from onnx import optimizer
    from nn_models import resnet20

    print("Loading model...")
    model = resnet20()
    cpu = torch.device("cpu")
    checkpoint = torch.load(input_file, map_location=cpu)
    model.load_state_dict(checkpoint["net"])

    with NamedTemporaryFile() as temp_file:
        print("Tracing model...")
        input_array = torch.ones(1, 1, 40, 100)
        torch.onnx.export(model, input_array, temp_file, keep_initializers_as_inputs=True)
        temp_file.seek(0)

        print("Validating model... ", end="")
        onnx_model = onnx.load(temp_file)
        onnx.checker.check_model(onnx_model)
        print("Pass.")

    print("Optimizing model...")
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(onnx_model, passes)

    onnx.save(optimized_model, output_file)

    print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PyTorch checkpoint into ONNX format.")
    parser.add_argument("input_file", type=str, help="Checkpoint file.")
    parser.add_argument("output_file", type=str, help="Output ONNX file.")
    args = vars(parser.parse_args())
    main(**args)
