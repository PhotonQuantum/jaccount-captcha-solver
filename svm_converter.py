import argparse
import pickle


def main(input_file: str, output_file: str):
    import onnx
    from onnx import optimizer
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    print("Loading model...")
    with open(input_file, mode="rb") as f:
        model = pickle.load(f)
    initial_type = [("float_input", FloatTensorType([400]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    print("Validating model... ", end="")
    onnx.checker.check_model(onnx_model)
    print("Pass.")

    print("Optimizing model...")
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(onnx_model, passes)

    onnx.save(optimized_model, output_file)

    print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a sklearn pickle into ONNX format.")
    parser.add_argument("input_file", type=str, help="Pickle file.")
    parser.add_argument("output_file", type=str, help="Output ONNX file.")
    args = vars(parser.parse_args())
    main(**args)
