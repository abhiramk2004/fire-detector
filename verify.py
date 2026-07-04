from openvino.runtime import Core


def main():
    # Initialize OpenVINO Runtime
    core = Core()

    # Print available devices
    print("=" * 50)
    print("Available Devices:")
    print(core.available_devices)

    # Load the model
    model_path = "models/openvino/fire.xml"
    model = core.read_model(model=model_path)

    # Compile the model for CPU
    compiled_model = core.compile_model(model=model, device_name="CPU")

    print("\nCompiled Successfully!")
    print(f"Target Device: CPU")

    # Get input information
    input_layer = compiled_model.input(0)
    print("\nInput Information")
    print("-" * 50)
    print(f"Name  : {input_layer.any_name}")
    print(f"Shape : {input_layer.shape}")
    print(f"Type  : {input_layer.element_type}")

    # Get output information
    output_layer = compiled_model.output(0)
    print("\nOutput Information")
    print("-" * 50)
    print(f"Name  : {output_layer.any_name}")
    print(f"Shape : {output_layer.shape}")
    print(f"Type  : {output_layer.element_type}")

    print("\n✓ Model verification successful!")


if __name__ == "__main__":
    main()
