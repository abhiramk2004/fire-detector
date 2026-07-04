"""
detector.py

OpenVINO inference engine for Fire Detection.

Responsibilities:
- Load OpenVINO Runtime
- Load the model
- Compile the model
- Perform inference
- Return raw model output
"""

from openvino.runtime import Core
import config


class FireDetector:
    """OpenVINO Fire Detection Model"""

    def __init__(self):
        self.core = Core()
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None

        self.load_model()

    def load_model(self):
        """Load and compile the OpenVINO model."""

        print("[INFO] Loading OpenVINO model...")

        model = self.core.read_model(config.MODEL_PATH)

        self.compiled_model = self.core.compile_model(
            model=model,
            device_name=config.DEVICE
        )

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        print("[INFO] Model loaded successfully.")
        print(f"[INFO] Device : {config.DEVICE}")
        print(f"[INFO] Input Shape : {self.input_layer.shape}")
        print(f"[INFO] Output Shape: {self.output_layer.shape}")

    def infer(self, input_tensor):
        """
        Perform inference.

        Args:
            input_tensor (numpy.ndarray):
                Shape -> (1,3,480,480)

        Returns:
            numpy.ndarray:
                Raw model output.
        """

        result = self.compiled_model([input_tensor])

        return result[self.output_layer]
