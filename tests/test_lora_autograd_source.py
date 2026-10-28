import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LoraAutogradSourceTests(unittest.TestCase):
    def test_lora_path_handles_quantized_inputs_and_batched_gradients(self):
        source = (ROOT / "q8_kernels" / "functional" / "linear.py").read_text(
            encoding="utf-8"
        )
        module = ast.parse(source)
        lora_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "Q8LinearLora"
        )
        forward = next(
            node
            for node in lora_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        backward = next(
            node
            for node in lora_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "backward"
        )

        self.assertIn("a_quant = a", source)
        self.assertIn("flat_grad_out_lora_b", source)
        self.assertIn("flat_lora_input", source)

        quantized_gradients = [
            node
            for node in ast.walk(backward)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "quantize_fp8"
        ]
        self.assertTrue(quantized_gradients)

        save_calls = [
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "save_for_backward"
        ]
        self.assertTrue(save_calls)
        self.assertFalse(
            any(
                isinstance(argument, ast.Constant) and argument.value is None
                for call in save_calls
                for argument in call.args
            )
        )


if __name__ == "__main__":
    unittest.main()
