import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RmsNormSourceTests(unittest.TestCase):
    def test_optional_weights_are_not_passed_to_save_for_backward(self):
        source = (ROOT / "q8_kernels" / "functional" / "rms_norm.py").read_text(
            encoding="utf-8"
        )
        module = ast.parse(source)
        rms_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "RMSNorm8bit"
        )
        forward = next(
            node
            for node in rms_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
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
        self.assertIn("ctx.has_weights", source)
        self.assertIn("weights = None", source)


if __name__ == "__main__":
    unittest.main()
