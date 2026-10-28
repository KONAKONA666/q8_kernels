import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RopeSourceTests(unittest.TestCase):
    def test_both_frequency_tensors_match_input_shape(self):
        source = (ROOT / "q8_kernels" / "functional" / "rope.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "assert cos_freqs.shape == x.shape and sin_freqs.shape == x.shape",
            source,
        )

    def test_rope_backward_returns_one_gradient_per_forward_argument(self):
        module = ast.parse(
            (ROOT / "q8_kernels" / "functional" / "rope.py").read_text(
                encoding="utf-8"
            )
        )
        rope_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "ROPE"
        )
        backward = next(
            node
            for node in rope_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "backward"
        )
        returns = [
            node
            for node in ast.walk(backward)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
        ]
        self.assertEqual(len(returns), 1)
        self.assertEqual(len(returns[0].value.elts), 4)


if __name__ == "__main__":
    unittest.main()
