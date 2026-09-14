import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from PIL import Image


class CLITests(unittest.TestCase):
    def test_integer_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (3, 3)).save(root / "image.png")
            (root / "parameters.json").write_text(json.dumps({"counts": [4], "weights": [1]}))
            result = subprocess.run([sys.executable, "-m", "coincounter.cli.count", "count",
                                     str(root / "image.png"), "--engine", "random_guess",
                                     "--parameters", str(root / "parameters.json")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "4\n")
