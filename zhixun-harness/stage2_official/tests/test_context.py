import json
import tempfile
import unittest
from pathlib import Path

from stage2_official.build_context import build, required_modules


class ImageContextTest(unittest.TestCase):
    def test_multiline_import_is_parsed(self):
        # A parenthesized import is legal Python; a single-line regex would miss every name
        # after the first and report a healthy image as broken.
        with tempfile.TemporaryDirectory() as tmp:
            entry = Path(tmp) / "entry.py"
            entry.write_text("from stage2_official import (\n admission,\n evolution,\n)\n",
                             encoding="utf-8")
            self.assertEqual(required_modules(entry), {"admission", "evolution"})

    def test_every_imported_module_reaches_the_image(self):
        # The rehearsal runs from source, so only the manifest can catch a missing file.
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "context"
            build(output)
            manifest = json.loads((output / "context-manifest.json").read_text(encoding="utf-8"))
            for origin in sorted((output / "app").rglob("*.py")):
                for module in required_modules(origin):
                    self.assertIn(f"app/stage2_official/{module}.py", manifest, origin.name)
            self.assertIn("app/stage2_official/attribution.py", manifest)


if __name__ == "__main__":
    unittest.main()
