import json
import tempfile
import unittest
from pathlib import Path

from stage2_official.build_context import build, required_modules


class ImageContextTest(unittest.TestCase):
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
