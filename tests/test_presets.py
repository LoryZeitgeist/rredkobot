import importlib
import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional


class PresetImagesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = Path(tempfile.mkdtemp())
        os.environ["PRESETS_DIR"] = str(self.tempdir)
        # Reload the bot module to apply the temporary presets directory.
        self.bot = importlib.reload(importlib.import_module("bot"))
        # Ensure directory structure exists.
        self.bot.PRESETS_DIR = self.tempdir.resolve()
        self.bot.ensure_presets_directory()

    def tearDown(self) -> None:
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def create_image_files(self, filenames) -> None:
        for name in filenames:
            path = self.tempdir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test-image")

    def test_images_cycle_without_repeats(self) -> None:
        filenames = ["first.jpg", "second.png", "third.webp"]
        self.create_image_files(filenames)

        trigger = {"images": filenames}
        bot_data = {}
        # Ensure deterministic shuffle for the purpose of the test.
        random.seed(12345)

        seen = []
        for _ in range(len(filenames)):
            image_path = self.bot.get_next_trigger_image(1, "test_preset", 0, trigger, bot_data)
            self.assertIsNotNone(image_path)
            seen.append(image_path.name)

        self.assertEqual(len(seen), len(set(seen)))
        self.assertSetEqual(set(seen), set(filenames))

        # Second cycle should again cover all files (order may differ).
        cycle_two = {
            self.bot.get_next_trigger_image(1, "test_preset", 0, trigger, bot_data).name
            for _ in range(len(filenames))
        }
        self.assertSetEqual(cycle_two, set(filenames))

    def test_list_preset_files_handles_allowed_extensions(self) -> None:
        filenames = [
            "one.jpg",
            "two.jpeg",
            "three.png",
            "four.webp",
            "ignore.txt",
            "sample.json",
        ]
        self.create_image_files(filenames)

        files = self.bot.list_preset_files()
        returned = {file_path.name for file_path in files}

        self.assertSetEqual(returned, {"one.jpg", "two.jpeg", "three.png", "four.webp"})
        self.assertNotIn("ignore.txt", returned)
        self.assertNotIn("sample.json", returned)


class PathResolutionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.original_presets_dir = os.environ.get("PRESETS_DIR")
        self.original_state_file = os.environ.get("STATE_FILE")

    def tearDown(self) -> None:
        self._restore_env("PRESETS_DIR", self.original_presets_dir)
        self._restore_env("STATE_FILE", self.original_state_file)

    def _restore_env(self, name: str, value: Optional[str]) -> None:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def test_relative_paths_resolve_from_project_root(self) -> None:
        os.environ["PRESETS_DIR"] = "relative_presets"
        os.environ["STATE_FILE"] = "relative_state/state.json"
        bot_module = importlib.reload(importlib.import_module("bot"))
        project_root = Path(bot_module.__file__).resolve().parent

        self.assertEqual(bot_module.PRESETS_DIR, (project_root / "relative_presets").resolve())
        self.assertEqual(
            bot_module.STATE_FILE,
            (project_root / "relative_state/state.json").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
