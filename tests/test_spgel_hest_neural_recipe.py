import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.spgel_hest_neural.recipe import REPO, SOURCE_FILES, recipe


class RecipeTests(unittest.TestCase):
    def test_current_actual_fixed_recipe_and_exact_budget(self):
        current = recipe(REPO / "configs/spgel_hest_neural/pilot_v1.json")
        self.assertEqual(current["budget"]["neural_fits"], 12)
        self.assertEqual(current["budget"]["actual_state_replays"], 36)
        self.assertEqual(current["budget"]["optimizer_updates_total"], 480)
        self.assertEqual(current["budget"]["new_unsupervised_PCA_reconstructions"], 4)
        self.assertEqual(len(SOURCE_FILES), len(set(SOURCE_FILES)))
        self.assertNotIn("src/spgel_hest/execution.py", SOURCE_FILES)

    def test_epoch_model_or_new_loss_changes_rejected(self):
        current = recipe(REPO / "configs/spgel_hest_neural/pilot_v1.json")
        with tempfile.TemporaryDirectory(dir=REPO / "tmp") as directory:
            target = Path(directory) / "changed.json"
            for section, key, value in [("budget", "epochs_per_fit", 80), ("architecture", "hidden", 256),
                                        ("training", "loss", "Cox"), ("runtime", "device", "CUDA")]:
                changed = copy.deepcopy(current)
                changed[section][key] = value
                target.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "RECIPE_CHANGED"):
                    recipe(target)


if __name__ == "__main__":
    unittest.main()
