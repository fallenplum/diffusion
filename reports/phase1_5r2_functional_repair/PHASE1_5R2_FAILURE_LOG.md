# Phase1.5R2 failure log

- MambaMIL: FAILED_TO_REPRODUCE inherited from Phase1.5R.
- CLAM survival: FAILED_TO_REPRODUCE inherited from Phase1.5R.
- No functional hard gate failed in the final run.

## Recovered development/interruption events

- `initializing` ``: AttributeError: module 'phase1_5r_confirmatory' has no attribute 'canonical_split_hash'
- `fresh_anchor` ``: RuntimeError: STOP_PHASE1_5R2: fresh Phase1R B1 anchor failed the ±0.02 Harrell gate
- `main_neural` `B1_GatedABMIL/seed_20260723/fold_1/K1024/learned256`: INTERRUPTED_BEFORE_FRESH_ANCHOR_GATE; no neural result promoted
- `unknown` `B1_GatedABMIL/seed_20260723/fold_1/K1024/learned256`: NameError: name 'h' is not defined
- `main_neural` ``: NameError: name 'h' is not defined
- `main_neural` ``: IndexError: list index out of range
- `finalize` ``: AttributeError: module 'phase1_5r_confirmatory' has no attribute 'repo_git_revision'
