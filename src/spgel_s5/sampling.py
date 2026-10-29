"""Training-partition-only controls for the separately locked S5 experiment.

No expression target or model score is used to choose wrong local pairs or
contrastive negatives. Random membership uses only current-train gene summary
statistics. The source arrays, program truth and medical output identities are
never changed in place.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..spgel.contracts import array_hash, require


def _rows_and_statistics(source, train, stats):
    rows = np.asarray(train, dtype=np.int64)
    n = len(source.tissue)
    require(rows.ndim == 1 and len(rows) >= 2 and len(np.unique(rows)) == len(rows)
            and ((rows >= 0) & (rows < n)).all(), "S5_UNIQUE_CURRENT_TRAIN_ROWS_REQUIRED")
    require(stats.fit_rows_sha256 == array_hash(rows), "S5_STATISTICS_TRAIN_SCOPE_MISMATCH")
    boundaries = np.asarray(stats.log_depth_quartiles, dtype=np.float64)
    require(boundaries.shape == (3,) and np.isfinite(boundaries).all()
            and (np.diff(boundaries) >= 0).all(), "S5_TRAIN_DEPTH_BOUNDARIES_REQUIRED")
    return rows, n, boundaries


def _metadata(source, rows, boundaries):
    # Index before computing any statistic: other partitions are not consumed.
    tissue = np.asarray(source.tissue)[rows]
    section = np.asarray(source.section)[rows]
    depth = np.asarray(source.library)[rows].astype(np.float64, copy=False)
    xy = np.asarray(source.xy_um)[rows].astype(np.float64, copy=False)
    require(tissue.shape == section.shape == depth.shape == (len(rows),)
            and xy.shape == (len(rows), 2) and np.isfinite(depth).all()
            and (depth > 0).all() and np.isfinite(xy).all(), "S5_TRAIN_PAIR_METADATA_INVALID")
    log_depth = np.log1p(depth)
    return tissue, section, log_depth, xy, np.searchsorted(boundaries, log_depth, side="right")


def make_wrong_pairs(source, train, stats, seed):
    """Fixed same-tissue/depth-quartile training derangement, not wrong truth.

    Minimize absolute log-depth difference with fixed tiny random tie jitter.
    Exclude own row and same-slide rows within 300 um. If a whole stratum has no
    perfect matching, disable its ITC anchors in ALL arms. Every training row
    still retains its own image regression/tissue supervision.

    partner_global_rows has length N: active train rows map to another current
    train row, inactive train rows map to self solely as a safe bank filler,
    non-train rows are -1. active is the independent eligibility mask.
    """
    rows, n, boundaries = _rows_and_statistics(source, train, stats)
    tissue, section, log_depth, xy, bins = _metadata(source, rows, boundaries)
    rng = np.random.default_rng(int(seed))
    partners = np.full(n, -1, dtype=np.int64)
    partners[rows] = rows
    active = np.zeros(n, dtype=bool)
    groups, absolute_differences = [], []
    for label, quartile in sorted({(int(t), int(b)) for t, b in zip(tissue, bins)}):
        local = np.flatnonzero((tissue == label) & (bins == quartile))
        size = len(local)
        matched = False
        if size >= 2:
            delta = xy[local, None, :] - xy[local][None, :, :]
            distance_squared = np.sum(delta * delta, axis=-1)
            allowed = ((section[local, None] != section[local][None, :])
                       | (distance_squared > 300.**2))
            np.fill_diagonal(allowed, False)
            cost = np.abs(log_depth[local, None] - log_depth[local][None, :])
            cost = cost + rng.random((size, size)) * 1e-12
            cost = np.where(allowed, cost, np.inf)
            try:
                left, right = linear_sum_assignment(cost)
                matched = (len(left) == size and np.isfinite(cost[left, right]).all())
            except ValueError:
                # scipy reports an infeasible assignment when a finite perfect
                # matching does not exist. No looser post-hoc pairing is used.
                matched = False
            if matched:
                a, b = rows[local[left]], rows[local[right]]
                partners[a] = b
                active[a] = True
                absolute_differences.extend(np.abs(log_depth[local[left]] - log_depth[local[right]]).tolist())
        groups.append(dict(tissue_code=label, train_depth_quartile=quartile,
            training_rows=size, complete_derangement=bool(matched),
            ITC_active_rows=size if matched else 0))
    enabled = rows[active[rows]]
    lookup = {int(row): i for i, row in enumerate(rows)}
    if len(enabled):
        assigned = partners[enabled]
        require(np.isin(assigned, rows).all() and (assigned != enabled).all()
                and len(np.unique(assigned)) == len(assigned), "S5_WRONG_PAIR_SCOPE_OR_DERANGEMENT")
        li = np.asarray([lookup[int(x)] for x in enabled])
        ri = np.asarray([lookup[int(x)] for x in assigned])
        require(np.array_equal(tissue[li], tissue[ri]) and np.array_equal(bins[li], bins[ri]),
                "S5_WRONG_PAIR_STRATUM_CHANGED")
        require(np.all((section[li] != section[ri])
                | (np.square(xy[li] - xy[ri]).sum(-1) > 300.**2)), "S5_WRONG_PAIR_LOCAL_TRUE_NEIGHBOR")
    audit = dict(schema="spgel_s5_wrong_training_pairs_v1",
        scope="current_training_partition_only", train_rows_sha256=array_hash(rows),
        mapping_sha256=array_hash(partners[rows]), active_sha256=array_hash(active[rows]),
        training_rows=len(rows), ITC_active_rows=int(active[rows].sum()),
        inactive_train_rows_use_self_only_as_bank_filler=True,
        all_arms_share_supported_anchors=True, regression_and_tissue_truth_unchanged=True,
        same_tissue_and_train_depth_quartile=True, same_slide_exclusion_radius_um=300.,
        pairing_cost="absolute_log1p_library_difference_plus_seeded_1e_minus12_tie_jitter",
        seed=int(seed), groups=groups, incomplete_groups_disable_ALL_their_ITC_anchors=True,
        candidate_gene_values_or_program_targets_used=False, heldout_metadata_used=False,
        paired_donor_identity_preserved=False,
        cross_training_donor_pairs_allowed_and_not_cell_source_proof=True,
        absolute_log_depth_difference_quantiles=(np.quantile(absolute_differences, [0, .5, .9, 1]).tolist()
            if absolute_differences else None))
    return dict(partner_global_rows=partners, active=active, audit=audit)


def make_batch_candidates(source, batch_rows, partner_rows, active, stats, seed):
    """Common image/gene negatives in ORIGINAL bank 0..B-1, never wrong bank.

    partner_rows and active are already gathered into the order of batch_rows.
    Both anchors' own source and wrong source are excluded, including their
    same-slide <=300um neighborhoods. Separate forward/reverse CE must still be
    computed by the caller; sharing negative indices does not merge the losses.
    """
    rows = np.asarray(batch_rows, dtype=np.int64)
    partners = np.asarray(partner_rows, dtype=np.int64)
    enabled = np.asarray(active, dtype=bool).copy()
    n = len(source.tissue)
    require(rows.ndim == 1 and len(rows) and len(np.unique(rows)) == len(rows)
            and partners.shape == enabled.shape == rows.shape
            and ((rows >= 0) & (rows < n)).all()
            and ((partners >= 0) & (partners < n)).all(), "S5_BATCH_PAIR_BANK_ROWS_INVALID")
    allowed_donors = np.asarray(stats.fit_donor_codes)
    require(np.isin(np.asarray(source.donor)[rows], allowed_donors).all()
            and np.isin(np.asarray(source.donor)[partners], allowed_donors).all(),
            "S5_BATCH_OR_PARTNER_CROSSES_TRAIN_DONORS")
    boundaries = np.asarray(stats.log_depth_quartiles, dtype=np.float64)
    tissue, section, depth, xy, bins = _metadata(source, rows, boundaries)
    pt, ps, pd, px, pb = _metadata(source, partners, boundaries)
    require(np.all(~enabled | ((rows != partners) & (tissue == pt) & (bins == pb))),
            "S5_ACTIVE_PAIR_IDENTITY_OR_STRATUM_INVALID")
    rng = np.random.default_rng(int(seed))
    negatives = np.full((len(rows), 8), -1, dtype=np.int64)
    exact, same_tissue, valid_total, no_negative = 0, 0, 0, 0
    for i in range(len(rows)):
        if not enabled[i]:
            continue
        current_far = (section != section[i]) | (np.square(xy - xy[i]).sum(-1) > 300.**2)
        partner_far = (section != ps[i]) | (np.square(xy - px[i]).sum(-1) > 300.**2)
        eligible = current_far & partner_far & (rows != rows[i]) & (rows != partners[i])
        choices = np.flatnonzero(eligible)
        if not len(choices):
            enabled[i] = False
            no_negative += 1
            continue
        priority = (tissue[choices] != tissue[i]).astype(int) * 2 + (bins[choices] != bins[i]).astype(int)
        chosen = choices[np.lexsort((choices, rng.random(len(choices)), priority))[:8]]
        negatives[i, :len(chosen)] = chosen
        exact += int(np.sum((tissue[chosen] == tissue[i]) & (bins[chosen] == bins[i])))
        same_tissue += int(np.sum(tissue[chosen] == tissue[i]))
        valid_total += len(chosen)
    audit = dict(schema="spgel_s5_common_batch_negatives_v1",
        scope="same_current_training_minibatch_original_gene_bank_only",
        batch_rows=len(rows), input_active_anchors=int(np.asarray(active, bool).sum()),
        active_anchors=int(enabled.sum()), no_negative_anchors_disabled_in_all_arms=no_negative,
        requested_negatives_per_anchor=8, valid_negative_count=valid_total,
        same_tissue_depth_negative_count=exact, same_tissue_negative_count=same_tissue,
        own_and_wrong_sources_and_their_same_slide_300um_neighbors_excluded=True,
        directions_share_identical_negative_indices=True, separate_directional_CE_required=True,
        fallback="eligible_nonmatched_tissue_depth_candidates_after_priority_groups",
        program_targets_or_expression_values_used=False,
        true_tissue_labels_used_only_as_training_sampler_metadata=True,
        batch_rows_sha256=array_hash(rows), partners_sha256=array_hash(partners),
        negative_indices_sha256=array_hash(negatives), active_sha256=array_hash(enabled), seed=int(seed))
    return negatives.copy(), negatives.copy(), enabled, audit


def permute_program_membership(source, train, stats, seed):
    """Non-renaming control: only model membership columns are permuted.

    Four true program outputs/targets and measured values/gene identities remain
    fixed. One column bijection preserves each signed row size and all overlaps.
    It is independently randomized within train-only coverage/mean/SD quartile
    bins; singleton bins are explicitly retained and reported.
    """
    rows, _, _ = _rows_and_statistics(source, train, stats)
    values = np.asarray(source.genes)[rows].astype(np.float64, copy=False)
    observed = np.asarray(source.gene_mask)[rows].astype(bool, copy=False)
    up, down = np.asarray(source.up, bool), np.asarray(source.down, bool)
    require(values.ndim == 2 and observed.shape == values.shape
            and up.shape == down.shape == (4, values.shape[1])
            and not (up & down).any() and np.isfinite(values[observed]).all(),
            "S5_RANDOM_MEMBERSHIP_TRAIN_GENE_SHAPES")
    counts = observed.sum(0)
    mean = np.where(observed, values, 0.).sum(0) / np.maximum(counts, 1)
    sd = np.sqrt(np.where(observed, (values - mean)**2, 0.).sum(0) / np.maximum(counts, 1))
    coverage = counts / len(rows)
    columns = []
    edges = {}
    for name, vector in (("coverage", coverage), ("mean", mean), ("sd", sd)):
        boundaries = np.quantile(vector, [.25, .5, .75])
        edges[name] = boundaries.tolist()
        columns.append(np.searchsorted(boundaries, vector, side="right"))
    bins = np.stack(columns, axis=1)
    permutation = np.arange(values.shape[1], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    groups = []
    for key in sorted({tuple(map(int, x)) for x in bins}):
        members = np.flatnonzero(np.all(bins == np.asarray(key), axis=1))
        if len(members) > 1:
            ordered = rng.permutation(members)
            permutation[ordered] = np.roll(ordered, -1)
        groups.append(dict(coverage_mean_sd_bins=list(key), genes=len(members),
            movable=len(members) > 1))
    require(np.array_equal(np.sort(permutation), np.arange(len(permutation)))
            and np.array_equal(bins[permutation], bins), "S5_MEMBERSHIP_NOT_MATCHED_COLUMN_BIJECTION")
    shuffled_up, shuffled_down = up[:, permutation].copy(), down[:, permutation].copy()
    before = np.concatenate((up, down)).astype(np.int64)
    after = np.concatenate((shuffled_up, shuffled_down)).astype(np.int64)
    require(np.array_equal(before.sum(1), after.sum(1))
            and np.array_equal(before @ before.T, after @ after.T),
            "S5_SIGNED_MEMBERSHIP_SIZES_OR_OVERLAPS_CHANGED")
    differences = int(np.sum(before != after))
    require(differences > 0, "S5_RANDOM_MEMBERSHIP_DID_NOT_CHANGE_ANY_EDGE")
    original_mean_coverage = (before @ coverage) / np.maximum(before.sum(1), 1)
    shuffled_mean_coverage = (after @ coverage) / np.maximum(after.sum(1), 1)
    audit = dict(schema="spgel_s5_random_membership_v1", scope="current_training_gene_statistics_only",
        train_rows_sha256=array_hash(rows), seed=int(seed), genes=len(permutation),
        permutation_sha256=array_hash(permutation), original_membership_sha256=array_hash(before),
        shuffled_membership_sha256=array_hash(after), fixed_gene_positions=int(np.sum(permutation == np.arange(len(permutation)))),
        moved_gene_positions=int(np.sum(permutation != np.arange(len(permutation)))),
        changed_signed_membership_cells=differences,
        original_signed_edges=int(before.sum()), shuffled_signed_edges=int(after.sum()),
        signed_row_sizes_exactly_preserved=True, all_signed_pairwise_overlaps_exactly_preserved=True,
        permutation_keeps_coverage_mean_sd_bins=True, bin_boundaries_train_only=edges, groups=groups,
        original_mean_measurement_coverage_by_signed_program=original_mean_coverage.tolist(),
        shuffled_mean_measurement_coverage_by_signed_program=shuffled_mean_coverage.tolist(),
        values_gene_IDs_outputs_and_medical_truth_columns_permuted=False,
        source_arrays_mutated=False, heldout_measurements_used=False,
        random_gene_sets_do_not_claim_medical_program_identity=True)
    return dict(up=shuffled_up, down=shuffled_down, gene_permutation=permutation, audit=audit)
