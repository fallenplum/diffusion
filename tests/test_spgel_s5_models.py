"""Synthetic parity and path isolation; no real data, optimizer or performance."""
import numpy as np
import pytest
import torch

from src.spgel.contracts import GeneBatch, ImageBatch
from src.spgel.losses import bidirectional_contrastive_loss
from src.spgel.models import SpatialProgramEncoder
from src.spgel_s5.models import ARMS, S5ProgramEncoder

torch.set_num_threads(2)


def fixture(b=6, n=None, *, input_grad=False):
    n = b if n is None else n
    rng = np.random.default_rng(1501)
    x = torch.tensor(rng.normal(size=(b, 1536)), dtype=torch.float32)
    x = (x / x.norm(dim=-1, keepdim=True)).requires_grad_(input_grad)
    neighbors = torch.stack((x.detach(), x.detach().roll(1, 0), x.detach().roll(2, 0)), 1)
    xy = torch.tensor([[[0., 0.], [80., 0.], [160., 0.]]] * b)
    image = ImageBatch(x, neighbors, xy, torch.ones((b, 3), dtype=torch.bool))
    values = torch.tensor(rng.normal(size=(n, 24)), dtype=torch.float32, requires_grad=input_grad)
    genes = GeneBatch(values, torch.ones((n, 24), dtype=torch.bool), torch.arange(24))
    up, down = np.zeros((4, 24), bool), np.zeros((4, 24), bool)
    for k in range(4):
        up[k, (np.arange(6) + k*3) % 24] = True
        down[k, (np.arange(6) + k*3 + 8) % 24] = True
    symbols = [f"FIXTURE_GENE_{i}" for i in range(24)]
    return image, genes, up, down, symbols


@pytest.mark.parametrize("new_arm,old_arm", [("M0", "S_A4"), ("M2", "S_A5")])
def test_old_reference_state_image_and_paired_outputs_are_exact(new_arm, old_arm):
    image, genes, up, down, symbols = fixture()
    tissue_mean = np.arange(12, dtype=np.float32).reshape(3, 4) / 20.
    old = SpatialProgramEncoder(old_arm, up, down, symbols, seed=47, tissue_program_mean=tissue_mean)
    new = S5ProgramEncoder(new_arm, up, down, symbols, seed=47, tissue_program_mean=tissue_mean)
    assert set(old.state_dict()) == set(new.state_dict())
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    a, b = old.image_only_predict(image, return_attention=True), new.image_only_predict(image, return_attention=True)
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    ir, gr = torch.tensor([[0, 0, 0], [1, 1, 1]]), torch.tensor([[0, 1, 2], [1, 2, 3]])
    pa = old.paired_scores(a, genes, ir, gr, paired_mode=True, return_details=True)
    pb = new.paired_scores(b, genes, ir, gr, paired_mode=True, return_details=True)
    for key, value in pa.items():
        if torch.is_tensor(value):
            assert torch.equal(value, pb[key]), key
        else:
            assert value == pb[key]


def test_effective_parameter_counts_and_common_initial_states():
    _, _, up, down, symbols = fixture()
    models = {a: S5ProgramEncoder(a, up, down, symbols, seed=31) for a in ARMS}
    expected = dict(M0=300435, M1=333203, M2=333203, M3=316819,
                    M4=316819, M5=333203, M6=333203)
    for arm, model in models.items():
        assert model.descriptor()["parameters"] == expected[arm]
        for name, value in models["M0"].state_dict().items():
            assert torch.equal(value, model.state_dict()[name]), (arm, name)
    for name in ("gene_to_image.weight", "image_to_gene.weight"):
        assert torch.equal(models["M1"].state_dict()[name], models["M2"].state_dict()[name])
    assert "image_to_gene.weight" not in models["M3"].state_dict()
    assert "gene_to_image.weight" not in models["M4"].state_dict()


@pytest.mark.parametrize("arm", ARMS)
def test_image_only_api_and_gradient_do_not_consume_gene_or_conditional_paths(arm):
    image, genes, up, down, symbols = fixture()
    model = S5ProgramEncoder(arm, up, down, symbols)
    with pytest.raises(TypeError):
        model.image_only_predict(image, genes=genes)
    with pytest.raises(TypeError):
        model.image_only_predict(image, tissue_truth=torch.zeros(6))
    with pytest.raises(TypeError):
        model.image_only_predict(image, library_size=torch.ones(6))
    first = model.image_only_predict(image)["program_predictions"]
    first.square().mean().backward()
    assert model.backbone.project[0].weight.grad is not None
    assert all(p.grad is None for p in model.gene_encoder.parameters())
    for name in ("gene_to_image", "image_to_gene"):
        module = getattr(model, name)
        assert module is None or all(p.grad is None for p in module.parameters())
    model.encode_gene_programs(GeneBatch(genes.values * 500, genes.observed, genes.gene_ids), paired_mode=True)
    assert torch.equal(first, model.image_only_predict(image)["program_predictions"])


@pytest.mark.parametrize("arm", ("M1", "M2", "M3", "M4"))
def test_every_registered_conditional_matrix_has_finite_nonzero_itc_gradient(arm):
    image, genes, up, down, symbols = fixture()
    model = S5ProgramEncoder(arm, up, down, symbols)
    output = model.image_only_predict(image)
    negatives = torch.stack((torch.arange(6).roll(1), torch.arange(6).roll(2)), 1)
    loss, audit = bidirectional_contrastive_loss(model, output, genes, negatives)
    assert audit["active_image_anchors"] == audit["active_gene_anchors"] == 6
    loss.backward()
    for name in ("gene_to_image", "image_to_gene"):
        module = getattr(model, name)
        if module is not None:
            assert module.weight.grad is not None
            assert torch.isfinite(module.weight.grad).all() and module.weight.grad.count_nonzero() > 0


@pytest.mark.parametrize("arm,want_gene_to_image,want_image_to_gene", [
    ("M0", False, False), ("M1", False, False), ("M2", True, True),
    ("M3", True, False), ("M4", False, True)])
def test_conditioned_token_cross_paths_are_separated(arm, want_gene_to_image, want_image_to_gene):
    image, genes, up, down, symbols = fixture(input_grad=True)
    model = S5ProgramEncoder(arm, up, down, symbols)
    outputs = model.image_only_predict(image)
    own = torch.arange(6)[:, None]
    result = model.paired_scores(outputs, genes, own, own, paired_mode=True, return_details=True)
    gi = torch.autograd.grad(result["conditioned_image_tokens"].square().sum(), genes.values,
                             retain_graph=True, allow_unused=True)[0]
    ig = torch.autograd.grad(result["conditioned_gene_tokens"].square().sum(), image.features,
                             retain_graph=True, allow_unused=True)[0]
    for gradient, expected in ((gi, want_gene_to_image), (ig, want_image_to_gene)):
        if expected:
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.count_nonzero() > 0
        else:
            assert gradient is None or gradient.count_nonzero() == 0


@pytest.mark.parametrize("arm", ARMS)
def test_separate_image_and_gene_banks_recompute_each_candidate(arm):
    image, genes, up, down, symbols = fixture(b=6, n=12)
    model = S5ProgramEncoder(arm, up, down, symbols)
    outputs = model.image_only_predict(image)
    ir, gr = torch.tensor([[0, 0, 0], [2, 2, 2]]), torch.tensor([[0, 6, 11], [2, 8, 9]])
    result = model.paired_scores(outputs, genes, ir, gr, paired_mode=True, return_details=True)
    assert result["scores"].shape == (2, 3) and result["valid"].all()
    assert torch.isfinite(result["scores"]).all()
    for k in range(3):
        one = model.paired_scores(outputs, genes, ir[:, k:k+1], gr[:, k:k+1],
                                  paired_mode=True, return_details=True)
        for key in ("scores", "conditioned_image_tokens", "conditioned_gene_tokens"):
            assert torch.allclose(result[key][:, k:k+1], one[key], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="SEPARATE_IMAGE_GENE_BANK_ROW_RANGE"):
        model.paired_scores(outputs, genes, torch.tensor([[6]]), torch.tensor([[11]]), paired_mode=True)
    with pytest.raises(ValueError, match="SEPARATE_IMAGE_GENE_BANK_ROW_RANGE"):
        model.paired_scores(outputs, genes, torch.tensor([[5]]), torch.tensor([[12]]), paired_mode=True)


def test_missing_gene_bank_and_pair_mode_fail_closed():
    image, genes, up, down, symbols = fixture(b=6, n=12)
    missing = GeneBatch(torch.full_like(genes.values, float("nan")),
                        torch.zeros_like(genes.observed), genes.gene_ids)
    ir, gr = torch.tensor([[0, 2]]), torch.tensor([[11, 8]])
    for arm in ARMS:
        model = S5ProgramEncoder(arm, up, down, symbols)
        outputs = model.image_only_predict(image)
        with pytest.raises(ValueError, match="PAIRED_SCORING_MODE_REQUIRED"):
            model.paired_scores(outputs, genes, ir, gr)
        result = model.paired_scores(outputs, missing, ir, gr, paired_mode=True)
        assert torch.isfinite(result["scores"]).all()
        assert not result["valid"].any() and result["scores"].count_nonzero() == 0


def test_descriptor_does_not_mislabel_self_reread_or_image_only_predictions():
    _, _, up, down, symbols = fixture()
    for arm in ARMS:
        description = S5ProgramEncoder(arm, up, down, symbols).descriptor()
        assert description["image_only_prediction_uses_conditional_reread"] is False
        assert description["conditional_reread_scope"] == "train_paired_objective_and_TASK_B_only"
        assert description["two_anchor_contrastive_losses_preserved"] is True
    assert S5ProgramEncoder("M1", up, down, symbols).descriptor()["query_context_mode"] == "self_modality"
    with pytest.raises(ValueError, match="S5_UNKNOWN_REGISTERED_ARM"):
        S5ProgramEncoder("M7", up, down, symbols)
