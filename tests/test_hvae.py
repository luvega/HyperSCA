"""H-VAE 模型单元测试"""
import pytest
import torch
import numpy as np

from src.models.hyperbolic.hvae import HyperbolicVAE, nb_loss, zinb_loss


@pytest.fixture
def small_model():
    return HyperbolicVAE(
        input_dim=500,
        latent_dim=8,
        encoder_layers=[128, 64],
        decoder_layers=[64, 128],
        gcn_layers=1,
        beta=1.0,
        gamma=0.1,
    )


@pytest.fixture
def synthetic_data():
    N, G = 50, 500
    x = torch.randn(N, G)
    x_raw = torch.randint(0, 50, (N, G)).float()
    edge_index = torch.randint(0, N, (2, 200))
    edge_weight = torch.rand(200)
    return x, x_raw, edge_index, edge_weight


class TestHVAEForward:
    def test_output_shapes(self, small_model, synthetic_data):
        x, _, edge_index, edge_weight = synthetic_data
        out = small_model(x, edge_index, edge_weight)

        N = x.shape[0]
        d = small_model.latent_dim
        G = x.shape[1]

        assert out["z"].shape == (N, d + 1)
        assert out["z_poincare"].shape == (N, d)
        assert out["mu"].shape == (N, d + 1)
        assert out["recon_mean"].shape == (N, G)
        assert out["recon_disp"].shape == (N, G)

    def test_poincare_norm_bounded(self, small_model, synthetic_data):
        x, _, edge_index, edge_weight = synthetic_data
        out = small_model(x, edge_index, edge_weight)
        norms = torch.norm(out["z_poincare"], dim=-1)
        assert torch.all(norms < 5.0)  # reasonable bound

    def test_no_nan_in_output(self, small_model, synthetic_data):
        x, _, edge_index, edge_weight = synthetic_data
        out = small_model(x, edge_index, edge_weight)
        for key in ["z", "z_poincare", "recon_mean", "recon_disp"]:
            assert not torch.isnan(out[key]).any(), f"NaN in {key}"


class TestHVAELoss:
    def test_loss_finite(self, small_model, synthetic_data):
        x, x_raw, edge_index, edge_weight = synthetic_data
        out = small_model(x, edge_index, edge_weight)
        loss = small_model.loss_function(x_raw, out, edge_index, x.shape[0])

        for key in ["total", "recon", "kl", "topo"]:
            assert torch.isfinite(loss[key]), f"Non-finite {key}: {loss[key]}"

    def test_backward_pass(self, small_model, synthetic_data):
        x, x_raw, edge_index, edge_weight = synthetic_data
        out = small_model(x, edge_index, edge_weight)
        loss = small_model.loss_function(x_raw, out, edge_index, x.shape[0])
        loss["total"].backward()

        # Check gradients exist
        has_grad = any(p.grad is not None for p in small_model.parameters())
        assert has_grad


class TestHVAEEmbeddings:
    def test_get_embeddings(self, small_model, synthetic_data):
        x, _, edge_index, edge_weight = synthetic_data
        poincare, lorentz = small_model.get_embeddings(x, edge_index, edge_weight)

        assert isinstance(poincare, np.ndarray)
        assert isinstance(lorentz, np.ndarray)
        assert poincare.shape[0] == x.shape[0]
        assert lorentz.shape[0] == x.shape[0]


class TestNBLoss:
    def test_nb_loss_shape(self):
        x = torch.randint(0, 50, (10, 100)).float()
        mean = torch.rand(10, 100) * 10 + 0.1
        disp = torch.ones(10, 100) * 5
        loss = nb_loss(x, mean, disp)
        assert loss.shape == (10,)

    def test_nb_loss_finite(self):
        x = torch.randint(0, 50, (10, 100)).float()
        mean = torch.rand(10, 100) * 10 + 0.1
        disp = torch.ones(10, 100) * 5
        loss = nb_loss(x, mean, disp)
        assert torch.isfinite(loss).all()
