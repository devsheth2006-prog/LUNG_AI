import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ctcancer.explain.gradcam import GradCAM  # noqa: E402
from ctcancer.models.classifiers import CNN3D, FocalLoss, MultimodalNet, TemperatureScaler, build_classifier  # noqa: E402
from ctcancer.models.unet import DiceCELoss, UNet  # noqa: E402


@pytest.mark.parametrize("dim,shape", [(2, (2, 1, 64, 64)), (3, (1, 1, 32, 32, 32))])
def test_unet_shapes(dim, shape):
    net = UNet(dim, 1, 2, base=4, depth=3 if dim == 3 else 4)
    out = net(torch.randn(shape))
    assert out.shape == (shape[0], 2, *shape[2:])
    loss = DiceCELoss()(out, torch.randint(0, 2, (shape[0], *shape[2:])))
    loss.backward()


@pytest.mark.parametrize("arch", ["resnet18", "densenet121"])
def test_backbone_forward_embed_and_staging(arch):
    m = build_classifier(arch, 2, pretrained=False)
    x = torch.rand(2, 3, 64, 64)
    assert m(x).shape == (2, 2) and m.embed(x).shape == (2, m.emb_dim)
    m.set_stage(1)
    assert not any(p.requires_grad for p in m.net.parameters())
    m.set_stage(2)
    assert any(p.requires_grad for p in m.net.parameters())


@pytest.mark.parametrize("pp", [False, True])
def test_gradcam_2d_and_3d(pp):
    m = build_classifier("resnet18", 2, pretrained=False)
    cam, cls, prob = GradCAM(m, m.gradcam_layer, plus_plus=pp)(torch.rand(1, 3, 64, 64), 1)
    assert cam.shape == (64, 64) and 0 <= cam.min() and cam.max() <= 1 and 0 <= prob <= 1
    m3 = CNN3D()
    cam3, _, _ = GradCAM(m3, m3.gradcam_layer, plus_plus=pp)(torch.rand(1, 1, 32, 32, 32))
    assert cam3.shape == (32, 32, 32)


def test_focal_loss_downweights_easy_examples():
    logits = torch.tensor([[4.0, -4.0], [0.1, 0.0]])
    y = torch.tensor([0, 0])
    fl = FocalLoss(2.0)
    easy = fl(logits[:1], y[:1])
    hard = fl(logits[1:], y[1:])
    assert easy < hard * 0.01


def test_temperature_scaling_reduces_overconfidence():
    torch.manual_seed(0)
    y = torch.randint(0, 2, (500,))
    logits = torch.randn(500, 2) * 0.5
    logits[torch.arange(500), y] += 1.0
    logits *= 6  # overconfident
    ts = TemperatureScaler().fit(logits, y)
    assert ts.temperature > 1.5


def test_multimodal_net_heads():
    net = MultimodalNet(16, 4, {"n_involvement": 2, "subtype": 3})
    out = net(torch.randn(5, 16), torch.randn(5, 4))
    assert out["n_involvement"].shape == (5, 2) and out["subtype"].shape == (5, 3)
    img_only = MultimodalNet(16, 0, {"n_involvement": 2})
    assert img_only(torch.randn(5, 16))["n_involvement"].shape == (5, 2)
    assert np.isfinite(out["subtype"].detach().numpy()).all()
