from granular_mpm import SandMPM3DConfig, ToolState3D


def test_public_imports() -> None:
    cfg = SandMPM3DConfig()
    assert cfg.dx > 0.0
    assert ToolState3D.__name__ == "ToolState3D"
