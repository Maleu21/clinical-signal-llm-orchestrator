from src.data import load_npz


def test_npz_loading():

    d = load_npz("data/processed/mitbih_windows.npz")

    assert d.X.shape[0] > 0
    assert d.X.shape[1] > 0
    assert len(d.y) == len(d.X)