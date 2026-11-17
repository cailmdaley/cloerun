import json
import pickle

import numpy as np
import pytest

pytest.importorskip("nautilus")

from cloerun._core import loader  # noqa: E402
from cloerun._core.chain_data import Chain  # noqa: E402
from cloerun._core.config import RunConfig  # noqa: E402
from cloerun._core.priors import build_priors  # noqa: E402
from cloerun._core.setup import LogLikelihood  # noqa: E402
from cloerun.chain import run_chain, test_loglike  # noqa: E402
from cloerun.sdv import write_sdv  # noqa: E402


@pytest.fixture(autouse=True)
def toy_probe(monkeypatch):
    monkeypatch.setitem(loader.PROBES, "toy", "toy_probe")


def _config(tmp_path, **extra):
    raw = {
        "probe": {"kind": "toy"},
        # num_mocks is the toy data vector's first entry.
        "data": {"redshifts": [1.0], "covariance": "unused", "num_mocks": 0.3},
        "fixed_parameters": {"b_{i}": 0.0, "unused_{i}": 1.0, "h": 0.7},
        "sampled_parameters": {
            "a": {"dist": "flat", "min": -1.0, "max": 1.0},
            "omch2": {"dist": "flat", "min": 0.0, "max": 1.0},
            "b1_photo_bin{i}": {"dist": "flat", "min": 0.0, "max": 1.0},
        },
        "fiducial_parameters": {"a": 0.3, "omch2": 0.12},
        "prune_parameters": True,
        "derived_outputs": ["a_plus_b1"],
        "sampler": {
            "kind": "nautilus",
            "settings": {"n_live": 100, "n_networks": 1, "seed": 1},
            "run_settings": {"verbose": False, "n_eff": 300},
        },
        "output": {
            "directory": str(tmp_path / "out"), "name": "toy",
            "sdv_file": str(tmp_path / "out" / "sdv.json"),
        },
        **extra,
    }
    return RunConfig.from_dict(raw, tmp_path)


def test_test_mode(tmp_path, capsys):
    assert test_loglike(_config(tmp_path)) == 0.0
    out = capsys.readouterr().out
    # Templates and names the likelihood does not read are dropped, except
    # the inputs of derived-parameter relations (omch2, h).
    assert "dropped: ['b1_photo_bin{i}', 'unused_{i}']" in out


def test_sdv_mode(tmp_path):
    path = write_sdv(_config(tmp_path))
    assert json.loads(path.read_text()) == {"theory": [0.3, 0.0]}


def test_chain_mode(tmp_path):
    chain = run_chain(_config(tmp_path))
    assert chain.columns == ["logweight", "loglikelihood", "a", "omch2", "a_plus_b1"]
    weights = np.exp(chain.logweight - chain.logweight.max())
    mean = np.average(chain.column("a"), weights=weights)
    assert mean == pytest.approx(0.3, abs=0.03)
    np.testing.assert_allclose(chain.column("a_plus_b1"), chain.column("a"))
    saved = Chain.load(tmp_path / "out" / "toy.npz")
    assert saved.columns == chain.columns
    assert (tmp_path / "out" / "toy.config.yaml").exists()


def test_noise_free_mock_uses_fiducial(tmp_path):
    config = _config(tmp_path)
    config.data.num_mocks = 0.9
    # The toy probe uses the spectroscopic data layout, which has no mock flag.
    config.data.noise_free_mock = True
    config.fiducial_parameters = {"a": -0.2, "omch2": 0.12}
    # Without the mock the data vector would be 0.9.
    assert test_loglike(config) == 0.0


def test_loglikelihood_pickles_without_likelihood(tmp_path):
    config = _config(tmp_path)
    priors = build_priors({"a": {"dist": "flat", "min": -1.0, "max": 1.0}}, [])
    loglike = LogLikelihood(
        config, {"b_1": 0.0, "b_2": 0.0, "h": 0.7, "omch2": 0.12}, priors.layout,
        data_vector=np.array([0.5, 0.0]),
    )
    assert loglike({"a": 0.5}) == (0.0, 0.5)
    copy = pickle.loads(pickle.dumps(loglike))
    assert copy._likelihood is None
    # Rebuilt in the copy, with the data vector set again.
    assert copy({"a": 0.4}) == pytest.approx((-0.5, 0.4))


def _loglikelihood(tmp_path):
    priors = build_priors({"a": {"dist": "flat", "min": -1.0, "max": 1.0}}, [])
    return LogLikelihood(
        _config(tmp_path), {"b_1": 0.0, "b_2": 0.0, "h": 0.7, "omch2": 0.12}, priors.layout,
        data_vector=np.array([0.5, 0.0]),
    )


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_non_finite_loglike_is_a_failed_point(tmp_path, monkeypatch, capsys, value):
    import toy_probe

    monkeypatch.setattr(toy_probe, "loglike", lambda likelihood, parameters: value)
    loglike, derived = _loglikelihood(tmp_path)({"a": 0.1})
    assert loglike == -np.inf and np.isnan(derived)
    assert "[loglike] non-finite value" in capsys.readouterr().err


def test_loglike_error_is_a_failed_point(tmp_path, monkeypatch, capsys):
    import toy_probe

    def fail(likelihood, parameters):
        raise ValueError("emulator out of range")

    monkeypatch.setattr(toy_probe, "loglike", fail)
    loglike, derived = _loglikelihood(tmp_path)({"a": 0.1})
    assert loglike == -np.inf and np.isnan(derived)
    assert "[loglike] ValueError: emulator out of range at {'a': 0.1}" in capsys.readouterr().err


def test_no_data_vector(tmp_path):
    config = _config(tmp_path)
    config.data.cells = None
    with pytest.raises(ValueError, match="No data vector"):
        test_loglike(config)


def test_test_loglike_uses_prior_median_without_fiducial(tmp_path, capsys):
    config = _config(tmp_path)
    config.fiducial_parameters = {}
    # The median of the flat prior on a is 0, the data 0.3.
    assert test_loglike(config) == pytest.approx(-0.5 * 0.3**2 / 0.1**2)
    assert "prior median used for: ['a', 'omch2']" in capsys.readouterr().out


def test_sdv_needs_a_probe_that_writes_spectra(tmp_path, monkeypatch):
    import toy_probe

    monkeypatch.delattr(toy_probe, "theory_spectra")
    with pytest.raises(ValueError, match="cannot write a synthetic data vector"):
        write_sdv(_config(tmp_path))


def test_vectorized_loglike_matches_scalar(tmp_path):
    loglike = _loglikelihood(tmp_path)
    a = np.array([-0.9, 0.1, 0.5, 0.7])
    loglike.config.sampler["settings"]["vectorized"] = True
    values, derived = loglike({"a": a})
    loglike.config.sampler["settings"]["vectorized"] = False
    scalar = np.array([loglike({"a": x}) for x in a])
    np.testing.assert_allclose(values, scalar[:, 0])
    np.testing.assert_allclose(derived, scalar[:, 1])


def test_vectorized_chain(tmp_path):
    config = _config(tmp_path)
    config.sampler["settings"].update(vectorized=True, n_batch=100)
    chain = run_chain(config)
    weights = np.exp(chain.logweight - chain.logweight.max())
    assert np.average(chain.column("a"), weights=weights) == pytest.approx(0.3, abs=0.03)
    np.testing.assert_allclose(chain.column("a_plus_b1"), chain.column("a"))


def test_vectorized_constraint_and_non_finite(tmp_path, monkeypatch, capsys):
    import toy_probe

    config = _config(tmp_path, constraints={"negative_w0_wa_sum": True})
    config.sampler["settings"]["vectorized"] = True
    priors = build_priors({"a": {"dist": "flat", "min": -1.0, "max": 1.0},
                           "wa": {"dist": "flat", "min": -1.0, "max": 2.0}}, [])
    loglike = LogLikelihood(
        config, {"b_1": 0.0, "b_2": 0.0, "h": 0.7, "omch2": 0.12, "w0": -1.0}, priors.layout,
        data_vector=np.array([0.5, 0.0]),
    )
    batch = toy_probe.loglike_batch

    def nan_at_second(likelihood, parameters):
        values = batch(likelihood, parameters)
        values[1] = np.nan
        return values

    monkeypatch.setattr(toy_probe, "loglike_batch", nan_at_second)
    # The third point has w0 + wa > 0.
    values, derived = loglike({"a": np.array([0.5, 0.4, 0.3, 0.4]),
                               "wa": np.array([0.0, 0.5, 1.5, 0.9])})
    np.testing.assert_allclose(values, [0.0, -np.inf, -np.inf, -0.5])
    np.testing.assert_allclose(derived, [0.5, np.nan, np.nan, 0.4])
    assert "1 non-finite values in a batch of 4" in capsys.readouterr().err


def test_vectorized_needs_loglike_batch(tmp_path, monkeypatch):
    import toy_probe

    monkeypatch.delattr(toy_probe, "loglike_batch")
    config = _config(tmp_path)
    config.sampler["settings"]["vectorized"] = True
    with pytest.raises(ValueError, match="has no loglike_batch"):
        run_chain(config)
