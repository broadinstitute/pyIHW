import numpy as np
import pandas as pd
import pytest

from pyihw import calibration_cells_per_grna, ihw, ihw_with_power


def test_ihw_with_power_aligns_power_by_pair():
    rng = np.random.default_rng(0)
    n = 3000
    results = pd.DataFrame(
        {
            "response_id": [f"g{i}" for i in range(n)],
            "grna_target": [f"t{i % 50}" for i in range(n)],
            "p_value": rng.random(n),
        }
    )
    results.loc[3, "p_value"] = np.nan
    power = results[["response_id", "grna_target"]].assign(power=rng.random(n))
    shuffled = power.sample(frac=1, random_state=0)

    weighted = ihw_with_power(results, shuffled, alpha=0.1, nbins=4)
    direct = ihw(results["p_value"], power["power"], 0.1, nbins=4)

    np.testing.assert_array_equal(weighted["ihw_adj_p_value"], direct.adj_pvalues())
    assert pd.isna(weighted.loc[3, "ihw_rejected"])
    assert weighted.attrs["ihw_result"].nbins == 4


def test_ihw_with_power_rejects_unmatched_pairs():
    results = pd.DataFrame(
        {"response_id": ["a", "b"], "grna_target": ["t", "t"], "p_value": [0.1, 0.2]}
    )
    power = pd.DataFrame({"response_id": ["a"], "grna_target": ["t"], "power": [0.5]})

    with pytest.raises(ValueError, match="power estimate"):
        ihw_with_power(results, power, alpha=0.1)


def test_calibration_cells_per_grna_splits_synthetic_targets():
    calibration = pd.DataFrame(
        {"response_id": ["g1", "g2", "g3"], "grna_target": ["nt1&nt2", "nt1&nt2", "nt3&nt1"]}
    )
    cells = {"nt1": np.arange(5), "nt2": np.arange(3), "nt3": np.arange(7)}

    counts = calibration_cells_per_grna(calibration, cells)

    assert counts.to_dict("records") == [
        {"grna_id": "nt1", "grna_target": "nt1&nt2", "num_cells": 5},
        {"grna_id": "nt2", "grna_target": "nt1&nt2", "num_cells": 3},
        {"grna_id": "nt3", "grna_target": "nt3&nt1", "num_cells": 7},
        {"grna_id": "nt1", "grna_target": "nt3&nt1", "num_cells": 5},
    ]
