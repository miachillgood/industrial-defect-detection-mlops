"""CLI argument parsing.

``--compute-pro`` has to work two ways: as a bare flag for humans, and with an
explicit value so ``dvc.yaml`` can template ``${evaluate.compute_pro}`` (DVC has
no conditionals, so the flag is always emitted).
"""

from __future__ import annotations

import argparse

import pytest

from spade.evaluate import _bool_arg, parse_args


def test_defaults():
    args = parse_args([])
    # PRO is on by default: measured at ~18 s across all 15 categories (~2 % of a
    # run), so there is no cost worth trading the extra metric for.
    assert args.compute_pro is True
    assert args.top_k == 5
    assert args.categories == "all"
    assert args.device == "auto"


def test_compute_pro_as_bare_flag():
    assert parse_args(["--compute-pro"]).compute_pro is True


@pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "on"])
def test_compute_pro_truthy_values(value):
    assert parse_args(["--compute-pro", value]).compute_pro is True


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off"])
def test_compute_pro_falsy_values(value):
    assert parse_args(["--compute-pro", value]).compute_pro is False


def test_compute_pro_rejects_garbage():
    with pytest.raises(SystemExit):
        parse_args(["--compute-pro", "maybe"])


def test_bool_arg_passes_through_actual_bools():
    assert _bool_arg(True) is True
    assert _bool_arg(False) is False


def test_bool_arg_raises_on_nonsense():
    with pytest.raises(argparse.ArgumentTypeError):
        _bool_arg("perhaps")


def test_categories_stay_a_string_for_main_to_split():
    assert parse_args(["--categories", "bottle,cable"]).categories == "bottle,cable"


def test_device_choices_are_enforced():
    with pytest.raises(SystemExit):
        parse_args(["--device", "tpu"])


def test_dvc_style_invocation_parses():
    """Exactly the shape dvc.yaml emits."""
    args = parse_args(
        [
            "--categories", "all",
            "--top-k", "5",
            "--device", "auto",
            "--batch-size", "32",
            "--num-workers", "6",
            "--bank-dtype", "float32",
            "--save-visualizations", "5",
            "--compute-pro", "true",
            "--output-dir", "artifacts/runs",
            "--run-name", "full-mvtec-k5",
        ]
    )
    assert args.compute_pro is True
    assert args.run_name == "full-mvtec-k5"
    assert args.num_workers == 6
