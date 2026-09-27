"""Jede Zahl aus README.md und App wird hier aus den echten Auswertungsfunktionen
neu berechnet. Toleranzband statt exakter Gleichheit von Anfang an (Lehre aus
rnn-demo/lstm-demo: eine einzelne Trainings-Trajektorie ist ueber viele Epochen
chaotisch empfindlich gegenueber Gleitkomma-Rundung zwischen Plattformen)."""
import pytest

import att_constants as C
import att_evaluation as ev


@pytest.fixture(scope="module")
def sensitivity_rows():
    return ev.sensitivity_sweep()


@pytest.fixture(scope="module")
def t_sweep_rows():
    return ev.t_sweep()


def test_claim_sensitivity_rnn_underflows_by_T_300(sensitivity_rows):
    by_T = {r["T"]: r for r in sensitivity_rows}
    assert by_T[300]["rnn"] == 0.0
    assert by_T[600]["rnn"] == 0.0
    assert by_T[1200]["rnn"] == 0.0


def test_claim_sensitivity_attention_stays_usable_at_T_1200(sensitivity_rows):
    by_T = {r["T"]: r for r in sensitivity_rows}
    assert by_T[1200]["attention"] > 1e-4
    assert by_T[1200]["lstm"] < 1e-10  # winzig, aber ungleich RNNs exaktem 0.0


def test_claim_sensitivity_ordering_holds_across_all_T(sensitivity_rows):
    """RNN faellt am schnellsten, Attention am langsamsten - ueber den ganzen
    gemessenen Bereich, nicht nur an den Endpunkten."""
    for row in sensitivity_rows:
        if row["T"] >= 40:  # bei sehr kurzem T noch keine klare Trennung
            assert row["rnn"] <= row["lstm"] <= row["attention"] + 1e-12


def test_claim_t_sweep_short_T_succeeds(t_sweep_rows):
    by_T = {r["T"]: r["rate"] for r in t_sweep_rows}
    assert by_T[10] >= 0.8  # Toleranz statt exaktem 100%


def test_claim_t_sweep_declines_with_T(t_sweep_rows):
    """Ehrlicher Befund: die Erfolgsquote sinkt trotz milder Verduennung mit
    T - kein perfektes T-unabhaengiges Training bei diesem einfachen Aufbau."""
    by_T = {r["T"]: r["rate"] for r in t_sweep_rows}
    assert by_T[150] <= by_T[10]
    assert by_T[150] < 0.5


def test_claim_gradient_check_below_1e_minus_6():
    assert ev.gradient_check() < 1e-6


def test_claim_reduction_check_degenerate_attention_is_exact():
    out = ev.reduction_check_degenerate_attention()
    assert out["identical"]
