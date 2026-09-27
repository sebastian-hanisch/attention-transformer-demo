"""Korrektheits-/Verhaltenstests mit billigen Parametern. Die offiziellen,
teuren Sweep-Werte stehen mit Toleranzband in test_claims.py (Modul-Fixtures,
je Sweep nur einmal berechnet)."""
import att_evaluation as ev


def test_sensitivity_sweep_runs_with_small_values():
    rows = ev.sensitivity_sweep(values=(10, 300))
    by_T = {r["T"]: r for r in rows}
    assert by_T[300]["rnn"] < by_T[10]["rnn"]
    assert by_T[300]["attention"] > by_T[300]["rnn"]


def test_t_sweep_short_T_is_easy():
    rows = ev.t_sweep(values=(10,), n_inits=4)
    assert rows[0]["rate"] >= 0.5


def test_analyse_returns_valid_accuracy():
    settings = ev.Settings(T=10, hidden=8, n_train=30, n_test=20, noise=0.3, eta=0.3,
                           epochs=30, seed=0)
    out = ev.analyse(settings)
    assert 0.0 <= out["test_accuracy"] <= 1.0
