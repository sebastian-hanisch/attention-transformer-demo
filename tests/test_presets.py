import att_constants as C
import att_evaluation as ev


def test_all_presets_have_valid_settings():
    for key, preset in C.PRESETS.items():
        assert C.T_MIN <= preset["T"] <= C.T_MAX
        assert C.HIDDEN_MIN <= preset["hidden"] <= C.HIDDEN_MAX


def test_preset_kurz_succeeds():
    p = C.PRESETS["kurz"]
    settings = ev.Settings(p["T"], p["hidden"], p["n_train"], p["n_test"], p["noise"],
                           p["eta"], p["epochs"], p["seed"])
    out = ev.analyse(settings)
    assert out["test_accuracy"] >= 0.9


def test_preset_lang_runs_and_produces_valid_accuracy():
    """Rauchtest: ein Einzel-Seed bei T=150 ist chaotisch seed-/
    plattformabhaengig - die robuste Aussage steht in test_claims.py."""
    p = C.PRESETS["lang"]
    settings = ev.Settings(p["T"], p["hidden"], p["n_train"], p["n_test"], p["noise"],
                           p["eta"], p["epochs"], p["seed"])
    out = ev.analyse(settings)
    assert 0.0 <= out["test_accuracy"] <= 1.0
