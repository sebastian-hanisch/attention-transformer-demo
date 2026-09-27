from streamlit.testing.v1 import AppTest

# Grosszuegig bemessen (Lehre aus lstm-demo/feedback_ci_platform_robust_tests.md:
# 300s reichte dort auf der Linux-CI NICHT, CI-Runner ist ca. 1,8x langsamer als
# lokal). Der T-Sweep hier trainiert nur EIN Modell (nicht RNN+LSTM wie dort),
# aber startet direkt mit hoher Sicherheitsmarge statt erst nach einem CI-Fehlschlag
# nachzubessern.
_APP_TIMEOUT = 600


def _fresh():
    at = AppTest.from_file("../app.py", default_timeout=_APP_TIMEOUT)
    at.run()
    return at


def test_app_runs_without_exception():
    at = _fresh()
    assert not at.exception


def test_footer_is_present():
    at = _fresh()
    captions = [c.value for c in at.caption]
    assert any("Sebastian Hanisch" in c and "Kontakt aufnehmen" in c for c in captions)


def test_preset_kurz_shows_high_accuracy():
    at = _fresh()
    btn = [b for b in at.button if b.label == "Kurze Sequenz — gelingt"][0]
    btn.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert float(metrics["Testgenauigkeit"].rstrip("%")) >= 90.0


def test_hidden_slider_extreme_values_do_not_crash():
    at = _fresh()
    h_slider = [s for s in at.slider if s.label == "Einbettungsdimension"][0]
    h_slider.set_value(h_slider.min).run()
    assert not at.exception
    h_slider = [s for s in at.slider if s.label == "Einbettungsdimension"][0]
    h_slider.set_value(h_slider.max).run()
    assert not at.exception
