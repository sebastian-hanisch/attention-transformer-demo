"""Regler-Grenzen, feste Annahmen, gemessene Werte und Presets."""

DEFAULT_SEED = 0
SIGNAL_POS = 0

T_MIN, T_MAX, T_DEFAULT = 2, 1200, 40
HIDDEN_MIN, HIDDEN_MAX, HIDDEN_DEFAULT = 2, 16, 8
N_TRAIN_MIN, N_TRAIN_MAX, N_TRAIN_DEFAULT = 20, 120, 60
N_TEST_MIN, N_TEST_MAX, N_TEST_DEFAULT = 20, 100, 40
NOISE_MIN, NOISE_MAX, NOISE_DEFAULT = 0.0, 0.6, 0.3
ETA_MIN, ETA_MAX, ETA_DEFAULT = 0.05, 0.6, 0.3
EPOCHS_MIN, EPOCHS_MAX, EPOCHS_DEFAULT = 10, 100, 60

# Sensitivitaets-Sweep (schnell, KEIN Training noetig - robust/deterministisch)
SENSITIVITY_T_VALUES = (2, 10, 40, 80, 150, 300, 600, 1200)

# Kaltstart-Erfolgsquote-Sweep (teuer, mit Spinner)
T_SWEEP_VALUES = (10, 40, 80, 150)
T_SWEEP_INITS = 6

PRESETS = {
    "kurz": dict(
        label="Kurze Sequenz — gelingt",
        T=10, hidden=8, n_train=60, n_test=40, noise=0.3, eta=0.3, epochs=60, seed=0,
        help="Bei T=10 lernt Attention zuverlässig - die Aufmerksamkeit konzentriert sich "
             "sichtbar auf die Signalposition.",
    ),
    "lang": dict(
        label="Lange Sequenz — Training bleibt schwer",
        T=150, hidden=8, n_train=60, n_test=40, noise=0.3, eta=0.3, epochs=60, seed=0,
        help="Die Empfindlichkeit auf die Signalposition bleibt strukturell hoch (siehe "
             "Sensitivitäts-Chart) - aber dieses einfache Training (ein Kopf, keine "
             "Stapelverarbeitung) findet hier trotzdem oft keine Lösung.",
    ),
}
