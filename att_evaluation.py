"""Kennzahlen: Settings-Dataclass, analyse()-Einstiegspunkt,
Sensitivitaets-Sweep (schnell, kein Training), Kaltstart-T-Sweep (teuer),
Gradienten-Check, Korrektheits-Kette (entartete Aufmerksamkeit)."""
from dataclasses import dataclass

import numpy as np

import att_constants as C
import att_model as m
import att_scenario as sc


@dataclass(frozen=True)
class Settings:
    T: int
    hidden: int
    n_train: int
    n_test: int
    noise: float
    eta: float
    epochs: int
    seed: int


def analyse(settings: Settings) -> dict:
    train_ds = sc.make_dataset(settings.n_train, settings.T, settings.seed, noise_std=settings.noise)
    test_ds = sc.make_dataset(settings.n_test, settings.T, settings.seed + 1000, noise_std=settings.noise)
    att = m.Attention(settings.hidden, seed=settings.seed)
    optimizer = m.SGD(settings.eta)
    result = m.train(att, optimizer, train_ds.X, train_ds.y, settings.epochs)
    return {"train_ds": train_ds, "test_ds": test_ds, "model": att, "result": result,
            "test_accuracy": m.accuracy(att, test_ds.X, test_ds.y)}


def sensitivity_sweep(values=C.SENSITIVITY_T_VALUES, hidden: int = C.HIDDEN_DEFAULT) -> list:
    """|d(score)/d(x_0)| bei UNTRAINIERTER Zufallsinitialisierung, RNN vs.
    LSTM vs. Attention - schnell, deterministisch, kein Training noetig
    (Lehre aus Stueck 5: eine trainingsfreie, robuste Messung vermeidet die
    Plattform-/Seed-Chaos-Probleme chaotischer Trainings-Trajektorien)."""
    rows = []
    for T in values:
        row = {"T": T}
        row["rnn"] = m.sensitivity_to_signal(m.RNN(hidden, seed=0), T)
        row["lstm"] = m.sensitivity_to_signal(m.LSTM(hidden, seed=0), T)
        row["attention"] = m.sensitivity_to_signal(m.Attention(hidden, seed=0), T)
        rows.append(row)
    return rows


def t_sweep(
    values=C.T_SWEEP_VALUES, n_inits=C.T_SWEEP_INITS, n_train=C.N_TRAIN_DEFAULT,
    n_test=C.N_TEST_DEFAULT, noise=C.NOISE_DEFAULT, hidden=C.HIDDEN_DEFAULT,
    eta=C.ETA_DEFAULT, epochs=C.EPOCHS_DEFAULT,
) -> list:
    """Kaltstart-Erfolgsquote (SGD, kein Curriculum) ueber n_inits
    Zufalls-Inits, fuer jedes T. Ehrlicher Befund (siehe README): trotz der
    viel milderen Sensitivitaets-Verduennung bleibt das Training bei diesem
    einfachen Aufbau (ein Kopf, keine Stapelverarbeitung) jenseits von T~10
    schwierig - strukturelles Potenzial allein reicht nicht."""
    rows = []
    for T in values:
        successes = 0
        for seed in range(n_inits):
            settings = Settings(T=T, hidden=hidden, n_train=n_train, n_test=n_test, noise=noise,
                               eta=eta, epochs=epochs, seed=seed)
            out = analyse(settings)
            if out["test_accuracy"] >= 0.9:
                successes += 1
        rows.append({"T": T, "rate": successes / n_inits})
    return rows


def gradient_check(T: int = 6, hidden: int = 4, seed: int = 2, eps: float = 1e-5) -> float:
    att = m.Attention(hidden, seed=seed)
    rng = np.random.default_rng(1)
    seq, _ = sc.make_sequence(rng, T)
    s0, cache = att.forward(seq)
    y = -np.sign(s0) if s0 != 0 else 1.0
    grads, _ = att.backward(y, s0, cache)

    def loss() -> float:
        s, _ = att.forward(seq)
        return max(0.0, -y * s)

    max_rel_err = 0.0
    for key, grad in grads.items():
        param = att.params[key]
        flat_param = param.reshape(-1)
        flat_grad = grad.reshape(-1)
        for idx in range(flat_grad.size):
            orig = flat_param[idx]
            flat_param[idx] = orig + eps
            l_plus = loss()
            flat_param[idx] = orig - eps
            l_minus = loss()
            flat_param[idx] = orig
            numeric = (l_plus - l_minus) / (2 * eps)
            analytic = float(flat_grad[idx])
            denom = max(abs(numeric), abs(analytic), 1e-12)
            max_rel_err = max(max_rel_err, abs(numeric - analytic) / denom)
    return max_rel_err


def reduction_check_degenerate_attention(hidden: int = 4, T: int = 5, focus_pos: int = 2,
                                         seed: int = 0) -> dict:
    """Aufmerksamkeit mit einer einzigen Position im Fokus (entartete,
    quasi-One-Hot-Verteilung): der Ausgang muss auf den WERT an dieser einen
    Position hinauslaufen (out=V[focus_pos]), von Hand nachrechenbar -
    unabhaengig von allen anderen Positionen."""
    att = m.Attention(hidden, seed=seed)
    rng = np.random.default_rng(3)
    seq = rng.normal(0, 1, size=T)
    _, cache = att.forward(seq)
    V = cache["V"]
    # Entartete Verteilung von Hand konstruiert (extrem auf focus_pos konzentriert)
    scores = np.full(T, -50.0)
    scores[focus_pos] = 50.0
    alpha_degenerate = m.softmax(scores)
    out_degenerate = alpha_degenerate @ V
    s_degenerate = float(att.params["Why"] @ out_degenerate + att.params["by"])
    s_manual = float(att.params["Why"] @ V[focus_pos] + att.params["by"])
    return {"alpha_max": float(alpha_degenerate.max()), "alpha_argmax": int(alpha_degenerate.argmax()),
            "s_degenerate": s_degenerate, "s_manual": s_manual,
            "identical": bool(np.isclose(s_degenerate, s_manual, atol=1e-6))}
