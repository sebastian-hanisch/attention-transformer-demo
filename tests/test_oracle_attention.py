"""Unabhängige Orakel für Attention, RNN/LSTM-Referenzen und die Empfindlichkeitsmessung:

* Empfindlichkeit |ds/dx_0| gegen eine eigene Vorwärts-Tangentenrechnung (von Hand
  abgeleitet, anderer Rechenweg als Complex-Step) bis T=1200 - einschließlich der winzigen
  Werte (RNN ~1e-83), die eine finite Differenz als 0,0 meldet,
* dieselbe Empfindlichkeit gegen eine zentrale finite Differenz bei kleinem T,
* Rückwärtspass der Attention gegen den Complex-Step-Gradienten einer eigenen Vorwärtsrechnung
  (Softmax von Hand).
"""
import numpy as np

import att_model as M
import att_scenario as sc


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def tangent_rnn(net, seq):
    p = net.params
    h, dh = np.zeros(net.H), np.zeros(net.H)
    for t, x in enumerate(seq):
        dx = 1.0 if t == 0 else 0.0
        h_new = np.tanh(p["Wxh"][:, 0] * x + p["Whh"] @ h + p["bh"])
        dh = (1 - h_new ** 2) * (p["Wxh"][:, 0] * dx + p["Whh"] @ dh)
        h = h_new
    return abs(p["Why"] @ dh)


def tangent_lstm(net, seq):
    p, H = net.params, net.H
    h, c, dh, dc = (np.zeros(H) for _ in range(4))
    for t, x in enumerate(seq):
        dx = 1.0 if t == 0 else 0.0
        pre = {g: p["W" + g][:, 0] * x + p["U" + g] @ h + p["b" + g] for g in "fiog"}
        dpre = {g: p["W" + g][:, 0] * dx + p["U" + g] @ dh for g in "fiog"}
        f, i, o, g = _sig(pre["f"]), _sig(pre["i"]), _sig(pre["o"]), np.tanh(pre["g"])
        df, di, do = f * (1 - f) * dpre["f"], i * (1 - i) * dpre["i"], o * (1 - o) * dpre["o"]
        dg = (1 - g ** 2) * dpre["g"]
        c_new, dc_new = f * c + i * g, df * c + f * dc + di * g + i * dg
        tc = np.tanh(c_new)
        h, dh = o * tc, do * tc + o * (1 - tc ** 2) * dc_new
        c, dc = c_new, dc_new
    return abs(p["Why"] @ dh)


def tangent_attention(net, seq):
    p, d, T = net.params, net.d, len(seq)
    E = np.array([p["We"][:, 0] * x + p["be"] for x in seq]) \
        + M.sinusoidal_positional_encoding(T, d)
    V = E @ p["Wv"].T
    z = (E @ p["Wk"].T) @ p["q"] / np.sqrt(d)
    a = np.exp(z - z.max())
    a /= a.sum()
    dE0 = p["We"][:, 0]  # nur Position 0 hängt von x_0 ab
    dz = np.zeros(T)
    dz[0] = (p["Wk"] @ dE0) @ p["q"] / np.sqrt(d)
    da = a * (dz - np.sum(a * dz))
    dout = da @ V + a[0] * (p["Wv"] @ dE0)
    return abs(p["Why"] @ dout)


def test_sensitivity_matches_independent_tangent_propagation_including_tiny_values():
    rng = np.random.default_rng(0)
    for _ in range(200):
        H = int(rng.integers(3, 9))  # H>=3: Werte bleiben über dem Unterlaufbereich
        T = int(rng.choice([2, 3, 5, 10, 40, 150, 300, 600, 1200]))
        seed, sseed = int(rng.integers(0, 500)), int(rng.integers(1, 100))
        seq, _ = sc.make_sequence(np.random.default_rng(sseed), T)
        for cls, tan in ((M.RNN, tangent_rnn), (M.LSTM, tangent_lstm),
                         (M.Attention, tangent_attention)):
            net = cls(H, seed=seed)
            ref = tan(net, seq)
            got = M.sensitivity_to_signal(net, T, seed=sseed)
            np.testing.assert_allclose(got, ref, rtol=1e-7, atol=0.0)


def test_rnn_sensitivity_at_large_T_is_tiny_but_not_zero():
    s = M.sensitivity_to_signal(M.RNN(8, seed=0), 1200)
    assert 1e-90 < s < 1e-70  # exakter Wert ~1,4e-83, keine Gleitkomma-Null


def test_sensitivity_matches_central_finite_difference_at_small_T():
    for cls in (M.RNN, M.LSTM, M.Attention):
        for T in (2, 5, 10):
            net = cls(8, seed=0)
            seq, _ = sc.make_sequence(np.random.default_rng(1), T)
            e = 1e-6
            a, b = seq.copy(), seq.copy()
            a[0] += e
            b[0] -= e
            fd = abs((net.forward(a)[0] - net.forward(b)[0]) / (2 * e))
            np.testing.assert_allclose(M.sensitivity_to_signal(net, T, seed=1), fd, rtol=1e-5)


def _att_loss_complex(p, seq, y, d):
    T = len(seq)
    E = np.array([p["We"][:, 0] * x + p["be"] for x in seq]) \
        + M.sinusoidal_positional_encoding(T, d)
    z = (E @ p["Wk"].T) @ p["q"] / np.sqrt(d)
    e = np.exp(z - np.max(z.real))
    a = e / np.sum(e)
    s = p["Why"] @ (a @ (E @ p["Wv"].T)) + p["by"]
    return -y * s


def test_attention_backward_matches_complex_step_gradient():
    rng = np.random.default_rng(4)
    for trial in range(40):
        d, T = int(rng.integers(2, 7)), int(rng.integers(1, 9))
        att = M.Attention(d, seed=trial)
        for k in att.params:
            att.params[k] = rng.normal(0, 0.8, size=att.params[k].shape)
        seq = rng.normal(0, 1, size=T)
        s, cache = att.forward(seq)
        y = -1.0 if s >= 0 else 1.0
        grads, _ = att.backward(y, s, cache)
        for key, g in grads.items():
            ref = np.zeros(np.shape(g))
            for idx in np.ndindex(*ref.shape):
                q = {k: np.asarray(v, dtype=complex).copy() for k, v in att.params.items()}
                q[key][idx] += 1e-30j
                ref[idx] = _att_loss_complex(q, seq, y, d).imag / 1e-30
            np.testing.assert_allclose(g, ref, atol=1e-9)
