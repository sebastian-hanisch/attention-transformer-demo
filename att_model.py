"""Einschichtige Selbstaufmerksamkeit (Vaswani et al. 2017, ein Kopf) mit
Rueckwaertspass von Hand, plus RNN- und LSTM-Referenzen (eigenstaendige
Kopien aus Stueck 4/5) fuer den direkten Vergleich.

Architektur: Eingabe-Einbettung + feste sinusfoermige Positionskodierung,
Schluessel/Werte an JEDER Position, EINE gelernte (nicht aus der Eingabe
abgeleitete) Anfrage - siehe README fuer die Begruendung (eine aus dem
letzten, irrelevanten Rausch-Zeitschritt abgeleitete Anfrage verhinderte in
der Vormessung jedes Lernen)."""
from dataclasses import dataclass, field

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _scalar(v):
    """float fuer reelle Eingaben (unveraendertes Verhalten); komplexe Werte bleiben komplex,
    damit die Empfindlichkeit per Complex-Step exakt berechnet werden kann."""
    return v if np.iscomplexobj(v) else float(v)


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def sinusoidal_positional_encoding(T: int, d: int) -> np.ndarray:
    """Feste (nicht gelernte) Positionskodierung (Vaswani et al. 2017),
    generalisiert auf jede Sequenzlaenge T ohne erneutes Lernen."""
    pe = np.zeros((T, d))
    positions = np.arange(T)[:, None]
    div = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
    pe[:, 0::2] = np.sin(positions * div)
    n_cos = d - len(div) if d % 2 else len(div)
    pe[:, 1::2] = np.cos(positions * div[:n_cos])
    return pe


class Attention:
    def __init__(self, d: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (1 + d))
        limit_d = np.sqrt(6.0 / (d + d))
        self.d = d
        self.params = {
            "We": rng.uniform(-limit, limit, size=(d, 1)),
            "be": np.zeros(d),
            "q": rng.uniform(-limit_d, limit_d, size=d),
            "Wk": rng.uniform(-limit_d, limit_d, size=(d, d)),
            "Wv": rng.uniform(-limit_d, limit_d, size=(d, d)),
            "Why": rng.uniform(-np.sqrt(6.0 / (d + 1)), np.sqrt(6.0 / (d + 1)), size=d),
            "by": np.zeros(()),
        }

    def forward(self, seq: np.ndarray):
        p = self.params
        T = len(seq)
        PE = sinusoidal_positional_encoding(T, self.d)
        E = np.array([p["We"][:, 0] * x + p["be"] for x in seq]) + PE
        K = E @ p["Wk"].T
        V = E @ p["Wv"].T
        scores = (K @ p["q"]) / np.sqrt(self.d)
        alpha = softmax(scores)
        out = alpha @ V
        s = _scalar(p["Why"] @ out + p["by"])
        cache = dict(seq=seq, E=E, K=K, V=V, alpha=alpha, out=out)
        return s, cache

    def backward(self, y: float, s: float, cache: dict):
        p = self.params
        seq, E, K, V, alpha, out = (cache["seq"], cache["E"], cache["K"], cache["V"],
                                     cache["alpha"], cache["out"])
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        if y * s > 0:
            return grads, None
        ds = -y
        grads["Why"] = ds * out
        grads["by"] = np.asarray(ds)
        dout = ds * p["Why"]

        dV = np.outer(alpha, dout)
        dalpha = V @ dout
        dscores = alpha * (dalpha - np.sum(alpha * dalpha))  # Softmax-Rueckwaertspass
        dK = np.outer(dscores, p["q"]) / np.sqrt(self.d)

        dE = dK @ p["Wk"] + dV @ p["Wv"]
        grads["Wk"] = dK.T @ E
        grads["Wv"] = dV.T @ E
        grads["q"] = (dscores @ K) / np.sqrt(self.d)
        grads["We"][:, 0] = (dE * np.asarray(seq)[:, None]).sum(axis=0)
        grads["be"] = dE.sum(axis=0)
        return grads, alpha

    def predict(self, seq: np.ndarray) -> float:
        s, _ = self.forward(seq)
        return 1.0 if s >= 0 else -1.0

    def n_params(self) -> int:
        return sum(v.size for v in self.params.values())


class RNN:
    """Referenz aus Stueck 4 (eigenstaendige Kopie)."""

    def __init__(self, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        H = n_hidden
        limit_xh = np.sqrt(6.0 / (1 + H))
        limit_hh = np.sqrt(6.0 / (H + H))
        limit_hy = np.sqrt(6.0 / (H + 1))
        self.H = H
        self.params = {
            "Wxh": rng.uniform(-limit_xh, limit_xh, size=(H, 1)),
            "Whh": rng.uniform(-limit_hh, limit_hh, size=(H, H)),
            "bh": np.zeros(H),
            "Why": rng.uniform(-limit_hy, limit_hy, size=H),
            "by": np.zeros(()),
        }

    def forward(self, seq: np.ndarray):
        T = len(seq)
        h = np.zeros(self.H)
        hs = [h]
        p = self.params
        for t in range(T):
            z = p["Wxh"][:, 0] * seq[t] + p["Whh"] @ h + p["bh"]
            h = np.tanh(z)
            hs.append(h)
        s = _scalar(p["Why"] @ hs[-1] + p["by"])
        return s, hs

    def backward(self, y: float, s: float, hs: list, seq: np.ndarray):
        T = len(seq)
        p = self.params
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        if y * s > 0:
            return grads, None
        ds = -y
        grads["Why"] = ds * hs[-1]
        grads["by"] = np.asarray(ds)
        dh_next = ds * p["Why"]
        for t in range(T, 0, -1):
            delta = dh_next * (1 - hs[t] ** 2)
            grads["Wxh"][:, 0] += delta * seq[t - 1]
            grads["Whh"] += np.outer(delta, hs[t - 1])
            grads["bh"] += delta
            dh_next = p["Whh"].T @ delta
        return grads, True

    def predict(self, seq: np.ndarray) -> float:
        s, _ = self.forward(seq)
        return 1.0 if s >= 0 else -1.0


class LSTM:
    """Referenz aus Stueck 5 (eigenstaendige Kopie)."""
    GATES = ("f", "i", "o", "g")

    def __init__(self, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        H = n_hidden
        lx = np.sqrt(6.0 / (1 + H))
        lh = np.sqrt(6.0 / (H + H))
        lhy = np.sqrt(6.0 / (H + 1))
        self.H = H
        self.params = {}
        for gate in self.GATES:
            self.params[f"W{gate}"] = rng.uniform(-lx, lx, size=(H, 1))
            self.params[f"U{gate}"] = rng.uniform(-lh, lh, size=(H, H))
            self.params[f"b{gate}"] = np.zeros(H)
        self.params["bf"] = np.ones(H)
        self.params["Why"] = rng.uniform(-lhy, lhy, size=H)
        self.params["by"] = np.zeros(())

    def forward(self, seq: np.ndarray):
        T = len(seq)
        p = self.params
        h = np.zeros(self.H)
        c = np.zeros(self.H)
        cache = {"h": [h], "c": [c], "f": [], "i": [], "o": [], "g": []}
        for t in range(T):
            x = seq[t]
            f = sigmoid(p["Wf"][:, 0] * x + p["Uf"] @ h + p["bf"])
            i = sigmoid(p["Wi"][:, 0] * x + p["Ui"] @ h + p["bi"])
            o = sigmoid(p["Wo"][:, 0] * x + p["Uo"] @ h + p["bo"])
            g = np.tanh(p["Wg"][:, 0] * x + p["Ug"] @ h + p["bg"])
            c = f * c + i * g
            h = o * np.tanh(c)
            cache["f"].append(f)
            cache["i"].append(i)
            cache["o"].append(o)
            cache["g"].append(g)
            cache["c"].append(c)
            cache["h"].append(h)
        s = _scalar(p["Why"] @ cache["h"][-1] + p["by"])
        return s, cache

    def backward(self, y: float, s: float, cache: dict, seq: np.ndarray):
        T = len(seq)
        p = self.params
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        if y * s > 0:
            return grads, None
        ds = -y
        grads["Why"] = ds * cache["h"][-1]
        grads["by"] = np.asarray(ds)
        dh_next = ds * p["Why"]
        dc_next = np.zeros(self.H)
        for t in range(T, 0, -1):
            x = seq[t - 1]
            h_prev = cache["h"][t - 1]
            c_prev = cache["c"][t - 1]
            f, i, o, g, c = (cache["f"][t - 1], cache["i"][t - 1], cache["o"][t - 1],
                             cache["g"][t - 1], cache["c"][t])
            tanh_c = np.tanh(c)
            dh = dh_next
            do = dh * tanh_c
            dc = dc_next + dh * o * (1 - tanh_c ** 2)
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_prev = dc * f
            dz_f = df * f * (1 - f)
            dz_i = di * i * (1 - i)
            dz_o = do * o * (1 - o)
            dz_g = dg * (1 - g ** 2)
            for gate, dz in zip(self.GATES, (dz_f, dz_i, dz_o, dz_g)):
                grads[f"W{gate}"][:, 0] += dz * x
                grads[f"U{gate}"] += np.outer(dz, h_prev)
                grads[f"b{gate}"] += dz
            dh_next = (p["Uf"].T @ dz_f + p["Ui"].T @ dz_i + p["Uo"].T @ dz_o + p["Ug"].T @ dz_g)
            dc_next = dc_prev
        return grads, True

    def predict(self, seq: np.ndarray) -> float:
        s, _ = self.forward(seq)
        return 1.0 if s >= 0 else -1.0


class SGD:
    def __init__(self, eta: float = 0.1):
        self.eta = eta

    def step(self, params: dict, grads: dict) -> None:
        for key, g in grads.items():
            params[key] -= self.eta * g


@dataclass
class TrainResult:
    errors_per_epoch: list = field(default_factory=list)


def train_epoch(model, optimizer, X: np.ndarray, y: np.ndarray) -> int:
    n_errors = 0
    for seq, label in zip(X, y):
        if isinstance(model, Attention):
            s, cache = model.forward(seq)
            grads, extra = model.backward(label, s, cache)
        elif isinstance(model, LSTM):
            s, cache = model.forward(seq)
            grads, extra = model.backward(label, s, cache, seq)
        else:  # RNN
            s, cache = model.forward(seq)
            grads, extra = model.backward(label, s, cache, seq)
        if extra is not None:
            n_errors += 1
            optimizer.step(model.params, grads)
    return n_errors


def train(model, optimizer, X: np.ndarray, y: np.ndarray, epochs: int) -> TrainResult:
    errors_per_epoch = []
    for _ in range(epochs):
        errors_per_epoch.append(train_epoch(model, optimizer, X, y))
    return TrainResult(errors_per_epoch)


def accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    correct = sum(model.predict(seq) == label for seq, label in zip(X, y))
    return correct / len(y)


def sensitivity_to_signal(model, T: int, seed: int = 1, eps: float = 1e-10) -> float:
    """|d(score)/d(x_0)| EXAKT per Complex-Step-Ableitung (Im[s(x_0 + i*eps)]/eps), an EINER
    festen (nicht trainierten) Zufallsinitialisierung - schnell, deterministisch, kein
    Training noetig. Direktes Mass dafuer, ob Information von Position 0 ueberhaupt am
    Ausgang ankommt.

    Keine finite Differenz: deren Rundungsrauschen (~1e-13 bei eps=1e-4) verschluckt jede
    Empfindlichkeit unterhalb davon und meldet dann 0.0, obwohl der wahre Wert (z. B. RNN
    bei T=1200 etwa 1e-83) in float64 durchaus darstellbar ist. Complex-Step hat keine
    Subtraktion und bleibt exakt, bis eps*Wert in float64 unterlaeuft (Wert < ~1e-300);
    der Abbruchfehler ist ~eps^2 (relativ ~1e-20)."""
    rng = np.random.default_rng(seed)
    import att_scenario as sc
    seq, _ = sc.make_sequence(rng, T)
    seq_c = seq.astype(complex)
    seq_c[0] += 1j * eps
    s, _ = model.forward(seq_c)
    return abs(float(np.imag(s)) / eps)
