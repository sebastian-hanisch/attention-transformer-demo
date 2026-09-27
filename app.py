"""Attention/Transformer — Zugriff statt Weitergabe

Sebastian Hanisch - Operations Research und Machine Learning

Stück 6 (LETZTES) der "Neuronale Netze"-Reihe der "Konzepte"-Reihe:
Perceptron -> MLP+Backpropagation -> {CNN, RNN -> LSTM -> Attention/Transformer}.
RNN und LSTM müssen Information schrittweise durch die Zeit WEITERREICHEN.
Aufmerksamkeit (Vaswani et al. 2017) gibt jedem Ausgabeschritt direkten
Zugriff auf JEDEN Eingabeschritt, ohne Rekursion - die "wie weit muss
Information wandern"-Frage verschwindet strukturell.

Lauffähig mit: streamlit run app.py
"""
import numpy as np
import streamlit as st

import att_constants as C
import att_evaluation as ev
import att_model as m
import att_presets as pr
import att_scenario as sc
import att_visualization as viz

st.set_page_config(page_title="Attention/Transformer", layout="wide")


@st.cache_data(show_spinner=False)
def _analyse(T, hidden, n_train, n_test, noise, eta, epochs, seed):
    settings = ev.Settings(T=T, hidden=hidden, n_train=n_train, n_test=n_test, noise=noise,
                           eta=eta, epochs=epochs, seed=seed)
    out = ev.analyse(settings)
    s, cache = out["model"].forward(out["test_ds"].X[0])
    return {"test_accuracy": out["test_accuracy"],
            "errors_per_epoch": out["result"].errors_per_epoch,
            "alpha": cache["alpha"], "example_seq": out["test_ds"].X[0],
            "example_label": out["test_ds"].y[0]}


@st.cache_data(show_spinner=False)
def _sensitivity_sweep():
    return ev.sensitivity_sweep()


@st.cache_data(show_spinner=False)
def _t_sweep():
    return ev.t_sweep()


@st.cache_data(show_spinner=False)
def _gradient_check():
    return ev.gradient_check()


@st.cache_data(show_spinner=False)
def _reduction_check():
    return ev.reduction_check_degenerate_attention()


st.title("🧠 Attention/Transformer — Zugriff statt Weitergabe")
st.markdown(
    "RNN (Stück 4) und LSTM (Stück 5) müssen Information über Zwischenzustände Schritt für "
    "Schritt durch die Zeit **weiterreichen** - das kostet Gradientenkraft (RNN: exponentiell, "
    "LSTM: langsamer, aber immer noch). **Aufmerksamkeit** (Vaswani et al. 2017) berechnet die "
    "Ausgabe stattdessen als gewichtete Summe über ALLE Zeitschritte gleichzeitig - **direkter "
    "Zugriff statt Weitergabe.** Das ist das letzte Stück der Neuronale-Netze-Reihe."
)
st.caption(
    "Stück 6 (LETZTES, Kind von LSTM) der 'Neuronale Netze'-Reihe. Die Linie ist damit "
    "vollständig: Perceptron → MLP+Backpropagation → {CNN, RNN → LSTM → Attention/Transformer}."
)

with st.expander("So funktioniert Aufmerksamkeit", expanded=True):
    st.markdown(
        "1. Jede Position bekommt einen Schlüssel $k_t$ und einen Wert $v_t$ (aus der "
        "Einbettung + einer festen Positionskodierung).\n"
        "2. Eine gelernte Anfrage $q$ vergleicht sich mit JEDEM Schlüssel: "
        "$\\alpha_t=\\text{softmax}_t(q\\cdot k_t/\\sqrt{d})$.\n"
        "3. Ausgabe: $\\text{out}=\\sum_t \\alpha_t v_t$ - eine gewichtete Summe über ALLE "
        "Positionen in EINEM Schritt, keine Rekursion.\n"
        "4. Das ist der Unterschied zu RNN/LSTM: der Rechenweg von jeder Position zur Ausgabe "
        "ist $O(1)$ lang, nicht $O(T)$."
    )

st.caption("🎯 Schnellstart – ein Klick lädt ein durchgerechnetes Beispiel:")
preset_cols = st.columns(len(C.PRESETS))
for col, (key, preset) in zip(preset_cols, C.PRESETS.items()):
    with col:
        st.button(preset["label"], help=preset["help"], on_click=pr.apply_preset, args=(key,),
                   use_container_width=True)

st.caption("🔗 Die Adresszeile speichert deine Einstellungen als Permalink.")

pr.load_permalink_settings()
pr.init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    T = st.slider("Sequenzlänge T", C.T_MIN, C.T_MAX, ss["T"], key="widget_T",
                 on_change=pr.store_from_widget, args=("T",))
    ss["T"] = T
    hidden = st.slider("Einbettungsdimension", C.HIDDEN_MIN, C.HIDDEN_MAX, ss["hidden"],
                       key="widget_hidden", on_change=pr.store_from_widget, args=("hidden",))
    ss["hidden"] = hidden
    n_train = st.slider("Trainingsbeispiele", C.N_TRAIN_MIN, C.N_TRAIN_MAX, ss["n_train"],
                        step=10, key="widget_n_train", on_change=pr.store_from_widget,
                        args=("n_train",))
    ss["n_train"] = n_train
    noise = st.slider("Rauschanteil", C.NOISE_MIN, C.NOISE_MAX, ss["noise"], step=0.05,
                      key="widget_noise", on_change=pr.store_from_widget, args=("noise",))
    ss["noise"] = noise
    eta = st.slider("Lernrate η", C.ETA_MIN, C.ETA_MAX, ss["eta"], step=0.02,
                    key="widget_eta", on_change=pr.store_from_widget, args=("eta",))
    ss["eta"] = eta
    epochs = st.slider("Epochen", C.EPOCHS_MIN, C.EPOCHS_MAX, ss["epochs"], step=10,
                       key="widget_epochs", on_change=pr.store_from_widget, args=("epochs",))
    ss["epochs"] = epochs
    seed = st.number_input("Seed", value=ss["seed"], step=1, key="widget_seed",
                           on_change=pr.store_from_widget, args=("seed",))
    ss["seed"] = seed
    st.button("🎲 Zufälliger Seed", on_click=pr.randomize_seed)
    n_test = C.N_TEST_DEFAULT

pr.sync_query_params(dict(T=T, hidden=hidden, n_train=n_train, n_test=n_test, noise=noise,
                         eta=eta, epochs=epochs, seed=seed))

with st.spinner("Trainiere Attention..."):
    out = _analyse(T, hidden, n_train, n_test, noise, eta, epochs, int(seed))

st.markdown("---")
col_seq, col_alpha = st.columns(2)
with col_seq:
    st.plotly_chart(viz.build_sequence_figure(out["example_seq"], out["example_label"]),
                    key=f"seq_{T}_{noise}_{seed}", use_container_width=True)
with col_alpha:
    st.plotly_chart(viz.build_attention_weights_figure(out["alpha"]),
                    key=f"alpha_{T}_{hidden}_{n_train}_{noise}_{eta}_{epochs}_{seed}",
                    use_container_width=True)
    st.caption(
        "Je höher der Balken an Position 0 (gestrichelte Linie), desto stärker "
        "konzentriert sich die gelernte Aufmerksamkeit auf die Signalposition."
    )

st.subheader("🎯 Was am Ende steht")
m1, m2 = st.columns(2)
m1.metric("Testgenauigkeit", f"{out['test_accuracy']*100:.0f}%")
m2.metric("Sequenzlänge T", f"{T}")

st.markdown("---")
st.subheader("🎯 Die zentrale Messung: Empfindlichkeit auf die Signalposition")
sens_rows = _sensitivity_sweep()
st.plotly_chart(viz.build_sensitivity_figure(sens_rows), key="sensitivity_chart",
                use_container_width=True)
st.caption(
    "Bei EINER festen Zufallsinitialisierung (kein Training nötig - schnell, deterministisch, "
    "plattformrobust) gemessen: wie stark ändert sich die Ausgabe, wenn sich der Wert an "
    "Position 0 winzig ändert? Beim RNN fällt das exponentiell auf exakt 0.0 (Gleitkomma-"
    "Unterlauf) - beim LSTM viel langsamer, aber auch geometrisch fallend. Bei Attention fällt "
    "es nur MILDE (durch die Softmax-Mittelung über mehr Positionen verdünnt) und bleibt auch "
    "bei T=1200 eine normale, benutzbare Zahl."
)

st.markdown("---")
st.subheader("🎯 Kaltstart-Erfolgsquote (ehrlich gemessen)")
with st.spinner("Berechne Erfolgsquote über 6 Zufalls-Initialisierungen je Sequenzlänge "
               "(einmalig, ~15 Sekunden)..."):
    sweep = _t_sweep()
st.plotly_chart(viz.build_t_sweep_figure(sweep), key="t_sweep_chart", use_container_width=True)
st.caption(
    "Ehrlicher Befund: die viel mildere Verdünnung übersetzt sich NICHT automatisch in "
    "zuverlässiges Training bei jedem T - bei diesem einfachen Aufbau (ein Kopf, keine "
    "Stapelverarbeitung) sinkt die Erfolgsquote mit wachsendem T trotzdem, nur langsamer und "
    "aus einem anderen Grund als beim RNN (hier: mehr konkurrierende Positionen erschweren das "
    "Schärfen der Aufmerksamkeit, keine exponentielle Gradientenkatastrophe)."
)

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    "| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |\n"
    "|---|---|---|\n"
    "| Milde Verdünnung reicht für Trainierbarkeit bei jedem T | Reicht nicht automatisch - "
    "die Erfolgsquote sinkt trotzdem mit T (siehe oben) | Mehrere Köpfe, Stapelverarbeitung, "
    "Lernraten-Zeitpläne (außerhalb dieses Rahmens) |\n"
    "| Nur EIN Kopf/EINE Schicht | Echte Transformer stapeln viele Köpfe/Schichten | — |\n"
    "| Position zählt, nicht Reihenfolge relativ zueinander | Aufgaben mit relativer Ordnung "
    "bräuchten andere Kodierungen | — |\n"
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Aufmerksamkeit:** $\alpha_t=\text{softmax}_t(q\cdot k_t/\sqrt{d})$, $\text{out}=\sum_t \alpha_t v_t$.

**Korrektheits-Kette (entartete Aufmerksamkeit):** wird $\alpha$ auf eine Position $j$
konzentriert ($\alpha_j\approx1$, alle anderen $\approx0$), muss $\text{out}\approx v_j$ gelten
- von Hand nachrechenbar.
"""
    )
    check = _reduction_check()
    c1, c2 = st.columns(2)
    c1.metric("Entartete Aufmerksamkeit", f"{check['s_degenerate']:.6f}")
    c2.metric("Manuell (nur v_j)", f"{check['s_manual']:.6f}")
    st.metric("Identisch?", "Ja" if check["identical"] else "Nein (Fehler!)")
    st.markdown("**Gradienten-Check** (gegen finite Differenzen):")
    st.metric("Maximaler relativer Fehler", f"{_gradient_check():.2e}")
    st.markdown(
        "**Literatur:** Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017 "
        "(arXiv:1706.03762)."
    )
    st.markdown(
        "**SOTA-Hinweis 2026:** Transformer bleibt weiterhin die dominante Architektur - "
        "praktisch jedes große Sprachmodell baut darauf auf. Mamba/State-Space-Modelle "
        "(Gu & Dao 2023/2024) bieten lineare statt quadratische Komplexität in der "
        "Sequenzlänge und sind ein zunehmend ernstgenommener Ausblick - bewusst nicht gebaut."
    )
    st.caption(
        "Implementiert in `att_model.py` (Attention, RNN-/LSTM-Referenzen), `att_scenario.py` "
        "(Sequenzen), `att_evaluation.py` (Sensitivitäts-/Erfolgsquote-Sweep, Gradienten-Check, "
        "Reduktions-Check), `att_visualization.py` (Plots)."
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) "
    "– Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung "
    "für Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
