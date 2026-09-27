"""Reine Plotly-Figure-Builder, keine Streamlit-Aufrufe."""
import numpy as np
import plotly.graph_objects as go

import att_constants as C

COLOR_RNN = "#d62728"
COLOR_LSTM = "#ff7f0e"
COLOR_ATT = "#1f77b4"


def build_sequence_figure(seq: np.ndarray, label: float, title: str = ""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=seq, mode="lines+markers", name="Sequenz",
                             line=dict(color="#7f7f7f"), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=[C.SIGNAL_POS], y=[seq[C.SIGNAL_POS]], mode="markers",
                             name="Signal", marker=dict(size=14, symbol="star",
                             color=COLOR_ATT if label > 0 else COLOR_RNN)))
    fig.update_layout(
        title=title or f"Sequenz (Signal = {'+1' if label > 0 else '-1'} an Position {C.SIGNAL_POS})",
        xaxis=dict(title="Zeitschritt t", fixedrange=True),
        yaxis=dict(title="Wert", fixedrange=True), height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_attention_weights_figure(alpha: np.ndarray, title="Aufmerksamkeitsgewichte α"):
    fig = go.Figure(go.Bar(x=list(range(len(alpha))), y=alpha, marker_color=COLOR_ATT))
    fig.add_vline(x=C.SIGNAL_POS, line_dash="dot", line_color=COLOR_RNN,
                 annotation_text="Signalposition")
    fig.update_layout(
        title=title, xaxis=dict(title="Zeitschritt t", fixedrange=True),
        yaxis=dict(title="Gewicht α", fixedrange=True), height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_sensitivity_figure(rows, title="Empfindlichkeit ‖∂s/∂x₀‖ bei Zufallsinitialisierung (kein Training)"):
    Ts = [r["T"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Ts, y=[r["rnn"] for r in rows], mode="lines+markers", name="RNN",
                             line=dict(color=COLOR_RNN)))
    fig.add_trace(go.Scatter(x=Ts, y=[r["lstm"] for r in rows], mode="lines+markers", name="LSTM",
                             line=dict(color=COLOR_LSTM)))
    fig.add_trace(go.Scatter(x=Ts, y=[r["attention"] for r in rows], mode="lines+markers",
                             name="Attention", line=dict(color=COLOR_ATT)))
    fig.update_layout(
        title=title, xaxis=dict(title="Sequenzlänge T", fixedrange=True, type="log"),
        yaxis=dict(title="Empfindlichkeit (log-Skala)", fixedrange=True, type="log"),
        height=360, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_t_sweep_figure(rows, title="Kaltstart-Erfolgsquote vs. Sequenzlänge (Attention, SGD)"):
    Ts = [str(r["T"]) for r in rows]
    fig = go.Figure(go.Bar(x=Ts, y=[r["rate"] * 100 for r in rows], marker_color=COLOR_ATT))
    fig.update_layout(
        title=title, xaxis=dict(title="Sequenzlänge T", fixedrange=True),
        yaxis=dict(title="Erfolgsquote (%)", fixedrange=True, range=[0, 105]),
        height=320, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
