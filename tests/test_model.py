import numpy as np

import att_evaluation as ev
import att_model as m
import att_scenario as sc


def test_attention_forward_returns_scalar_and_valid_distribution():
    att = m.Attention(4, seed=0)
    seq = np.zeros(10)
    s, cache = att.forward(seq)
    assert isinstance(s, float)
    assert np.isclose(cache["alpha"].sum(), 1.0)
    assert np.all(cache["alpha"] >= 0)


def test_backward_returns_none_when_correctly_classified():
    att = m.Attention(4, seed=0)
    att.params["Why"][:] = 0.0
    att.params["by"] = np.asarray(5.0)
    seq = np.zeros(5)
    s, cache = att.forward(seq)
    grads, extra = att.backward(1.0, s, cache)
    assert extra is None
    assert np.allclose(grads["Wk"], 0.0)


def test_gradient_check_below_1e_minus_6():
    assert ev.gradient_check() < 1e-6


def test_sinusoidal_positional_encoding_is_deterministic_and_bounded():
    pe = m.sinusoidal_positional_encoding(20, 8)
    assert pe.shape == (20, 8)
    assert np.all(np.abs(pe) <= 1.0)
    pe2 = m.sinusoidal_positional_encoding(20, 8)
    np.testing.assert_array_equal(pe, pe2)


def test_reduction_check_degenerate_attention_is_exact():
    out = ev.reduction_check_degenerate_attention()
    assert out["identical"]
    assert out["alpha_max"] > 0.999


def test_sensitivity_rnn_underflows_far_earlier_than_attention():
    """Kernbefund: das RNN unterlaeuft float64 (exakt 0.0) lange bevor
    Attention auch nur annaehernd verschwindet."""
    s_rnn = m.sensitivity_to_signal(m.RNN(8, seed=0), T=300)
    s_att = m.sensitivity_to_signal(m.Attention(8, seed=0), T=300)
    assert s_rnn == 0.0
    assert s_att > 1e-6


def test_sensitivity_lstm_decays_slower_than_rnn_but_faster_than_attention():
    T = 600
    s_rnn = m.sensitivity_to_signal(m.RNN(8, seed=0), T)
    s_lstm = m.sensitivity_to_signal(m.LSTM(8, seed=0), T)
    s_att = m.sensitivity_to_signal(m.Attention(8, seed=0), T)
    assert s_rnn < s_lstm < s_att


def test_training_reduces_errors_at_short_T():
    ds = sc.make_dataset(30, T=10, seed=1)
    att = m.Attention(8, seed=0)
    result = m.train(att, m.SGD(0.3), ds.X, ds.y, epochs=30)
    assert result.errors_per_epoch[-1] <= result.errors_per_epoch[0]
