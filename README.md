# Attention/Transformer – Zugriff statt Weitergabe – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-attention-transformer-demo.streamlit.app/)**

Stück 6 (**letztes Stück**) der **Neuronale-Netze-Reihe** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Stück 4 und 5 zeigten: RNN und LSTM müssen Information Schritt für Schritt durch die Zeit
**weiterreichen** – das kostet Gradientenkraft (RNN: exponentiell, LSTM: langsamer, aber immer
noch). **Aufmerksamkeit** (Vaswani et al. 2017) berechnet die Ausgabe stattdessen als gewichtete
Summe über ALLE Zeitschritte gleichzeitig – **direkter Zugriff statt Weitergabe.** Diese Demo
zeigt sowohl den strukturellen Vorteil als auch eine echte, ehrliche Einschränkung: der Vorteil
ist real, aber er macht das Training bei großem $T$ nicht automatisch zuverlässig.

**Einordnung in die Reihe (VOLLSTÄNDIG):**

```
Perceptron (WURZEL)                              [gebaut]
 └─ MLP + Backpropagation                        [gebaut]
      ├─ CNN                                     [gebaut]
      └─ RNN                                     [gebaut]
           └─ LSTM                               [gebaut]
                └─ Attention/Transformer         [DIESES STÜCK, LETZTES]
```

**Ergebnis in Kürze:** Bei einer festen, UNTRAINIERTEN Zufallsinitialisierung gemessen, wie stark
sich die Ausgabe ändert, wenn sich der Eingabewert an der Signalposition winzig ändert
($|\partial s/\partial x_0|$): das RNN fällt exponentiell auf exakt $0{,}0$ (Gleitkomma-Unterlauf,
spätestens bei $T=300$), das LSTM fällt langsamer, aber ebenfalls geometrisch (bei $T=1200$ nur
noch $5{,}2\cdot10^{-14}$), Attention fällt nur **mild** (durch die Softmax-Mittelung über mehr
Positionen verdünnt) und bleibt bei $T=1200$ eine normale, benutzbare Zahl ($\approx 10^{-3}$).
**Aber:** diese viel mildere Verdünnung übersetzt sich NICHT automatisch in zuverlässiges
Kaltstart-Training bei jedem $T$ – bei diesem einfachen Aufbau (ein Kopf, keine Stapelverarbeitung)
sinkt die Erfolgsquote mit wachsendem $T$ trotzdem (100 %/83 %/17 %/0 % bei $T=10/40/80/150$), nur
aus einem anderen Grund als beim RNN.

## Warum dieses Problem

Stück 5 endete mit der Lehre: ein nicht-verschwindender Gradient ist notwendig, aber nicht
hinreichend für zuverlässiges Training. Attention geht das strukturelle Problem an der Wurzel an
– keine Rekursion, kein Weiterreichen, jeder Ausgabeschritt hat direkten $O(1)$-Zugriff auf jeden
Eingabeschritt. Diese Demo prüft, ob das die in Stück 4/5 gefundenen Probleme wirklich beseitigt.

## Vorab-Hypothesen (vor der Messung notiert, hier geprüft)

| Hypothese | Ergebnis |
|---|---|
| Sensitivität auf die Signalposition bleibt über einen viel größeren $T$-Bereich brauchbar als bei RNN/LSTM | ✅ bei $T=1200$: RNN exakt $0{,}0$, LSTM $5{,}2\cdot10^{-14}$, Attention $1{,}0\cdot10^{-3}$ |
| Gradienten-Check (Rückwärtspass durch Softmax-Aufmerksamkeit) unter $10^{-6}$ | ✅ $1{,}6\cdot10^{-8}$ |
| Entartete (quasi-One-Hot) Aufmerksamkeit reduziert sich exakt auf $\text{out}=v_j$ | ✅ identische Ausgabe ($0{,}186265=0{,}186265$) |
| ⚠️ **Ehrlich geprüft, nicht einfach angenommen:** die viel mildere Verdünnung reicht automatisch für hohe Kaltstart-Erfolgsquote bei jedem $T$ | ❌ **Reicht nicht automatisch** – Erfolgsquote sinkt trotzdem mit $T$ (100 % → 0 % zwischen $T=10$ und $T=150$), nur langsamer und aus einem anderen Grund als beim RNN (siehe unten) |

## Befunde (gemessen, keine Behauptungen)

**Empfindlichkeit $|\partial s/\partial x_0|$** (bei EINER festen, untrainierten
Zufallsinitialisierung, keine Trainingsvarianz, daher exakte Werte statt Toleranzband):

| T | RNN | LSTM | Attention |
|---|---|---|---|
| 2 | 0,311 | 0,00081 | 0,432 |
| 10 | 0,00327 | 0,0726 | 0,106 |
| 40 | 7,4·10⁻⁵ | 0,00401 | 0,0387 |
| 80 | 2,50·10⁻⁷ | 0,00341 | 0,0197 |
| 150 | 9,44·10⁻¹² | 0,00423 | 0,0116 |
| 300 | 0,0 | 5,96·10⁻⁵ | 0,00588 |
| 600 | 0,0 | 2,12·10⁻¹¹ | 0,00234 |
| 1200 | 0,0 | 5,20·10⁻¹⁴ | 0,00102 |

Bei sehr kleinem $T$ ($\le10$) ist die Reihenfolge noch nicht sauber getrennt (Zufallsinit-Rauschen
dominiert) – ab $T=40$ gilt durchgehend RNN ≤ LSTM ≤ Attention, mit wachsendem Abstand.

**Kaltstart-Erfolgsquote** (6 Zufalls-Initialisierungen je $T$, SGD, kein Curriculum, Attention
allein – gleiche Kaltstart-Bedingungen wie RNN/LSTM in Stück 4/5):

| T | Erfolgsquote |
|---|---|
| 10 | 100 % |
| 40 | 83 % |
| 80 | 17 % |
| 150 | 0 % |

## Modell und Verfahren

- `att_scenario.py` – Signal-in-Rauschen-Sequenzgenerator (eigenständige Kopie aus lstm-demo).
- `att_model.py` – `Attention` (Einbettung + feste Sinus-Positionskodierung + gelernte
  Anfrage/Schlüssel/Wert + Softmax-Aufmerksamkeit + Rückwärtspass von Hand), `RNN`- und
  `LSTM`-Referenzen (eigenständige Kopien für den direkten Sensitivitäts-Vergleich), `SGD`,
  `sensitivity_to_signal()` (trainingsfreie Empfindlichkeitsmessung).
- `att_evaluation.py` – Sensitivitäts-Sweep (schnell, kein Training), Kaltstart-Erfolgsquote-Sweep
  (teuer), Gradienten-Check, Korrektheits-Kette (entartete Aufmerksamkeit).
- `att_visualization.py` – Plotly: Sequenz-Anzeige, Aufmerksamkeitsgewichte, Sensitivität-vs-T
  (log-log, RNN/LSTM/Attention), Erfolgsquote-vs-T.

## Was die App zeigt

Sequenzlänge, Einbettungsdimension, Trainingsgröße, Rauschen, Lernrate, Epochen und Seed in der
Sidebar; eine Beispielsequenz mit den gelernten Aufmerksamkeitsgewichten daneben; Testgenauigkeit
für die aktuelle Konfiguration; der Sensitivitäts-Sweep (RNN/LSTM/Attention, schnell, kein
Spinner nötig) als zentraler Befund; der Kaltstart-Erfolgsquote-Sweep (mit Spinner und
Zeitangabe, ca. 15 Sekunden); ein "📐"-Abschnitt mit Korrektheits-Kette, Gradienten-Check,
Literatur und einem SOTA-Ausblick (Mamba/State-Space-Modelle).

## Was nicht funktioniert hat / Grenzen

**Zwei echte Fehlschläge auf dem Weg zur funktionierenden Aufmerksamkeit:** Die erste Version
leitete die Anfrage aus dem letzten (irrelevanten, verrauschten) Zeitschritt ab – das Training
scheiterte bei jedem $T$ vollständig (0 %), weil die Anfrage selbst kein brauchbares Signal trug.
Behoben durch eine dedizierte, gelernte, **inhaltsunabhängige** Anfrage (ein "Klassifikations-Token"
-Muster). Selbst danach scheiterte großes $T$ weiterhin, weil reine inhaltsbasierte Aufmerksamkeit
bei sehr großem $T$ statistisch mit Extremwert-Rauschen verwechselt wird (das Maximum vieler
Rauschziehungen kann die echte Signalgröße erreichen/übertreffen) – behoben durch eine feste
Sinus-Positionskodierung.

**Bewusste Plan-Anpassung:** die ursprünglich geplante Kaltstart-T-Sweep-Messung wäre – wie in
Stück 4/5 gelernt – über viele Trainingsepochen chaotisch plattform-/seed-abhängig gewesen. Der
**Hauptbefund** der Demo wurde deshalb bewusst auf eine **trainingsfreie, deterministische**
Sensitivitätsmessung bei fester Zufallsinitialisierung verschoben (kein Training, daher exakte,
plattformrobuste Werte). Der Kaltstart-Erfolgsquote-Sweep bleibt als ehrlicher Nebenbefund
erhalten (mit Toleranzband getestet), zeigt aber, dass die mildere Verdünnung allein nicht
automatisch zu einer flachen Erfolgsquote-Kurve führt.

**Grenzen:** ein Kopf, eine Schicht (echte Transformer stapeln viele Köpfe/Schichten). Nur
Positions-, keine relative Ordnungskodierung. Kein Multi-Head, keine Lernraten-Zeitpläne, kein
Layer-Norm/Residual – bewusst außerhalb dieses Rahmens, um jeden Mechanismus einzeln
nachvollziehbar zu halten.

## Tests

28 Tests, `python -m pytest tests/ -v` (Laufzeit lokal ca. 27 Sekunden – deutlich schneller als
rnn-demo/lstm-demo, da der Hauptbefund trainingsfrei ist):
- `test_scenario.py` – Reproduzierbarkeit, Signalposition, Klassenbalance.
- `test_model.py` – Forward/Backward, Gradienten-Check, Positionskodierung, Sensitivitäts-Ordnung,
  Korrektheits-Kette, Trainings-Rauchtest.
- `test_evaluation.py` – Sweep-Funktionen mit billigen Parametern (Korrektheit, nicht die
  offiziellen Zahlen).
- `test_claims.py` – jede Zahl oben nachgerechnet, mit Toleranzband (Modul-Fixtures berechnen
  jeden Sweep nur einmal).
- `test_presets.py`, `test_app.py` – Presets, Regler-Extremwerte, Footer.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `att_constants.py` | Regler-Grenzen, Presets |
| `att_scenario.py` | Sequenzgenerator |
| `att_model.py` | Attention, RNN-/LSTM-Referenzen, SGD, Sensitivitätsmessung |
| `att_evaluation.py` | Sweeps, Gradienten-Check, Reduktions-Check |
| `att_visualization.py` | Plotly-Plots |
| `att_presets.py` | Permalink-Sync, Presets |
| `tests/` | pytest-Suite |

## Bewusst nicht umgesetzt

Kein Multi-Head-/Mehrschicht-Transformer (ein Kopf, eine Schicht genügt, um den Kernmechanismus
zu zeigen). Keine relative Positionskodierung. Kein Curriculum-Learning-Vergleich (anders als in
Stück 5 – hier ist der Hauptbefund bewusst trainingsfrei, ein Curriculum-Vergleich würde vom
eigentlichen Punkt ablenken).

## Fazit der ganzen Reihe (6 Stücke)

Diese Demo schließt die Neuronale-Netze-Reihe ab. Der rote Faden: **Perceptron** (linear
trennbar, ein Update), **MLP+Backpropagation** (nichtlineare Trennung durch verdeckte Schichten,
Gradienten von Hand), **CNN** (geteilte Gewichte nutzen räumliche Struktur), **RNN**
(Rekursion für Sequenzen – aber der Gradient verschwindet mit $T$), **LSTM** (ein Zellzustand
rettet den Gradienten strukturell, aber Training bleibt bei großem $T$ ohne Curriculum
unzuverlässig), **Attention/Transformer** (kein Weiterreichen mehr nötig – der mildeste,
robusteste strukturelle Befund der ganzen Reihe, aber auch hier: strukturelles Potenzial allein
garantiert kein einfaches Training). Jedes Stück maß seinen eigenen Kernmechanismus ehrlich,
inklusive mehrerer widerlegter Ausgangshypothesen (Stück 5: LSTM ist beim Kaltstart NICHT
zuverlässiger als RNN; Stück 6: mildere Verdünnung löst das Trainingsproblem NICHT automatisch) –
konsistent mit dem Grundsatz der ganzen Portfolio-Reihe: messen statt annehmen.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
streamlit run app.py
```

## Literatur

- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017 (arXiv:1706.03762).
- Bahdanau, D., Cho, K. & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to
  Align and Translate.* arXiv:1409.0473 (früher Aufmerksamkeits-Mechanismus vor Vaswani et al.).
- Gu, A. & Dao, T. (2023/2024). *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces.* (State-Space-Modelle als linearer Ausblick zum quadratischen Attention-Aufwand –
  bewusst nicht gebaut, siehe SOTA-Hinweis in der App.)

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Neuronale Netze: vom Perceptron zum Transformer](https://sebastianhanisch.net/konzepte-neuronale-netze.html).
