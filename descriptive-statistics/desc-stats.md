---
title: "Deskriptive Statistik für Data Scientists"
layout: single
author_profile: true
author: Christina Kratsch
lecture_name: "Data Science"
lecture_desc: "Grundlegende Methoden für die Exploration und das Management von Daten."
licence: "CC-BY"
licence_desc: 2024 | HTW Berlin 
classes: wide
---


## Was ist Deskriptive Statistik?

Die deskriptive Statistik analysiert Daten, um Muster und Trends zu erkennen, ohne inferenzielle Schlüsse zu ziehen. Sie nutzt Kennzahlen wie Mittelwert, Varianz und grafische Darstellungen wie Histogramme, um Daten zu beschreiben und zu visualisieren. In diesem Kapitel werden wir die Grundlagen der deskriptiven Statistik einführen und ihre Bedeutung für die Datenanalyse herausstellen.

## Visualisierung mit Histogrammen

**Histogramme** sind eine wichtige grafische Darstellungsmethode in der deskriptiven Statistik, um die Lage und Verteilung von Daten zu visualisieren. Sie ermöglichen es, die Häufigkeitsverteilung von Daten in bestimmten Intervallen zu betrachten und Muster sowie Strukturen in den Daten zu identifizieren. Ein Histogramm besteht aus einer Reihe von rechteckigen Balken, wobei die Breite jedes Balkens einem Intervall von Datenwerten entspricht und die Höhe die Häufigkeit oder relative Häufigkeit dieser Werte in diesem Intervall repräsentiert.

Ein Histogramm ermöglicht es, die Form der Datenverteilung zu visualisieren. Je nach Form der Verteilung können Daten symmetrisch, schief oder bimodal sein. Aus der Form der Verteilung lassen sich dann, je nach Kontext, relevante Aussagen über fachliche Zusammenhänge machen: so können durch die Betrachtung von Histogrammen z.B. Ausreißer in den Daten identifiziert werden. Ausreißer sind Datenpunkte, die deutlich von der allgemeinen Verteilung abweichen. Auch kann man mit Histogrammen unter Umständen erkennen, ob sich Muster aus mehreren Verteilungen überlagern.

Histogramme helfen dadurch dabei, die geeigneten nächsten Schritte für die vorliegenden Daten zu wählen. Je nach Form der Verteilung können unterschiedliche statistische Tests und Modelle angewendet werden.

Folgender Python-Code kann für die Erstellung eines Histogramms verwendet werden:
```python
import matplotlib.pyplot as plt
import numpy as np

# Generierung von Beispiel-Daten
data = np.random.randn(1000)

# Erstellung des Histogramms
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Beschriftungen und Titel hinzufügen
plt.xlabel('Datenwerte')
plt.ylabel('Häufigkeit')
plt.title('Histogramm der Beispiel-Daten')

# Anzeigen des Histogramms
plt.show()
```

## Eigenschaften einer Stichprobe: Mittelwert, Varianz und Kurtosis

Die Eigenschaften einer Stichprobe liefern wichtige Informationen über die Verteilung und Charakteristik der Daten. Statistische Tests vergleichen diese Eigenschaften häufig zwischen verschieden Bedingungen oder Stichprobene.

### Mittelwert

Der Mittelwert einer Stichprobe ist die durchschnittliche Summe aller Beobachtungswerte. Mathematisch wird der Mittelwert $\bar{x}$ einer Stichprobe mit $n$ Beobachtungen wie folgt berechnet:

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

Der Mittelwert bietet eine zentrale Maßzahl, die die Tendenz der Daten zur Mitte hin beschreibt.

### Varianz

Die Varianz einer Stichprobe misst die durchschnittliche quadratische Abweichung der Beobachtungswerte vom Mittelwert. Sie wird durch folgende Formel berechnet:

$$
\text{Var}(X) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

Eine hohe Varianz deutet auf eine größere Streuung der Daten hin, während eine niedrige Varianz auf eine geringere Streuung hinweist.

### Kurtosis

Die Kurtosis einer Stichprobe beschreibt die Form der Verteilung und gibt Auskunft darüber, wie stark die Verteilung im Vergleich zur Normalverteilung "spitz" oder "flach" ist. Eine positive Kurtosis deutet auf eine relativ spitze Verteilung mit schweren Schwänzen hin, während eine negative Kurtosis auf eine flachere Verteilung mit leichteren Schwänzen hinweist.

Die Kurtosis wird durch die vierte zentrale Moment berechnet und kann mathematisch ausgedrückt werden als:

$$
\text{Kurt}(X) = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{\sigma} \right)^4
$$

wobei $\sigma$ die Standardabweichung ist.

### Python-Beispiel für Berechnungen von Mittelwert, Varianz und Kurtosis

```python
import numpy as np

# Beispiel-Daten
data = np.random.normal(loc=0, scale=1, size=100)

# Berechnung des Mittelwerts
mean = np.mean(data)

# Berechnung der Varianz
variance = np.var(data)

# Berechnung der Kurtosis
kurtosis = np.mean((data - mean) ** 4) / np.var(data)**2

print("Mittelwert:", mean)
print("Varianz:", variance)
print("Kurtosis:", kurtosis)
```

## Eigenschaften einer Stichprobe: Quantile und Perzentile

Quantile und Perzentile sind wichtige statistische Maße, die zur Beschreibung der Verteilung von Daten verwendet werden. Sie sind besonders wertvoll, um Vergleichbarkeit bei stark unterschiedlichen Stichproben oder Größenordnungen herzustellen.

**Quantile** sind Werte, die eine Stichprobe in gleich große Teile unterteilen. Das \(p\)-Quantil einer Stichprobe teilt die Daten so auf, dass ein Anteil von \(p\) der Daten kleiner oder gleich dem Quantilwert ist und ein Anteil von \(1-p\) der Daten größer oder gleich diesem Wert ist.

Mathematisch wird das \(p\)-Quantil \(Q_p\) wie folgt definiert:

$$
Q_p = \text{inf} \{x : F(x) \geq p\}
$$

wobei \(F(x)\) die Verteilungsfunktion der Stichprobe ist.

**Perzentile** sind spezielle Quantile, die in Prozenten angegeben werden. Das \(p\)-Perzentil entspricht dem \(p\)-Quantil, wobei \(p\) in Prozent ausgedrückt wird.

**Ein Beispiel:** Angenommen, wir haben eine Liste von Gehältern in einem Unternehmen:

```
{2000, 2500, 2700, 3000, 3200, 3500, 3800, 4000, 4200, 5000}
```

Um das 50. Perzentil zu berechnen, sortieren wir zunächst die Daten:

```
{2000, 2500, 2700, 3000, 3200, 3500, 3800, 4000, 4200, 5000}
```

Da die Anzahl der Datenpunkte (zehn) gerade ist, nehmen wir den Durchschnitt der beiden mittleren Werte, um den Median zu erhalten. In diesem Fall sind die mittleren Werte 3200 und 3500. Der Durchschnitt von 3200 und 3500 ist 3350. Daher beträgt das 50. Perzentil 3350. Das bedeutet, dass 50% der Gehälter in diesem Unternehmen 3350 oder weniger sind und 50% der Gehälter 3350 oder mehr betragen.

**Hinweis:** Konkret das 50. Perzentil nennt man auch **Median**.

Quantile und Perzentile finden breite Anwendung in verschiedenen Bereichen der Statistik und Datenanalyse. Sie werden beispielsweise verwendet, um die Verteilung von Einkommen in einer Bevölkerung zu analysieren, um Schwellenwerte in medizinischen Tests zu bestimmen oder um den Erfolg von Marketingkampagnen zu messen.

### Python-Beispiel für die Berechnung von Quantilen und Perzentilen

```python
import numpy as np

# Beispiel-Daten
data = np.random.normal(loc=0, scale=1, size=100)

# Berechnung des Median (50. Perzentil)
median = np.percentile(data, 50)

# Berechnung des 25. und 75. Perzentils (Interquartilsabstand)
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)

print("Median (50. Perzentil):", median)
print("25. Perzentil (Q1):", q1)
print("75. Perzentil (Q3):", q3)
```

## Datentransformation: Zentrierung einer Stichprobe

Die Zentrierung einer Stichprobe ist ein wichtiger Schritt in der Datenanalyse, um die Daten auf eine bestimmte Referenzebene zu zentrieren und somit bestimmte Effekte zu eliminieren oder zu minimieren, oder die Vergleichbarkeit zwischen Stichproben zu erhöhen. Im folgenden werden wir die Bedeutung der Zentrierung einer Stichprobe erläutern und verschiedene Methoden zur Durchführung dieser Datentransformation betrachten.

Die Zentrierung einer Stichprobe ist besonders wichtig, wenn absolute Werte der Beobachtungen weniger aussagekräftig sind als ihre Abweichungen von einem bestimmten Referenzpunkt. Durch die Zentrierung können wir den Mittelpunkt der Daten verschieben und somit bestimmte Effekte oder Trends genauer untersuchen.

### Mittelwertzentrierung

Bei der Mittelwertzentrierung wird von jedem Datenpunkt der Mittelwert der Stichprobe subtrahiert. Mathematisch ausgedrückt bedeutet dies, dass für jeden Datenpunkt $x_i$ und den Mittelwert $\bar{x}$ die transformierte Variable $x'_i$ durch die Formel

$$
x'_i = x_i - \bar{x}
$$

berechnet wird.

### Medianzentrierung

Ähnlich wie bei der Mittelwertzentrierung wird bei der Medianzentrierung der Median der Stichprobe als Referenzpunkt verwendet. Jeder Datenpunkt wird um den Median zentriert:

$$
x'_i = x_i - \text{median}
$$

Die Medianzentrierung ist robust gegenüber Ausreißern und kann in Situationen mit starken Ausreißern bevorzugt werden.

### Python-Beispiel für die Zentrierung einer Stichprobe

```python
import numpy as np

# Beispiel-Daten
data = np.random.normal(loc=10, scale=2, size=100)

# Mittelwert der Stichprobe berechnen
mean = np.mean(data)

# Median der Stichprobe berechnen
median = np.median(data)

# Mittelwertzentrierung
data_centered_mean = data - mean

# Medianzentrierung
data_centered_median = data - median
```

## Visualisierung mit Boxplots: Beispiele und Anwendungen

Boxplots sind ein nützliches Werkzeug in der deskriptiven Statistik, um die Verteilung von Daten grafisch darzustellen. In diesem Kapitel werden wir die Konzepte von Boxplots erläutern, ihre Bestandteile erklären und ihre Anwendungen in der Datenanalyse diskutieren.

Ein Boxplot, auch als Box-and-Whisker-Plot bekannt, visualisiert die Verteilung von Daten anhand verschiedener statistischer Kennzahlen. Die Hauptbestandteile eines Boxplots sind:

- Die Box, die den Interquartilsabstand (IQR) darstellt, also den Bereich zwischen dem 25. und dem 75. Perzentil.
- Die Linie in der Box, die den Median repräsentiert, also das 50. Perzentil.
- Die Whisker, die die Ausdehnung der Daten außerhalb des Interquartilsbereichs anzeigen.
- Mögliche Ausreißer, die als separate Punkte außerhalb der Whisker dargestellt werden.

Boxplots werden häufig verwendet, um die Verteilung von Daten zu vergleichen, Ausreißer zu identifizieren und Muster oder Trends in den Daten zu erkennen. Sie sind besonders nützlich, wenn man mit mehreren Gruppen oder Kategorien von Daten arbeitet und einen schnellen Überblick über die Verteilung und Streuung der Daten erhalten möchte.

### Python-Beispiel für die Erzeugung eines Boxplots

```python
import matplotlib.pyplot as plt
import numpy as np

# Beispiel-Daten für drei Gruppen
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(1, 1, 100)
data3 = np.random.normal(2, 1, 100)

# Erzeugung des Boxplots
plt.boxplot([data1, data2, data3], labels=['Gruppe 1', 'Gruppe 2', 'Gruppe 3'])
plt.title('Boxplot der Daten nach Gruppen')
plt.xlabel('Gruppen')
plt.ylabel('Werte')
plt.grid(True)
plt.show()
```

## Eigenschaften einer Stichprobe: Kovarianz

Die Kovarianz ist ein wichtiges Maß in der deskriptiven Statistik, das die lineare Beziehung zwischen zwei Variablen in einer Stichprobe misst. In diesem Kapitel werden wir die Bedeutung der Kovarianz erläutern, ihre Berechnungsmethode diskutieren und ihre Anwendungen in der Datenanalyse betrachten.

Die **Kovarianz** zwischen zwei Variablen \(X\) und \(Y\) wird durch die folgende Formel definiert:

$$
\text{cov}(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x}) (y_i - \bar{y})
$$

wobei \(x_i\) und \(y_i\) die einzelnen Beobachtungen von \(X\) bzw. \(Y\) sind, \(\bar{x}\) und \(\bar{y}\) die Mittelwerte von \(X\) bzw. \(Y\) sind, und \(n\) die Anzahl der Beobachtungen ist.

Die Kovarianz kann positive, negative oder null sein:
- Eine positive Kovarianz zeigt an, dass größere Werte von \(X\) tendenziell mit größeren Werten von \(Y\) einhergehen, und umgekehrt.
- Eine negative Kovarianz zeigt an, dass größere Werte von \(X\) tendenziell mit kleineren Werten von \(Y\) einhergehen, und umgekehrt.
- Eine Kovarianz von null deutet darauf hin, dass keine lineare Beziehung zwischen den beiden Variablen besteht.

Die Kovarianz wird häufig verwendet, um den Grad der Abhängigkeit zwischen zwei Variablen zu messen. Sie wird in der Finanzanalyse verwendet, um das Risiko und die Rendite von Anlagen zu bewerten, in der Biostatistik, um Beziehungen zwischen verschiedenen Merkmalen zu untersuchen, und in der Maschinenlernalgorithmen, um Merkmale zu selektieren oder Redundanzen zu identifizieren.

### Python-Beispiel für die Berechnung der Kovarianz

```python
import numpy as np

# Beispiel-Daten
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])

# Berechnung der Kovarianz
covariance = np.cov(x, y)[0, 1]

print("Kovarianz zwischen X und Y:", covariance)
```


## Datentransformation: Whitening

**Whitening** ist eine weitere wichtige Form der Datentransformation, die darauf abzielt, die Daten zu dekorrelieren und ihre Varianz zu normieren. Das Ziel ist es also, die Daten so zu transformieren, dass sie eine Einheitsvarianz haben ($\sigma_d^2 = 1$ für alle Dimensionen $d$) und in allen Dimensionen unkorreliert sind ($cov(d, e) = 0$ für alle Paare von Dimensionen $d, e$). Dies kann es erleichten, Muster in den Daten zu erkennen und verbessert oft die Leistung von Algorithmen.

Mathematisch wird der Whitening-Prozess in zwei Schritten durchgeführt:

1. **Dekorrelation**: Die Daten werden so transformiert (rotiert), dass die neue Kovarianzmatrix eine Diagonalmatrix ist, was bedeutet, dass die einzelnen Dimensionen unkorreliert sind.
2. **Skalierung**: Die Daten werden dann so skaliert, dass jede Dimension eine Einheitsvarianz hat.

Whitening findet breite Anwendung in verschiedenen Bereichen der Datenanalyse und des maschinellen Lernens. Es wird häufig verwendet, um Bilddaten vor der Verarbeitung durch Convolutional Neural Networks (CNNs) zu normalisieren, um Zeitreihendaten in der Finanzanalyse zu glätten und um Merkmale in hochdimensionalen Daten zu entdecken.

### Python-Beispiel für Whitening

```python
import numpy as np

# Beispiel-Daten
data = np.random.rand(100, 2)  # 100 Datenpunkte mit 2 Dimensionen

# Dekorrelation der Daten
data -= np.mean(data, axis=0)  # Mittelwertzentrierung
cov = np.cov(data, rowvar=False)
eig_vals, eig_vecs = np.linalg.eigh(cov)
whitened_data = np.dot(eig_vecs, np.dot(np.diag(1.0 / np.sqrt(eig_vals + 1e-5)), eig_vecs.T))

# Ausgabe der whitened Daten
print("Whitened Daten:")
print(whitened_data)
```