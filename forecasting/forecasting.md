---
title: "Forecasting: Analyse und Einfache Prognose"
layout: single
author_profile: true
author: Christina Kratsch
lecture_name: "Data Science"
lecture_desc: "Grundlegende Methoden für die Exploration und das Management von Daten."
licence: "CC-BY"
licence_desc: 2024 | HTW Berlin 
classes: wide
---


Was würden Sie tun, wenn Sie wüssten, wie die Zukunft aussehen würde? Für einige Leute könnte die Antwort lauten: „Bitcoin kaufen im Jahr 2017“. Oder für Einzelhändler könnte die Antwort lauten: „2019 Toilettenpapier oder Silikonchips auffüllen“. Was auch immer Ihre Antwort ist, wir sind uns einig, dass es von Vorteil wäre, Informationen über die Zukunft zu haben.
 
Obwohl die Menschheit schon seit Jahrtausenden Wahrsager danach befragt, was die Zukunft für uns bereithält, ist es immer noch eine Herausforderung, die Zukunft vorherzusagen. Selbst mit unserer heutigen Technologie könnten Vorhersagen über das Wetter in einer Stunde massiv falsch sein. Für einige Anwendungsfälle können wir das Verhalten einer Zeitreihe jedoch gut genug modellieren, um ziemlich genaue Vorhersagen über die Zukunft zu treffen.
 
**Zeitreihenanalyse und -prognose** sind zwei umfassende Themen, die überwältigend sein können. Einen ersten Teil zur Visualisierung und Bearbeitung von Zeitreihendaten haben wir bereits [an anderer Stelle](/modules/time-series-data/timeseries.md)  vorgestellt.  In diesem Modul stellen wir die wesentlichen Konzepte des **Forecastings** vor. Um einige Konzepte zu veranschaulichen, verwenden wir Codebeispiele in Python und Methoden aus der Bibliothek `statsmodels`.


## Zeitreihenprognose und Zeitreihenanalyse
In diesem Abschnitt werden die drei Schlüsselbegriffe definiert: Zeitreihen, Zeitreihenanalyse und Zeitreihenprognose.



![](https://site.wandb.ai/wp-content/uploads/2023/09/l31.png)

Eine **Zeitreihe** ist eine Folge zeitabhängiger Datenpunkte. Das bedeutet, dass jedem Datenpunkt ein Zeitstempel zugeordnet ist. Idealerweise werden diese Datenpunkte in konstanten Abständen (z. B. jeden Tag) und in chronologischer Reihenfolge (z. B. Montag, Dienstag, Mittwoch usw.) gemessen.

![](https://site.wandb.ai/wp-content/uploads/2023/09/l32.png)

Zeitreihen sind in der Regel numerische Werte, z. B. Verkaufsdaten, sie können aber auch kategorial sein, z. B. Ereignisdaten. Zeitreihendaten liegen normalerweise im Tabellenformat vor (z. B. CSV-Dateien) mit einer Spalte für die Zeitstempel und mindestens einer Spalte für die Zeitreihenwerte.

![](https://site.wandb.ai/wp-content/uploads/2023/09/l33.png)

Die **Zeitreihenanalyse** ähnelt der explorativen Datenanalyse (EDA) im Data Science-Workflow, da sie Beziehungen zwischen Variablen und Ausreißern analysiert. Darüber hinaus wollen wir aber
* langfristige Trends oder 
* sich kurzfristig wiederholende Muster
finden.
 
Ähnlich wie bei Standard-EDA-Techniken sind auch die Darstellung und Berechnung statistischer Merkmale wie Mittelwert und Standardabweichung für die Zeitreihenanalyse unerlässlich. Sie müssen jedoch zusätzliche Methoden verwenden, um die zusätzliche Dimension der Zeit zu erkunden.

![](https://site.wandb.ai/wp-content/uploads/2023/09/l34.png)

Bei der **Zeitreihenvorhersage** versuchen wir vorherzusagen, wie sich eine Folge von Beobachtungen in der Zukunft fortsetzen wird. Dazu analysieren wir zunächst historische Daten ähnlich der Zeitreihenanalyse. Anschließend passen wir ein Modell an die historischen Daten an, um unsere Vorhersagen zu treffen.
Für Zeitreihenprognosen gibt es viele Anwendungsfälle und Problemstellungen. Sie können sich unter anderem in folgenden Aspekten unterscheiden:
* Anzahl der vorherzusagenden beobachteten Zeitreihen (univariat vs. multivariat)
* Prognosezeitraum (kurzfristig vs. langfristig)
* Saisonalität
* Anzahl der relevanten Variablen
* Linearität der Abhängigkeiten.
* Einfluss (unbekannter?) externer Faktoren
* Verfügbarkeit historischer Daten
* Zusammenhang zwischen Zukunft und Historie


## Anwendungsfälle für Zeitreihenanalyse und -prognose

Zeitreihenanalyse und Zeitreihenprognose haben verschiedene Anwendungsfälle. Hier nur einige Beispiele:
* **Finanzen:**  Eines der beliebtesten Beispiele für Zeitreihenprognosen sind Aktienkursvorhersagen. Während wir alle gerne wissen würden, wie wir berechnen können, welche Aktie die nächste Google- oder Amazon-Aktie sein wird, ist dies auch einer der anspruchsvollsten Anwendungsfälle der Zeitreihenprognose.
* **Wetter:**  Einer der anderen allgemein bekannten Anwendungsfälle der Zeitreihenvorhersage ist die Wettervorhersage. Es ist auch ein hervorragendes Beispiel dafür, dass Vorhersagen mit einem kürzeren Vorhersagehorizont (z. B. das Wetter von morgen) normalerweise zuverlässiger sind als Vorhersagen mit einem längeren Vorhersagehorizont (z. B. das Wetter in einer Woche ab heute).
* **Nachfrage:**  So ziemlich alles mit Nachfrage ist ein guter Anwendungsfall für Zeitreihenanalyse und -prognose. Beispiele hierfür sind die Stromerzeugung, der Lagerbestand, die Personalbesetzung, die Anzahl der Passagiere in einem Bus oder Flugzeug oder der Verkehr auf einer Autobahn.

## Mathematische Grundlagen der Zeitreihenanalyse

In diesem Abschnitt werden Ihnen einige der wesentlichen Konzepte der Zeitreihenanalyse vorgestellt: Die *Komponenten* einer Zeitreihe und *Stationarität*.

Die folgende Abbildung zeigt ein Beispiel für eine Zeitreihe: Das Liniendiagramm zeigt die monatlichen Werte der Passagierzahlen einer Fluggesellschaft zwischen 1949 und 1961. Es gibt also für jeden Monat eine Instanz in den Daten. Das Ziel der Zeitreihenanalyse besteht darin, die zeitliche Struktur der Daten zu beschreiben, also ein Modell zu finden, das die monatlichen Änderungen erklärt. Mit einem solchen Modell kann man entweder retrospektiv die Daten analysieren, um Wissen zu extrahieren, oder sogar die Werte für die nächsten Zeitschritte vorhersagen.

![](https://data-science-crashkurs.de/_images/dbb6b1e14de782cf2966a5b5b22db49a8ec92fafe4b7f2dd7af8921f7f95fe11.png)

### Komponenten einer Zeitreihe

Bei der Auswahl einer Prognosemethode müssen wir zunächst die Muster in den Zeitreihendaten identifizieren und dann ein Modell auswählen, das die Muster angemessen erfassen kann. Eine beobachtete Zeitreihe kann in die folgenden **Komponenten** zerlegt werden:
* Level: der Durchschnittswert.
* Der Trend $T_t$:  eine langfristige steigende oder fallende Tendenz.
* Die *Saisonalität* der Zeitreihe $S_t$:  ein kurzfristiges, sich wiederholendes Muster zu festgelegten Zeiträumen, z. B. wöchentlich, monatlich oder jährlich.
* Zyklus:  ein sich kurzfristig wiederholendes Muster (nicht in festen Zeiträumen).
* Rauschen: unregelmäßige oder sogar zufällige, kurzfristige Schwankungen.
* Die *Autokorrelation* zwischen den Zeitpunkten $R_t$. Die Autokorrelation modelliert, wie der Wert zum Zeitpunkt $t$ von den vorherigen Zeitpunkten abhängt, also wie $x_t$ mit den Werten $x_{t-1}, x_{t-2}, ...$ korreliert ist. Die Autokorrelation modelliert die Änderungen der Zeitreihe, die nicht durch den Trend oder den saisonalen Effekt erfasst werden.

Saisonale und zyklische Muster können in Zeitreihendaten schwer zu unterscheiden sein. Wenn die Häufigkeit eines wiederkehrenden Musters kalendermäßig konstant ist, dann ist das Muster saisonal. Ansonsten ist es zyklisch.

Eine Zeitreihe wird immer als Summe aus verschiedenen Komponenten modelliert - dies drückt die Annahmen des Data Scientists aus. Eine vereinfachte Möglichkeit, eine Zeitreihe zu modellieren, wäre zum Beispiel, sie nur als Summe von Trend, Saisonalität und Autokorrelation auszudrücken:

$$x_t = T_t + S_t + R_t.$$

Wir können auch die Funktion `saisonal_decompose()` aus der Bibliothek `statsmodels` verwenden. Diese Zerlegungsmethode und teilt eine Zeitreihe in nur drei Komponenten: Trend, Saisonalität und Rauschen (Residuum).

![](https://site.wandb.ai/wp-content/uploads/2023/09/Screenshot-2023-09-29-at-14.21.00-1024x197.png)

![](https://site.wandb.ai/wp-content/uploads/2023/09/l35.png)


## Stationarität 

**Stationarität** beschreibt das Konzept, dass die Art und Weise, wie sich eine Zeitreihe verändert, auch in Zukunft gleich bleibt. Mathematisch ausgedrückt ist eine Zeitreihe stationär, wenn ihre statistischen Eigenschaften (Mittelwert, Varianz und Kovarianz) unabhängig von der Zeit sind. Das ist eine sehr wünschenswerte Eigenschaft, wenn man das Ziel hat, die Zukunft vorherzusagen!

Für eine Zeitreihe bedeutet Stationarität insbesondere auch, dass es keinen Trend und keine saisonalen Effekte geben darf - dies gilt üblicherweise nicht für echte Daten. Unser Ansatz, die Daten einer Zeitreihe stationär zu machen, besteht deshalb darin, dass wir zuerst den Trend $T_t$ und anschließend die Saisonalität $S_t$ modellieren und diese von den Daten abziehen. Im Detail ist der Ablauf wie folgt:

- Wir modellieren den Trend $T_t$ der Zeitreihe $\{x_t\}$. 
- Wir berechnen die trendbereinigte Zeitreihe $\hat{x}_t = x_t-T_t$.
- Wir modellieren die Saisonalität $S_t$ der trendbereinigten Zeitreihe $\{\hat{x}_t\}$. 
- Wir berechnen die trend- und saisonbereinigte Zeitreihe $\hat{\hat{x}}_t\ = \hat{x}_t -S_t$.

Wir betrachten nun jeweils zwei Ansätze für die Modellierung des Trends und der Saisonalität, welche wir anschließend verwenden, um die Zeitreihe stationär zu machen.

## Ansatz #1: Stationarität durch Regression und das saisonale Mittel

Der erste Ansatz für die Trendbereinigung ist eine Regression. Der Trend der Flugpassagierdaten sieht ungefähr linear aus, das heißt, wir sehen ein über die Zeit gleichmäßiges Wachstum. In diesem Fall können wir eine lineare Regression berechnen, um den Trend zu modellieren:

```python
import numpy as np
from sklearn.linear_model import LinearRegression


X = np.arange(1,air_passengers_df.shape[0]+1).reshape(-1,1)
Y = air_passengers_df.values

regr = LinearRegression()
regr.fit(X,Y)
T_pred = regr.predict(X)
air_passengers_detrended = air_passengers_df - T_pred

f, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].set_title('%.2f mehr Passagiere pro Monat (linearer Trend)' % regr.coef_[0])
axes[0].plot(X, Y)
axes[0].plot(X, T_pred, color='black')
axes[0].set_xlabel('Zeit $t$')
axes[0].set_ylabel('Passagiere')

axes[1].set_title('Trendbereinigte Zahl der Passagiere')
axes[1].plot(X, air_passengers_detrended)
axes[1].set_xlabel('Zeit $t$')
axes[1].set_ylabel('Passagiere')
plt.show()
```

![](https://data-science-crashkurs.de/_images/cf031d1485ce34f8d719858860d6036aef980ffacb6cb4566b179b2bdc092c8b.png)

Links sehen wir die lineare Regression der Zeitreihe. Wie man sieht, passt die Regression sehr gut zur Form der Zeitreihe. Die Steigung der Regression (also der Koeffizient) ist das monatliche Wachstum der Passagierzahlen. Dies ist der Vorteil der Regression zur Modellierung des Trends: Wir können nicht nur die trendbereinigte Zeitreihe berechnen, sondern bekommen auch noch eine genaue Interpretation des Trends, nämlich dass es 2,66 mehr Passagiere jeden Monat gibt.

Rechts sehen wir die trendbereinigte Zeitreihe. Man sieht, dass sich das Mittel der Werte nicht mehr erhöht, die Werte sind also in etwa zentriert um die 0. Dies sollte nach einer erfolgreichen Trendbereinigung immer der Fall sein. Wir können außerdem die Datenbereiche der y-Achsen vergleichen. Links war die Größe des Datenbereichs noch $600-100=500$. Rechts ist die Größe nur noch $150-(-100)=250$. Der Trend erklärt also in etwa 50% der absoluten Änderung der Werte über die Zeit.

Die trendbereinigten Daten haben noch ein klares Muster in der Schwankung rund um null. Das ist die **Saisonalität** der Daten. Da wir monatliche Daten haben, ist es eine vernünftige Annahme, dass es sich um ein jährliches Muster handelt. Dies können wir durch das *saisonale Mittel* erfassen. Hierzu müssen wir die Länge der Saison kennen, also die Anzahl von Zeitschritten, nach der eine Saison von vorne beginnt. In unseren Beispieldaten können wir die Saison durch das *monatliche Mittel* mit einer Länge von $s=12$ Zeitschritten für die Saison modellieren. Hierzu berechnen wir die Differenz der monatlichen Mittelwerte vom gesamten Mittelwert der Zeitreihe. Das können wir dann von der trendbereinigten Zeitreihe abziehen, um die trend- und saisonbereinigte Zeitreihe zu erhalten als

$$\hat{\hat{x}}_t = \hat{x}_t - (mean(\{\hat{x}_{t'}\}_{\{t' \in 1, ..., T: mod(t',s)=mod(t,s)\}}-mean(\{\hat{x}_t\})),$$

wobei $mod(t,s)$ der Rest der Division von $t$ durch $s$ ist. Für $s=12$ und $t=26$ ist $mod(t,s)=2$. Das saisonale Mittel würde man zum Beispiel für alle Zeitpunkte $t'$ mit $mod(t',s) = 2$ berechnen, also 2, 14, 26, 38 etc. Bei unseren Beispieldaten bekommen wir für eine Saison der Länge $s=12$ folgendes Ergebnis. 

```python
monthly_means = air_passengers_detrended.groupby(air_passengers_detrended.index.month).mean()

# mean of detrended data is zero
air_passengers_detrended_unseasonal = air_passengers_detrended - np.tile(monthly_means,12)

f, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].set_title('Monatliche Abweichung vom Mittelwert')
axes[0].bar(['Jan','Feb','Mär','Apr','Mai','Jun','Jul','Aug','Sep','Okt','Nov','Dez'], monthly_means)
axes[0].set_xlabel('Zeit $t$')
axes[0].set_ylabel('Passagiere')

axes[1].set_title('Trend- und saisonbereinigte Zahl der Passagiere')
axes[1].plot(X, air_passengers_detrended_unseasonal)
axes[1].set_xlabel('Zeit $t$')
axes[1].set_ylabel('Passagiere')
plt.show()
```

![](https://data-science-crashkurs.de/_images/a1f1d5c1baf3d4e6c6d3f3961b748aa33c58891ddbb2cab2ca91888df7a13627.png)

Wie man sieht, gibt es im Juli im Mittel über 60 Passagiere mehr als im Gesamtmittel. Im November sind es hingegen 60 Passagiere weniger. Der saisonale Effekt der anderen Monate schwankt zwischen diesen beiden Werten. Im März gibt es zum Beispiel fast gar keinen saisonalen Effekt. Das saisonale Mittel liefert, ähnlich wie die Regression, also eine genaue Information über die Veränderung.

Wenn wir den saisonalen Effekt entfernen, bekommen wir die Zeitreihe in der rechten Grafik. Der Datenbereich der y-Achse ist jetzt auf $100-(-40)=140$ reduziert. Durch Trend und Saisonalität konnten wir also bereits den Großteil der Varianz in den Daten erklären. Leider erkennt man immer noch ein Muster in den Daten. Das ursprüngliche saisonale Muster hatte eine Spitze nach oben und zwei direkt nebeneinander liegende Spitzen nach unten. Das saisonbereinigte Muster sieht etwas anders aus:

- Für $t<70$ gibt es zwei Spitzen nach oben und eine Spitze nach unten. 
- Zwischen $t=80$ und $t=100$ gibt es keine klar erkennbaren Spitzen. 
- Für $t>100$ gibt es eine Spitze nach oben und zwei Spitzen nach unten. 

Das Muster zeigt, dass der saisonale Effekt nicht vollständig korrigiert wurde. Der Grund dafür ist, dass der saisonale Effekt auch einen Trend hat: Genau wie die Zahl der Passagiere steigt, werden auch die Schwankungen während der Saison von Jahr zu Jahr stärker. In der Mitte der Daten passt die Schätzung des saisonalen Mittels. Zu Beginn der Daten ist die Schätzung des saisonalen Mittels höher als die eigentliche saisonale Abweichung, weshalb das saisonale Muster invertiert wird. Am Ende der Daten ist das saisonale Mittel kleiner als die eigentliche saisonale Abweichung, weshalb ein Teil des saisonalen Effekts nicht korrigiert wird.

Insgesamt ist unsere Zeitreihe also nicht vollständig stationär, da es immer noch Muster in den Daten gibt und insbesondere die Varianz daher nicht überall gleich ist. Dieses Muster ist jedoch relativ schwach, wenn man berücksichtigt, wie viel Information wir bereits durch den Trend und die Saison erklärt haben. Dies kann man sich veranschaulichen, indem wir die y-Achse wieder mit einem Datenbereich der ursprünglichen Größe von 500 anzeigen.

```python
monthly_means = air_passengers_detrended.groupby(air_passengers_detrended.index.month).mean()

# mean of detrended data is zero
air_passengers_detrended_unseasonal = air_passengers_detrended - np.tile(monthly_means,12)

fig, ax = plt.subplots()
ax.set_title('Trend- und saisonbereinigte Zahl der Passagiere\nUrsprüngliche Datenbereichsgröße auf der y-Achse')
ax.plot(X, air_passengers_detrended_unseasonal)
ax.set_xlabel('Zeit $t$')
ax.set_ylabel('Passagiere')
ax.set_ylim(-250,250)
plt.show()
```
![](https://data-science-crashkurs.de/_images/3e99a193ef915ca29452b32836d0828997faedcab1443846515174558b5e5c8f.png)


### Ansatz #2: Stationarität durch Differencing

Der zweite Ansatz zur Bereinigung des Trends und der Saison ist das sogenannte *Differencing*. Wie der Name schon vermuten lässt, ist das Differencing an das Konzept der Ableitung angelehnt und führt zu einer besseren Bereinigung als die Regression und das saisonale Mittel. Die Idee des Differencing ist, dass der Trend nichts anderes als die Steigung der Zeitreihe ist und die Steigung ist nichts anderes als die erste Ableitung. Wenn der Trend nicht konstant ist, sondern sich mit der Zeit ändert, sieht man dies in der zweiten Ableitung. Wenn wir also den Trend berechnen und bereinigen wollen, können wir die Ableitung hierfür nutzen.

Das Differencing basiert auf einer Variante der *Punktsteigungsformel* zum Berechnen der Steigung einer Geraden. Die Steigung zwischen zwei Punkten $(x_1, y_1), (x_2, y_2)$ ist dann definiert als $\frac{y_2-y_1}{x_2-x_1}$. Da wir äquidistante Zeitschritte $t$ haben, ist für zwei benachbarte Zeitpunkte $(t-1, x_{t-1}), (t, x_t)$ die Steigung

$$\frac{x_t-x_{t-1}}{t-(t-1)} = \frac{x_t-x_{t-1}}{1} = x_t-x_{t-1}.$$

Diese Steigung nennt man auch die *Differenz erster Ordnung* der Zeitreihe und wir schreiben

$$\Delta^1 x_t = x_t - x_{t-1}.$$

Mit der Differenz erster Ordnung kann man lineare Trends entfernen als  

$$\hat{x}_t = \Delta^1 x_t.$$

Wenn sich die Änderung des Mittelwerts mit der Zeit ändert, müssen wir die *Differenz zweiter Ordnung* berechnen, die definiert ist als

$$\Delta^2 x_t = \Delta^1 x_t - \Delta^1 x_{t-1} = x_t - 2\cdot x_{t-1} + x_{t-2}.$$

Da wir in etwa einen linearen Trend in unseren Daten haben, können wir ohne Weiteres die Differenz erster Ordnung zur Trendbereinigung benutzen.


```python
air_passengers_differenced = air_passengers_df.diff(1)

f, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].set_title('Luftfahrt-Passagiere über die Zeit')
axes[0].plot(X, Y)
axes[0].set_xlabel('Zeit $t$')
axes[0].set_ylabel('Passagiere')

axes[1].set_title('Trendbereinigte Zahl der Passagiere (d=1)')
axes[1].plot(X, air_passengers_differenced)
axes[1].set_xlabel('Zeit $t$')
axes[1].set_ylabel('Passagiere')
plt.show()
```

![](https://data-science-crashkurs.de/_images/9eafdd440f902596461a2141ed0f730b69310b1808012dd3d248ea676d2675bf.png)

Die trendbereinigte Zeitreihe ist um null zentriert, aber wir sehen immer noch saisonale Schwankungen. Der Datenbereich der y-Achse ist auf $75-(-100)=175$ reduziert, der Trend erklärt also bereits etwa zwei Drittel der Änderung der Werte über die Zeit. 

Differencing kann auch zur Modellierung der Saison genutzt werden. Hierzu betrachtet man aber nicht benachbarte Zeitpunkte, sondern die Differenz zwischen zwei aufeinander folgenden Punkten der Saison. Wenn unsere Saison eine Länge von $s$ hat, können wir die Saison bereinigen, indem wir $\hat{\hat{x}} = \Delta_s \hat{x}_t$ berechnen als

$$\Delta_s \hat{x}_t = \hat{x}_t - \hat{x}_{t-s}.$$

Das saisonale Differencing mit einer Saison von $s=12$ Zeitschritten führt zu folgendem Ergebnis. 

```python
air_passengers_differenced_seasonal = air_passengers_differenced.diff(periods=12)

f, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].set_title('Trendbereinigte Zahl der Passagiere (d=1)')
axes[0].plot(X, air_passengers_differenced)
axes[0].set_xlabel('Zeit $t$')
axes[0].set_ylabel('Passagiere')

axes[1].set_title('Saisonale Bereinigung mit Differencing ($s=12$)')
axes[1].plot(X, air_passengers_differenced_seasonal)
axes[1].set_xlabel('Zeit $t$')
axes[1].set_ylabel('Passagiere')
plt.show()
```

![](https://data-science-crashkurs.de/_images/98237ee3482739ada00de52cc5bc44b67a4052829c430ea31f038ae990f8b7e1.png)


Nach der Bereinigung ist auf der rechten Seite kein Muster mehr in den Daten zu erkennen. Die Schwankung um null sieht komplett zufällig aus, was ein starker Hinweis darauf ist, dass die Daten jetzt stationär sind. Der Bereich der y-Achse ist jetzt auf $40-(-40)=80$ reduziert, also noch mal halbiert gegenüber der trendbereinigten Zeitreihe.

### Vergleich der Ansätze

Beide Ansätze, den Trend und die Saison zu modellieren, haben Stärken und Schwächen. Der Vorteil der Regression und der saisonalen Mittelwerte ist die Interpretierbarkeit. Da dies jedoch nur bei linearen Trends und saisonalen Effekten ohne Trend funktioniert, ist die Anwendbarkeit des Verfahrens limitiert und die Daten sind hinterher möglicherweise nicht stationär. Beim Differencing ist das Gegenteil der Fall: Die Bereinigung des Trends und der saisonalen Effekte funktioniert in der Regel sehr gut, da man auch Differenzen höherer Ordnung verwenden kann. Dafür gibt es keine Formel, die man interpretieren kann.

<!-- 
@todo: Weiteres Kapitel zu ARMA, ARIMA & Co einfügen, im Wesentlichen als Kopie davon:
https://data-science-crashkurs.de/chapters/kapitel_09.html
-->

## ⚑  Quellen

Die hier genannten Texte, Grafiken und Code-Beispiele wurden übernommen, teilweise ins Deutsche übersetzt, erweitert und kombiniert auf Basis der folgenden beiden Quellen:
* [A Gentle Introduction to Time Series Analysis & Forecasting](https://wandb.ai/site/articles/a-gentle-introduction-to-time-series-analysis-forecasting) von Weights & Biased, Zugriff Feb 2024.
* [Data Science Crash Kurs, Kapitel 9: Zeitreihenanalye](https://data-science-crashkurs.de/chapters/kapitel_09.html) von Steffen Herbold, Zugriff Feb 2024.
{: .notice--info}





