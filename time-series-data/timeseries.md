---
title: "Zeitreihen visualisieren"
layout: single
author_profile: true
author: Christina Kratsch
lecture_name: "Data Science"
lecture_desc: "Grundlegende Methoden für die Exploration und das Management von Daten."
licence: "CC-BY"
licence_desc: 2024 | HTW Berlin 
classes: wide
---

Eine Zeitreihe ist ein Datensatz, dessen Werte zu unterschiedlichen, meist regelmäßigen Zeitpunkten gemessen werden. Viele Zeitreihen sind gleichmäßig in einer bestimmten Häufigkeit verteilt, beispielsweise stündliche Wettermessungen, tägliche Zählungen von Website-Besuchen oder monatliche Gesamtverkäufe. Zeitreihen können auch unregelmäßig verteilt und sporadisch sein, beispielsweise zeitgestempelte Daten im Ereignisprotokoll eines Computersystems oder eine Historie von Notrufen. Die Pandas-Zeitreihentools lassen sich für beide Arten von Zeitreihen gleichermaßen gut anwenden.

Dieses Tutorial konzentriert sich hauptsächlich auf die Datenverarbeitungs- und Visualisierungsaspekte der Zeitreihenanalyse. Anhand einer Zeitreihe von Energiedaten werden wir sehen, wie Techniken wie zeitbasierte Indizierung, Resampling und rollierende Fenster uns dabei helfen können, Schwankungen im Strombedarf und im Angebot an erneuerbaren Energien im Laufe der Zeit zu untersuchen. Wir werden folgende Themen behandeln:

* Der Datensatz: Open Power Systems Data
* Zeitreihendatenstrukturen
* Zeitbasierte Indizierung
* Visualisierung von Zeitreihendaten
* Saisonalität
* Frequenzen
* Resampling
* Rollfenster
* Trends

Wir werden dazu Pandas, Matplotlib und Seaborn verwenden.

## Der Datensatz: Open Power Systems Data

In diesem Tutorial arbeiten wir mit täglichen Zeitreihen von Open Power System Data (OPSD) für Deutschland, das in den letzten Jahren seine Produktion erneuerbarer Energien rasant ausgebaut hat. Der Datensatz umfasst landesweite Gesamtwerte des Stromverbrauchs, der Windenergieproduktion und der Solarenergieproduktion für den Zeitraum 2006–2017. Die Daten können Sie [hier](https://github.com/jenfly/opsd/raw/master/opsd_germany_daily.csv) herunterladen.

Stromproduktion und -verbrauch werden als Tagessummen in Gigawattstunden (GWh) angegeben. Die Spalten der Datendatei sind:

* Date – Das Datum (Format JJJJ-MM-TT)
* Consumption – Stromverbrauch in GWh
* Wind – Windstromproduktion in GWh
* Solar – Solarstromproduktion in GWh
* Wind+Solar – Summe der Wind- und Solarstromproduktion in GWh

Wir werden untersuchen, wie sich Stromverbrauch und -produktion in Deutschland im Laufe der Zeit verändert haben, und mithilfe von Pandas-Zeitreihentools Fragen beantworten wie:
* Wann ist der Stromverbrauch typischerweise am höchsten und wann am niedrigsten?
* Wie variiert die Wind- und Solarenergieproduktion je nach Jahreszeit?
* Was sind die langfristigen Trends beim Stromverbrauch, bei Solarenergie und Windkraft?
* Wie verhält sich die Stromerzeugung aus Wind- und Solarenergie zum Stromverbrauch und wie hat sich dieses Verhältnis im Laufe der Zeit verändert?

## Datenstrukturen für Zeitreihen

Bevor wir uns mit den OPSD-Daten befassen, stellen wir kurz die wichtigsten Pandas-Datenstrukturen für die Arbeit mit Datums- und Uhrzeitangaben vor. Bei Pandas wird ein einzelner Zeitpunkt als `Timestamp` dargestellt. Mit der Funktion to_datetime() können wir Timestamp aus Zeichenfolgen in einer Vielzahl von Datums-/Uhrzeitformaten erstellen. Lassen Sie uns Pandas importieren und einige Datums- und Uhrzeitangaben in Timestamp konvertieren.

```python
import pandas as pd
pd.to_datetime('2018-01-15 15:45')
Timestamp('2018-01-15 15:45:00')
pd.to_datetime('7/8/1952')
Timestamp('1952-07-08 00:00:00')
```

Wie wir sehen können, leitet `to_datetime()` automatisch ein Datums-/Uhrzeitformat basierend auf der Eingabe ab. Im obigen Beispiel wird das mehrdeutige Datum „7/8/1952“ als Monat/Tag/Jahr angenommen und als 8. Juli 1952 interpretiert. Alternativ können wir den Parameter `dayfirst` verwenden, um Pandas anzuweisen, das Datum als August zu interpretieren 7, 1952.

```python
pd.to_datetime('7/8/1952, dayfirst=True)
Timestamp('1952-08-07 00:00:00')
```

Wenn wir `to_datetime()` eine Liste oder ein Array von Zeichenfolgen als Eingabe bereitstellen, wird eine Folge von Datums-/Uhrzeitwerten in einem DatetimeIndex-Objekt zurückgegeben. Dabei handelt es sich um die Kerndatenstruktur, die einen Großteil der Pandas-Zeitreihenfunktionalität unterstützt.

```python
pd.to_datetime(['05.01.2018', '08.07.1952', '10. Okt. 1995'])
DatetimeIndex(['2018-01-05', '1952-07-08', '1995-10-10'], dtype='datetime64[ns]', freq=None)
```

Im obigen DatetimeIndex gibt der Datentyp `datetime64[ns]` an, dass die zugrunde liegenden Daten als 64-Bit-Ganzzahlen in Einheiten von Nanosekunden (ns) gespeichert werden. Diese Datenstruktur ermöglicht es Pandas, große Sequenzen von Datums-/Uhrzeitwerten kompakt zu speichern und vektorisierte Operationen mithilfe von [NumPy datetime64-Arrays](https://docs.scipy.org/doc/numpy-1.15.0/reference/arrays.datetime.html) effizient durchzuführen.

## Erstellen eines Zeitreihen-DataFrames

Um mit Zeitreihendaten in Pandas zu arbeiten, verwenden wir einen `DatetimeIndex` als Index für unseren DataFrame (oder unsere Serie). Sehen wir uns an, wie das mit unserem OPSD-Datensatz geht. Zuerst verwenden wir die Funktion `read_csv()`, um die Daten in einen DataFrame einzulesen und dann ihre Form anzuzeigen.

```python
opsd_daily = pd.read_csv('opsd_Germany_daily.csv')
opsd_daily.shape
```
Der DataFrame verfügt über 4383 Zeilen und deckt den Zeitraum vom 1. Januar 2006 bis zum 31. Dezember 2017 ab. Um zu sehen, wie die Daten aussehen, verwenden wir die Methoden `head()` und `tail()`, um die ersten drei und letzten drei Zeilen anzuzeigen.

```python
opsd_daily.head(3)
```

Schauen wir uns als Nächstes die Datentypen jeder Spalte an.

```python
opsd_daily.dtypes
```

Da die Datumsspalte nun den richtigen Datentyp hat, legen wir sie als Index des DataFrame fest.

```python
opsd_daily = opsd_daily.set_index('Date')
opsd_daily.head(3)
```

```python
opsd_daily.index
```

Alternativ können wir die obigen Schritte in einer einzigen Zeile zusammenfassen, indem wir die Parameter `index_col` und `parse_dates` der Funktion `read_csv()` verwenden. Dies ist oft eine nützliche Abkürzung.


Referenzen:
https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/

```python
opsd_daily = pd.read_csv('opsd_germany_daily.csv', index_col=0, parse_dates=True)
```

## Erste Zugriffe

Da der Index unseres DataFrame nun ein DatetimeIndex ist, können wir die gesamte leistungsstarke zeitbasierte Indizierung von Pandas nutzen, um unsere Daten zu bearbeiten und zu analysieren, wie wir in den folgenden Abschnitten sehen werden.

Ein weiterer nützlicher Aspekt des DatetimeIndex besteht darin, dass die einzelnen Datums-/Uhrzeitkomponenten alle als Attribute wie Jahr, Monat, Tag usw. verfügbar sind. Fügen wir opsd_daily ein paar weitere Spalten hinzu, die den Jahres-, Monats- und Wochentagsnamen enthalten.

```python
# Add columns with year, month, and weekday name
opsd_daily['Year'] = opsd_daily.index.year
opsd_daily['Month'] = opsd_daily.index.month
opsd_daily['Weekday Name'] = opsd_daily.index.day_name()
# Display a random sampling of 5 rows
opsd_daily.sample(5, random_state=0)
```

## Time-based indexing

Eine der leistungsstärksten und praktischsten Funktionen von Pandas-Zeitreihen ist die zeitbasierte Indizierung – die Verwendung von Datums- und Uhrzeitangaben zur intuitiven Organisation und zum Zugriff auf unsere Daten. Bei der zeitbasierten Indizierung können wir Datums-/Uhrzeitformatierte Zeichenfolgen verwenden, um Daten in unserem DataFrame mit dem loc-Accessor auszuwählen. Die Indizierung funktioniert ähnlich wie die standardmäßige labelbasierte Indizierung mit loc, verfügt jedoch über einige zusätzliche Funktionen.

Beispielsweise können wir Daten für einen einzelnen Tag mithilfe einer Zeichenfolge wie „10.08.2017“ auswählen.

```python
opsd_daily.loc['2017-08-10']
```

Wir können auch einen Teil der Tage auswählen, z. B. „20.01.2014“: „22.01.2014“. Wie bei der regulären labelbasierten Indizierung mit loc umfasst der Slice beide Endpunkte.


```python
opsd_daily.loc['2014-01-20':'2014-01-22']
```

Eine weitere sehr praktische Funktion der Pandas-Zeitreihen ist die teilweise String-Indizierung, bei der wir alle Datums- und Uhrzeitangaben auswählen können, die teilweise mit einer bestimmten Zeichenfolge übereinstimmen. Beispielsweise können wir mit opsd_daily.loc['2006'] das gesamte Jahr 2006 oder mit opsd_daily.loc['2012-02'] den gesamten Monat Februar 2012 auswählen.

```python
opsd_daily.loc['2012-02']
```

## Visualisierung von Zeitreihendaten

Mit Pandas und Matplotlib können wir unsere Zeitreihendaten einfach visualisieren. In diesem Abschnitt behandeln wir einige Beispiele und einige nützliche Anpassungen für unsere Zeitreihendiagramme. Wir verwenden für unsere Diagramme den Seaborn-Stil und passen die Standardfigurengröße an eine geeignete Form für Zeitreihendiagramme an.

Erstellen wir mithilfe der plot()-Methode des DataFrame ein Liniendiagramm der gesamten Zeitreihe des täglichen Stromverbrauchs in Deutschland.

```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11, 4)})
opsd_daily['Consumption'].plot(linewidth=0.5);
```

Wir können sehen, dass die plot()-Methode ziemlich gute Teilstrichpositionen (alle zwei Jahre) und Beschriftungen (die Jahre) für die x-Achse ausgewählt hat, was hilfreich ist. Bei so vielen Datenpunkten ist das Liniendiagramm jedoch überfüllt und schwer zu lesen. Lassen Sie uns stattdessen die Daten als Punkte darstellen und uns auch die Sonnen- und Windzeitreihen ansehen.

```python
cols_plot = ['Consumption', 'Solar', 'Wind']
axes = opsd_daily[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily Totals (GWh)')
```

Wir können bereits einige interessante Muster erkennen:

* Der Stromverbrauch ist im Winter am höchsten, vermutlich aufgrund der Elektroheizung und des erhöhten Beleuchtungsverbrauchs, und im Sommer am niedrigsten.
* Der Stromverbrauch scheint sich in zwei Gruppen aufzuspalten – eine mit Schwankungen, die etwa bei 1400 GWh zentriert sind, und eine andere mit weniger und stärker verstreuten Datenpunkten, die etwa bei 1150 GWh zentriert sind. Wir könnten vermuten, dass diese Cluster Wochentagen und Wochenenden entsprechen, und wir werden dies in Kürze weiter untersuchen.
* Die Solarstromproduktion ist im Sommer am höchsten, wenn das Sonnenlicht am stärksten ist, und im Winter am niedrigsten.
* Die Windenergieproduktion ist im Winter am höchsten, vermutlich aufgrund stärkerer Winde und häufigerer Stürme, und im Sommer am niedrigsten.
* Es scheint im Laufe der Jahre einen stark steigenden Trend bei der Windenergieproduktion zu geben.

Alle drei Zeitreihen weisen eindeutig eine Periodizität auf – in der Zeitreihenanalyse oft als **Saisonalität** bezeichnet –, bei der sich ein Muster in regelmäßigen Zeitabständen immer wieder wiederholt. Die Verbrauchs-, Solar- und Windzeitreihen schwanken auf einer jährlichen Zeitskala zwischen hohen und niedrigen Werten, entsprechend den saisonalen Wetteränderungen im Laufe des Jahres. Saisonalität im Allgemeinen muss jedoch nicht mit den meteorologischen Jahreszeiten übereinstimmen. Beispielsweise weisen Daten zu Einzelhandelsumsätzen häufig jährliche Saisonalität auf, wobei die Umsätze im November und Dezember vor den Feiertagen zunahmen.

Saisonalität kann auch auf anderen Zeitskalen auftreten. Das obige Diagramm deutet darauf hin, dass es beim Stromverbrauch in Deutschland möglicherweise eine gewisse wöchentliche Saisonalität gibt, die an Wochentagen und Wochenenden auftritt. Lassen Sie uns die Zeitreihe in einem einzelnen Jahr grafisch darstellen, um weitere Untersuchungen durchzuführen.


```python
ax = opsd_daily.loc['2017', 'Consumption'].plot()
ax.set_ylabel('Daily Consumption (GWh)');
```

Jetzt können wir die wöchentlichen Schwankungen deutlich erkennen. Ein weiteres interessantes Merkmal, das auf dieser Granularitätsebene deutlich wird, ist der drastische Rückgang des Stromverbrauchs Anfang Januar und Ende Dezember, während der Feiertage.

Lassen Sie uns weiter hineinzoomen und uns nur Januar und Februar ansehen.

```python
ax = opsd_daily.loc['2017-01':'2017-02', 'Consumption'].plot(marker='o', linestyle='-')
ax.set_ylabel('Daily Consumption (GWh)');
```

Wie wir vermutet haben, ist der Verbrauch an Wochentagen am höchsten und am Wochenende am niedrigsten.

## Anpassen von Zeitreihendiagrammen

Um die wöchentliche Saisonalität des Stromverbrauchs im obigen Diagramm besser zu veranschaulichen, wäre es schön, vertikale Gitterlinien auf einer wöchentlichen Zeitskala zu haben (statt am ersten Tag jedes Monats). Wir können unseren Plot mit `matplotlib.dates` anpassen.

Da Datums-/Uhrzeit-Ticks in matplotlib.dates etwas anders gehandhabt werden als in der plot()-Methode des DataFrame, erstellen wir den Plot direkt in matplotlib. Dann verwenden wir `mdates.WeekdayLocator()` und `mdates.MONDAY`, um die Ticks der x-Achse auf den ersten Montag jeder Woche zu setzen. Wir verwenden auch mdates.DateFormatter(), um die Formatierung der Teilstrichbeschriftungen zu verbessern, indem wir die Formatcodes verwenden, die wir zuvor gesehen haben.


myFmt = mdates.DateFormatter('%d')
ax.xaxis.set_major_formatter(myFmt)

```python
import matplotlib.dates as mdates
fig, ax = plt.subplots()
ax.plot(opsd_daily.loc['2017-01':'2017-02', 'Consumption'], marker='o', linestyle='-')
ax.set_ylabel('Daily Consumption (GWh)')
ax.set_title('Jan-Feb 2017 Electricity Consumption')
# Set x-axis major ticks to weekly interval, on Mondays
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
# Format x-tick labels as 3-letter month name and day number
myFmt = mdates.DateFormatter('%m')
ax.xaxis.set_major_formatter(myFmt)
```

Jetzt haben wir an jedem Montag vertikale Gitternetzlinien und schön formatierte Häkchenbeschriftungen, sodass wir leicht erkennen können, welche Tage Wochentage und Wochenenden sind.

Es gibt viele andere Möglichkeiten, Zeitreihen zu visualisieren, je nachdem, welche Muster Sie untersuchen möchten – Streudiagramme, Heatmaps, Histogramme usw. In den folgenden Abschnitten sehen wir weitere Visualisierungsbeispiele, einschließlich Visualisierungen von Zeitreihendaten, die auf irgendeine Weise transformiert wurden, beispielsweise aggregierte oder geglättete Daten.

## Saisonalität

Lassen Sie uns als Nächstes die Saisonalität unserer Daten mit Boxplots weiter untersuchen, indem wir die boxplot()-Funktion von Seaborn verwenden, um die Daten nach verschiedenen Zeiträumen zu gruppieren und die Verteilungen für jede Gruppe anzuzeigen. Wir gruppieren die Daten zunächst nach Monaten, um die jährliche Saisonalität zu visualisieren.

```python
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for name, ax in zip(['Consumption', 'Solar', 'Wind'], axes):
sns.boxplot(data=opsd_daily, x='Month', y=name, ax=ax)
ax.set_ylabel('GWh')
ax.set_title(name)
# Remove the automatic x-axis label from all but the bottom subplot
if ax != axes[-1]:
    ax.set_xlabel('')
```

Diese Boxdiagramme bestätigen die jährliche Saisonalität, die wir in früheren Diagrammen gesehen haben, und liefern einige zusätzliche Erkenntnisse: 
* Obwohl der Stromverbrauch im Winter im Allgemeinen höher und im Sommer niedriger ist, sind der Median und die unteren beiden Quartile im Dezember und Januar niedriger als im November und Februar, wahrscheinlich aufgrund der Schließung von Geschäften über die Feiertage. Wir haben dies in der Zeitreihe für das Jahr 2017 gesehen, und der Boxplot bestätigt, dass dieses Muster über die Jahre hinweg konsistent ist. 
* Während sowohl die Solar- als auch die Windenergieproduktion eine jährliche Saisonalität aufweisen, weist die Windenergieverteilung viel mehr Ausreißer auf, was die Auswirkungen gelegentlicher extremer Windgeschwindigkeiten im Zusammenhang mit Stürmen und anderen vorübergehenden Wetterbedingungen widerspiegelt.

Als Nächstes gruppieren wir die Zeitreihe zum Stromverbrauch nach Wochentagen, um die wöchentliche Saisonalität zu untersuchen.

```python
sns.boxplot(data=opsd_daily, x='Weekday Name', y='Consumption');
```

Erwartungsgemäß ist der Stromverbrauch an Wochentagen deutlich höher als am Wochenende. Die geringen Ausreißer an Wochentagen liegen vermutlich an Feiertagen.

## Weiterführende Inhalte

In diesem Abschnitt wurde eine kurze Einführung in die Saisonalität von Zeitreihen gegeben. In einer weiterführenden Analyse kann die Anwendung eines rollierenden Fensters auf die Daten auch dabei helfen, Saisonalität auf verschiedenen Zeitskalen zu visualisieren. Andere Techniken zur Analyse der Saisonalität umfassen [Autokorrelationsdiagramme](https://pandas.pydata.org/pandas-docs/stable/visualization.html#autocorrelation-plot), welche die Korrelationskoeffizienten der Zeitreihe mit sich selbst bei unterschiedlichen Zeitverzögerungen darstellen.

Zeitreihen mit starker Saisonalität lassen sich oft gut mit Modellen darstellen, die das Signal in Saisonalität und einen langfristigen Trend zerlegen, und diese Modelle können zur Vorhersage zukünftiger Werte der Zeitreihe verwendet werden. Ein einfaches Beispiel für ein solches Modell ist die [klassische saisonale Deckomposition (Zerlegung)](https://otexts.org/fpp2/classical-decomposition.html), wie [in diesem Tutorial](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/) gezeigt. Ein ausgefeilteres Beispiel ist das [Prophet-Modell](https://facebook.github.io/prophet/) von Facebook, das die Zeitreihe mittels Kurvenanpassung zerlegt und dabei Saisonalität auf mehreren Zeitskalen, Feiertagseffekte, abrupte Wendepunkte und langfristige Trends berücksichtigt, wie in [diesem Tutorial](https://towardsdatascience.com/time-series-analysis-in-python-an-introduction-70d5a5b1d52a) gezeigt.


## Quellen

**⚑ Referenzen** <br> Diese Lektion wurde aus einem Tutorial namens ["Tutorial: Time Series Analysis with Pandas"](https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/) von Jennifer Walker entnommen und ins Deutsche übertragen. Das Tutorial enthält auch weiterführende Anleitungen und Referenzen.
{: .notice--info}