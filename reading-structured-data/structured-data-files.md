---
title: "Strukturierte Daten lesen: Dateien"
layout: single
author_profile: true
author: Christina Kratsch
lecture_name: "Data Science"
lecture_desc: "Grundlegende Methoden für die Exploration und das Management von Daten."
licence: "CC-BY"
licence_desc: 2024 | HTW Berlin 
classes: wide
---


**Strukturierte Daten** sind Daten, die in einem bestimmten, vordefinierten Format vorliegen. Das macht es üblicherweise einfach, sie in Datenbanken oder Tabellen zu organisieren. Diese Daten sind in klar definierten Spalten und Zeilen angeordnet und folgen einem einheitlichen **Schema**. Beispiele für strukturierte Daten sind Datenbanken, Excel-Tabellen oder CSV-Dateien. Sie sind leicht zu durchsuchen, zu analysieren und zu verarbeiten, da die Datenfelder und ihre Beziehungen gut definiert sind.

Im Gegensatz dazu sind unstrukturierte Daten nicht in einem vordefinierten Format organisiert. Sie können Textdokumente, Bilder, Videos, Social-Media-Beiträge, E-Mails und andere Arten von Informationen umfassen, die nicht einfach in Tabellen oder Datenbanken eingeteilt werden können. 

In dieser Lektion lernen und wiederholen wir
* Daten aus gewöhnlichen Textdateien zu laden
* den Umgang mit `sqlite3` und grundlegende SQL-Befehle
* den Einsatz von `SQLAlchemy`.

## Einfache Dateien laden, lesen, schreiben

Wie jede Programmiersprache bietet Python einfache Boardmittel zum Öffnen und Lesen von Dateien:

```python
file = open(file='textfile.txt', mode='r')
text = file.readlines()
print(text)
``` 

Hier öffnen wir `textfile.txt` im Lese-Modus (ohne Schreibrechte) in einem relativen Pfad. 

**Hinweis:** Gerade zwischen unterschiedlichen Betriebssystemen macht die Schreibweise von Pfaden und der Zugriff auf Dateien immer mal Probleme. Ein Blick auf die detaillierte [Dokumentation der `open()` Funkion](https://docs.python.org/3/library/functions.html#open) kann hier vielleicht helfen.

Der Output der Funktion `readlines()` liefert uns in unserem Beispiel folgenden Text mit dem mittleren *Escsape Character* (hier: Zeilenumbruch) `\n`:
```
['This is a text file.\n', 'Now you can read it!']
```

Wir sollten abschließend nicht vergessen, die Datei auch wieder zu schließen:
```python
file.close()
```

Grundsätzlich gibt es aber auch elegantere Methoden zum Arbeiten mit Dateien mit Hilfe des `with` Statements:

```python
with open(file='textfile.txt', mode='r') as f:
    text = f.readlines()
print(text)
```

## JSON

JavaScript Object Notation (JSON) ist ein textbasiertes Format zur Darstellung und Speicherung von Daten und wird hauptsächlich für Webanwendungen verwendet. Wir können JSON verwenden, um Daten in einer Textdatei zu speichern oder Daten an und von einer Anwendungsprogrammierschnittstelle (API) zu übertragen.

APIs ermöglichen es uns, eine Anfrage an einen Webdienst zu senden, um verschiedene Aufgaben auszuführen. Ein Beispiel ist das Senden eines Texts (z. B. einer E-Mail-Nachricht) an eine API und das Zurücksenden der Stimmung (positiv, negativ oder neutral) an uns. JSON sieht einem Wörterbuch in Python sehr ähnlich. Hier ist zum Beispiel ein Beispielwörterbuch in Python, das Daten darüber speichert, wie viele Bücher und Artikel wir gelesen haben und was ihre Themen umfassen:

```json
data_dictionary = {
    'books': 12,
    'articles': 100,
   'subjects': ['math',
                'programming',
                'data science']}
```

Als Dictionary bietet dies eine praktische Datenstruktur, die die Basis einer umfangreichen Analyse sein könnte. Wir können diese Struktur in Python in eine Zeichenfolge im JSON-Format konvertieren, indem wir zuerst das integrierte JSON-Modul importieren und dann `json.dumps()` verwenden.

```python
import json
json_string = json.dumps(data_dictionary)
print(json_string)
```

Der resultierende String sieht so aus:
```
'{"books": 12, "articles": 100, "subjects": ["math", "programming", "data science"]}'
```

Diesen String können wir auch zurück in ein Dictionary verwandeln:

```python
data_dict = json.loads(json_string)
print(data_dict)
```

Um ein JSON dictionary aus einer Datei zu lesen, hilft der Befehl `json.load()`:

```python
with open('reading.json') as f:
    loaded_data = json.load(f)
print(loaded_data)
```

Diese wenigen Befehle können die Grundlage für umfassende Datenanalysen im Austausch mit Dateien oder APIs darstellen.

## Objekte mit Picke persistieren

Manchmal möchten wir ein Python-Objekt direkt speichern, beispielsweise ein Wörterbuch oder ein anderes Python-Objekt. Dies kann passieren, wenn wir Python-Code zur Datenverarbeitung oder -erfassung ausführen und die Ergebnisse für eine spätere Analyse speichern möchten. Eine einfache Möglichkeit hierfür ist die Verwendung des integrierten Pickle-Moduls. Wie bei vielen Dingen in Python ist Pickle ein wenig humorvoll – es ist wie das Einlegen von Gemüse, aber wir legen stattdessen Daten ein. Wenn wir beispielsweise dasselbe Datenwörterbuch verwenden, das wir zuvor verwendet haben (mit Daten zu Büchern und Artikeln, die wir gelesen haben), können wir dieses Wörterbuchobjekt wie folgt in einer Pickle-Datei speichern:

```python
import pickle as pk
data_dictionary = {
    'books': 12,
    'articles': 100,
   'subjects': ['math',
                'programming',
                'data science']}
with open('readings.pk', 'wb') as f:
    pk.dump(data_dictionary, f)
```

Sobald wir Daten in einer Pickle-Datei haben, können wir sie wie folgt in Python laden:

```python
with open('readings.pk', 'rb') as f:
    data = pk.load(f)
```

Beachten Sie, dass wir jetzt „rb“ für unser Modusargument verwenden, was „Binärdatei lesen“ bedeutet. Da die Datei im Binärformat vorliegt, müssen wir dies beim Öffnen der Datei angeben. Sobald wir die Datei geöffnet haben, laden wir den Inhalt der Datei mit pickle.load(f) in die Datenvariable (was im vorherigen Code pk.load(f) ist, da wir pickle als pk bezeichnet haben). Pickle ist eine großartige Möglichkeit, fast jedes Python-Objekt schnell und einfach zu speichern. Für bestimmte, organisierte Daten kann es jedoch besser sein, sie in einer SQL-Datenbank zu speichern, worauf wir jetzt eingehen.


## Quelle
 Dieses Modul wurde stark angelehnt an das Buch "[Pratical Data Science](https://learning.oreilly.com/library/view/practical-data-science/9781801071970/Text/Chapter_3.xhtml#_idParaDest-93)" von Aathan George.
{: .notice--info}

