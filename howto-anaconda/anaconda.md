---
title: "HowTo: Anaconda für virtuelle Umgebungen"
layout: single
author_profile: true
author: Christina Kratsch
lecture_name: "Data Science"
lecture_desc: "Grundlegende Methoden für die Exploration und das Management von Daten."
licence: "CC-BY"
licence_desc: 2024 | HTW Berlin 
toc: false
classes: wide
---

### Einführung in Anaconda

In der Welt der Datenwissenschaft und Python-Entwicklung ist Anaconda zu einem unverzichtbaren Werkzeug geworden. In diesem Artikel werden wir uns damit befassen, warum virtuelle Umgebungen wichtig sind, was Anaconda ist, wie man es installiert und wie man es verwendet.

### 1. Warum virtuelle Umgebungen?

Virtuelle Umgebungen sind entscheidend, um Projektabhängigkeiten zu verwalten und Konflikte zwischen verschiedenen Projekten zu vermeiden. Es ist realistisch, dass Sie Python gleichzeitig für ganz unterschiedliche Projekte (z.B. Data Science, Webentwicklung, Skripte) verwenden. Früher oder später kommen sich die dafür benötigten Libraries und Abhängigkeiten in die Quere. 

Indem Sie für jedes Projekt eine eigene, isolierte Python-Umgebung einrichten, in der Sie Bibliotheken und Pakete installieren können ohne die Systeminstallation zu beeinträchtigen, schaffen Sie Ordnung und Effizienz in der Ausführung und vermeiden Konflikte und Abhängigkeiten.

### 2. Was ist Anaconda?

Anaconda ist eine Open-Source-Distribution von Python und R, die speziell für Datenwissenschaft, maschinelles Lernen und wissenschaftliches Rechnen entwickelt wurde. Die Distribution wird mit einer Vielzahl von vorinstallierten Bibliotheken geliefert, darunter NumPy, pandas, matplotlib und viele mehr. Vor allem aber bietet Anaconda ein eigenes Paketverwaltungssystem, das es einfach macht, virtuelle Umgbungen zu verwalten und zusätzliche Pakete zu installieren.

Anaconda bietet den Konsolenbefehl `conda`, den Sie in vielen Tutorials finden werden:

```bash
conda --help
```

Mit dem Anaconda Navigator steht Ihnen aber eine ebenso mächtige graphische Umgebung zur Verwaltung Ihrer virtuellen Umgebungen zur Verfügung.


### 3. Wie installiere ich Anaconda?

Die Installation von Anaconda ist einfach und unkompliziert. Gehen Sie auf die [Anaconda-Website](https://www.anaconda.com/products/distribution) und laden Sie das entsprechende Installationspaket für Ihr Betriebssystem herunter. Folgen Sie dann den Anweisungen des Installationsassistenten.

### 4. Wie verwende ich Anaconda?

Nach der Installation können Sie Anaconda Navigator oder die Anaconda-Befehlszeile verwenden, um auf verschiedene Funktionen zuzugreifen. Mit dem Anaconda Navigator können Sie problemlos virtuelle Umgebungen erstellen, Pakete installieren und Anwendungen wie Jupyter Notebook starten. Die Anaconda-Befehlszeile bietet ähnliche Funktionen.

Ein Beispiel für die Erstellung einer virtuellen Umgebung mit Anaconda:

```bash
conda create --name myenv python=3.9
```

Aktivieren Sie dann die Umgebung:

* Unter Windows:

```bash
activate myenv
```

* Unter macOS und Linux:

```bash
source activate myenv
```

Nun sind Sie in der virtuellen Umgebung, hier steht Ihnen das eben frisch installierte Python (hier in der gewünschten Version 3.9) zur Verfügung. Sie können nun Pakete in Ihrer virtuellen Umgebung installieren und verwenden, ohne die Systeminstallation zu beeinträchtigen, hier exemplarisch für das Paket `pandas`:

```bash
conda install pandas
```

Wenn Sie in der virtuellen Umgebung zu einem beliebigen Zeitpunkt Python starten, steht Ihnen eine Python-Konfiguration mit genau den von Ihnen hier installierten Paketen zur Verfügung.

```bash
# python nutzen im conda environment
python myprogram.py
```

In diesem Artikel haben wir einen kurzen Überblick über Anaconda gegeben und warum es ein wichtiges Werkzeug für Python-Entwickler und Datenwissenschaftler ist. Von der Verwaltung virtueller Umgebungen bis zur Installation und Nutzung von Paketen bietet Anaconda eine benutzerfreundliche Plattform für die Entwicklung und Ausführung von Python-Anwendungen. Für weiterführende Themen verweisen wir auf den [User Guide der Anaconda Website](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).