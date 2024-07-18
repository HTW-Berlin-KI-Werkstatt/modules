---
title: "HowTo: Git"
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

### Warum Git?

Durch die Verwendung von <em>Git</em> können Entwickler Änderungen verfolgen, Branches erstellen, verschiedene Versionen verwalten und problemlos zwischen ihnen wechseln. Das ist wichtig, weil es eine Versionskontrolle ermöglicht, welche die Zusammenarbeit an Softwareprojekten erleichtert und die Entwicklung effizienter gestaltet. Es ermöglicht auch das Zusammenführen von Codeänderungen mehrerer Entwickler und das effektive Management von Konflikten. Darüber hinaus bietet Git eine zuverlässige Sicherung der Projektgeschichte und ermöglicht es Teams, agil zu arbeiten, indem sie flexibel auf Anforderungsänderungen reagieren können. Insgesamt trägt Git dazu bei, den Entwicklungsprozess transparenter, nachvollziehbarer und kollaborativer zu gestalten.

### Git lernen

Um Git zu erlernen, stehen im Internet unzählige Tutorials zur Verfügung, wir listen hier einige Empfehlungen:
* [einen Schnellkurs zum Durchlesen](https://lerneprogrammieren.de/git/) oder
* [einen detaillierten Lernpfad mit interaktiven Anwendungen](https://learn.microsoft.com/de-de/training/paths/intro-to-vc-git/)

Für alle Betriebssysteme finden Sie [hier](https://git-scm.com/) gute Downloads und Anleitungen, um Git auf Ihrem Betriebssystem einzurichten.

Darüber hinaus sollten die ersten fünf Kapitel [dieses Onlinebuchs von Git-SCM](https://git-scm.com/book/en/v2) Sie auf alles vorbereiten, was es über Git im Alltag zu wissen gibt.

Die Kurse von Git SCM gibt es auch als Video:

[<img src="https://git-scm.com/images/video/ep1.png">](https://git-scm.com/video/what-is-version-control) 
[<img src="https://git-scm.com/images/video/ep2.png">](https://git-scm.com/video/what-is-git)<br><br>
[<img src="https://git-scm.com/images/video/ep3.png">](https://git-scm.com/video/get-going)
[<img src="https://git-scm.com/images/video/ep4.png">](https://git-scm.com/video/quick-wins)<br><br>

Für Nutzer von Visual Studio ist vielleicht diese Seite interessant:
* [How to use Visual Studio with Git](https://learn.microsoft.com/en-us/visualstudio/version-control/git-with-visual-studio?view=vs-2022) 

**Aufgabe 1:** Schließen Sie Ihre eigene Git-Schulung mit einer der hier genannten Ressourcen oder einen vergleichbaren Kurs ab. Stellen Sie sicher, dass Ihnen Konzepte wie `clone`, `commit`, `push`, `fetch` sowie Branches und das Entstehen von Merge Konflikten vertraut sind. 
{: .notice--warning}

**Aufgabe 2:** Die meisten Kurse zu Git konzentrieren sich auf den Umgang mit der Konsole. Wenn Sie Git in Ihrer IDE verwenden möchten, stellen Sie sicher, dass Ihnen bewusst ist, welcher Button welchem Konsolenbefehl entspricht.
{: .notice--warning}

### Git und Visual Studio

Auch wenn Ihnen die Theorie im Umgang mit Git bekannt ist, führt die konkrete Umsetzung mit Visual Studio manchmal zu Problemen. Folgen Sie einem der beiden Workflows unten, wenn Sie ein neues C#-Projekt unter Versionskontrolle bringen wollen. 

**Hinweis:** Wenn Sie in einem Team arbeiten, müssen die folgenden Schritte zur Initialisierung Ihres Projektes nur von einem einzigen Teammitglied ausgeführt werden. Sprechen Sie sich vorher im Team ab.
{: .notice--info} 

#### Falls noch kein Remote Repository exisitiert

1. Legen Sie ein neues Repository im GitLab an. Achten Sie darauf, dass das Repository leer ist (kein Template, keine Readme, keine Lizenz...).
2. Notieren Sie sich die URL des neuen Repositories (zugänglich über den Clone-Button), üblicherweise hat das die Form
   ```
   https://gitlab.rz.htw-berlin.de/user/repo_name.git
   ```
3. Öffnen Sie lokal auf Ihrem Rechner Ihr Visual Studio und legen Sie ein neues Projekt (C# Konsolenapp, Windows Forms, ...) an.
4. Jetzt fügen wir Versionskontrolle zum Projekt hinzu. Wählen Sie im Visual Studio **Git** > **Create Git Repository**.
5. Im nun geöffneten Fenster wählen Sie links unter "Other" ein **Existing Remote** und fügen dort die URL Ihres eben neu angelegten Repositories ein.

**Achtung:** Es ist **sehr** zu empfehlen, Ihr Projekt jetzt mit den wichtigsten Metadaten auszustatten. Wählen Sie im Dialog die Option "Add Readme", wählen Sie schon jetzt eine Lizenz, und, vor allem, **wählen Sie ein gitignore-Template** (der *Visual Studio Default* genügt)! Sehr, sehr viel Ärger bleibt Ihnen an späterer Stelle dadurch erspart.
{: .notice--danger} 

6. Wählen Sie "Create and Push".


#### Falls bereits ein Remote Repository exisitiert

Manchmal wurde bereits - aus welchen Gründen auch immer, ein Repository für Sie angelegt. Eventuell befinden sich auch schon Dateien darin, aber keine Visual Studio Solution. Ein einfaches "Create & Push" wie oben wird in diesem Fall zu Problemen führen. Gehen Sie dann stattdessen folgendermaßen vor:

1. Notieren Sie sich die URL des bestehenden Repositories (zugänglich über den Clone-Button), üblicherweise hat das die Form
   ```
   https://gitlab.rz.htw-berlin.de/user/repo_name.git
   ```
2. Öffnen Sie Visual Studio. Legen Sie KEIN neues Projekt an, sondern wählen Sie beim Start die Option "Clone from Repository". 
3. Clonen Sie das exisitierende Repository in einen passenden neuen Projektordner. Sie sollten nun einen Repository Explorer sehen, der Ihnen Ihre bestehenden Dateien zeigt.
4. Wählen Sie nun **New** > **Project** und konfigurieren Sie das passende Projekt. Achten Sie darauf, dass Ihre Projekt- und Solution-Datei in den Ordner des eben geklonten Repositores gelegt werden.

**Achtung:** Bevor Sie nun die neu erstellen Projekt- und Solution-Dateien als Änderungen ins Repository pushen, empfiehlt es sich **sehr**, noch eine `.gitignore`-Datei hinzuzugügen, falls diese noch nicht existiert. Prüfen Sie auch im *Folder View* Ihres Solution Explorers (oder alternativ einfach im Windows Explorer), dass die Datei im Hauptverzeichnis liegt.  
{: .notice--danger} 

Tipp: Als Einträge für Ihre `.gitignore` können Sie [dieses Template](https://github.com/github/gitignore/blob/main/VisualStudio.gitignore) verwenden.
{: .notice--info}

5. Nachdem Sie die `.gitignore` eingerichtet haben, sollten sich die getrackten Changes in Ihrem Git Explorer auf wenige Dateien beschränken. Diese können Sie nun committen und pushen.

### Git und Python

Da Python mit deutlich weniger Projekt- und Konfigurationsdateien arbeitet als z.B. C#, wird der Umgang mit Python von vielen IDEs deutlich loser und Datei-basierter gehandhabt. Folgen Sie im Zweifel den Anweisungen Ihrer IDE und den etablierten best practices für eine Ordnerstruktur (z.B. bei graphischen oder Webanwendungen). 

Eine `.gitignore` zu haben empfiehlt sich hier aber ebenso, folgen Sie z.B. den [offizellen Empfehlungen von GitHub für Python-Projekte](https://github.com/github/gitignore/blob/main/Python.gitignore).