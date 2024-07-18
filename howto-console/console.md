---
title: "Arbeiten mit der Konsole"
layout: single
author_profile: true
author: Christina Kratsch
lecture_name: "Basic HowTo"
lecture_desc: "Technische Erklärungen"
licence: "CC-BY"
licence_desc: 2024 | HTW Berlin 
toc: true
toc_label: "Inhaltsverzeichnis"
toc_icon: "cog"
toc_sticky: true 
---

Die Konsole bzw. das Terminal ist wie ein guter Freund:
* immer verfügbar (jedes Betriebssystem, jede Hardware unterstützt Konsolen)
* immer zuverlässig (macht ganz minimalistisch genau das, was gesagt wird, ohne UX-Schnickschnack)
* immer ehrlich (liefert alle Warnungen und Fehler direkt lesbar).

Hier finden Sie ein paar grundlegende Funktionalitäten, die Ihnen in allen Lebenslagen weiterhelfen.

**Anmerkungen:**
- MacOS Nutzer finden eine Konsole über die App Terminal.
- Windows Nutzer finden eine Konsole über cmd.exe oder Powershell, dort sind aber einige Befelhle anders benannt. Stattdessen empfiehlt sich sehr ein Blick in [WSL](https://learn.microsoft.com/en-us/windows/wsl/about).
- Python-Entwickler unter Windows benötigen kein WSL, das [Anaconda Prompt](https://www.anaconda.com/) ist mächtig genug.

## Grundlegende Konsolen-Befehle

- `ls`: Auflisten von Dateien und Verzeichnissen in einem Verzeichnis.
    - Beispiel: `ls /home/user` oder nur `ls` für das aktuelle Verzeichnis
- `cd`: Wechseln des Verzeichnisses.
    - Beispiel: `cd /home/user/Documents`
- `pwd`: Anzeigen des aktuellen Verzeichnisses.
    - Beispiel: `pwd`
- `touch`: Erstellen einer leeren Datei.
    - Beispiel: `touch example.txt`
- `cp`: Kopieren von Dateien und Verzeichnissen.
    - Beispiel: `cp source.txt destination.txt`
- `mv`: Verschieben oder Umbenennen von Dateien und Verzeichnissen.
    - Beispiel: `mv old.txt new.txt`
    - Beispiel: `mv datei.txt Verzeichnis/`
- `rm`: Löschen von Dateien und Verzeichnissen.
    - Beispiel: `rm example.txt`
    - Beispiel: `rm -r Verzeichnis`
    - Achtung: kein Papierkorb!!!
- `cat`: Anzeigen und Zusammenfügen von Dateien.
    - Beispiel: `cat file.txt`
- `echo`: Ausgeben von Text oder Variablen.
    - Beispiel: `echo Hello World`
    - Beispiel: `echo $PATH` zum Ausgeben der Variable
- `man`: Anzeigen der Handbuchseite eines Befehls.
    - Beispiel: `man ls`
- `exit`: Beenden der Shell oder des aktuellen Prozesses.

Bitte beachten Sie, dass die genaue Funktionsweise dieser Befehle von Ihrem Betriebssystem abhängen kann. Es ist immer eine gute Idee, die man-Seiten oder die Hilfe (--help) für einen Befehl zu konsultieren, wenn Sie sich nicht sicher sind.


## Befehle für die Arbeit mit Git

Ich empfehle grundsätzlich, git auf der Konsole zu steuern, weil man sehr viel mehr Feedback zum Ablauf und Erfolg der Aktionen bekommt.

- `git init`: Initialisiert ein neues Git-Repository im aktuellen Ordner
    - Beispiel: `git init`
- `git clone`: Klont ein Repository in ein neues Verzeichnis.
    - Beispiel: `git clone https://github.com/user/repo.git`
- `git add`: Fügt Dateiänderungen im Arbeitsverzeichnis zur Staging-Area hinzu.
    - Beispiel: `git add .`
    - Beispiel: `git add datei.txt`
- `git commit`: Commitet die Änderungen in der Staging-Area.
    - Beispiel: `git commit -m "Commit message"`
- `git status`: Zeigt den Status des Arbeitsverzeichnisses und der Staging-Area an.
- `git pull`: Holt Änderungen von einem Remote-Repository und integriert sie.
    - Beispiel: `git pull origin main` holt die branch `main` vom Remote `origin` und mergt sie.
- `git push`: Schickt Änderungen an ein Remote-Repository.
    - Beispiel: `git push origin main` schickt den aktuellen Stand vom lokalen `main` ins Remote `origin`.
- `git branch`: Listet, erstellt oder löscht Branches.
    - Beispiel: `git branch feature-branch`
- `git checkout`: Wechselt Branches oder stellt Dateien wieder her.
    - Beispiel: `git checkout feature-branch`
- `git merge`: Fügt Änderungen von einem Branch in einen anderen zusammen.
    - Beispiel: `git merge feature-branch` mergt `feature-branch` in die aktuell ausgewählte Branch.


## Befehle für die Arbeit mit `dotnet`

Mit der Konsole können Sie neue **Projekte anlegen und konfigurieren**. Manchmal geht das schneller als im VS:

- `dotnet new`: Erstellt ein neues Projekt, eine Konfigurationsdatei oder eine Lösungsdatei.
    - Beispiel: `dotnet new console -n MyApp`
- `dotnet build`: Baut ein Projekt und alle seine Abhängigkeiten.
    - Beispiel: `dotnet build`
- `dotnet run`: Baut und führt ein .NET-Projekt aus.
    - Beispiel: `dotnet run`
- `dotnet test`: Führt Unit-Tests mit dem Test-Runner aus.
    - Beispiel: `dotnet test`
- `dotnet publish`: Veröffentlicht ein .NET-Projekt für den Einsatz auf einem Host-System.
    - Beispiel: `dotnet publish --configuration Release`
- `dotnet add package`: Fügt ein NuGet-Paket zu einem Projekt hinzu.
    - Beispiel: `dotnet add package Newtonsoft.Json`
- `dotnet remove package`: Entfernt ein NuGet-Paket von einem Projekt.
    - Beispiel: `dotnet remove package Newtonsoft.Json`
- `dotnet restore`: Stellt die Abhängigkeiten und Tools eines Projekts wieder her.
    - Beispiel: `dotnet restore`
- `dotnet clean`: Reinigt die Ausgabe eines Projekts.
    - Beispiel: `dotnet clean`

Ab und zu kommt man in Situationen, bei denen VS sich sperrt, die Solution korrekt anzulegen (weil man z.B. bestehenden Code aus einer externen Quelle integrieren muss). Dann helfen vielleicht folgende Befehle zur **Manipulation von Solutions**:

- `dotnet new sln`: Erstellt eine neue Solution-Datei.
    - Beispiel: `dotnet new sln -n MySolution`
- `dotnet sln add`: Fügt ein Projekt zu einer Solution hinzu.
    - Beispiel: `dotnet sln MySolution.sln add ./MyApp/MyApp.csproj`
- `dotnet sln remove`: Entfernt ein Projekt aus einer Solution.
    - Beispiel: `dotnet sln MySolution.sln remove ./MyApp/MyApp.csproj`
- `dotnet sln list`: Listet alle Projekte in einer Solution auf.
    - Beispiel: `dotnet sln MySolution.sln list`

Vielleicht hilft auch diese erweiterte Erklärung zum [Umgang mit Projekten und Solutions](https://learn.microsoft.com/en-us/visualstudio/ide/creating-solutions-and-projects).

## Python-Entwicklung mit Conda Environments

Conda environments erleichtern Ihnen die Arbeit mit Python massiv. Visual Studio Code und Jupyter Notebook sind je nach Version manchmal etwas sperrig. Die Konsole erleichtert Ihnen die Arbeit und macht den Zustand Ihrer Environments transparent.

- `conda create`: Erstellt eine neue Conda-Umgebung.
    - Beispiel: `conda create --name myenv`
    - - Beispiel: `conda create --name myenv python=3.9`
- `conda activate`: Aktiviert eine Conda-Umgebung.
    - Beispiel: `conda activate myenv`
- `conda deactivate`: Deaktiviert die aktuelle Conda-Umgebung.
    - Beispiel: `conda deactivate`
- `conda env list`: Listet alle Conda-Umgebungen auf.
- `conda list`: Listet alle Pakete in der aktuellen Umgebung auf.
    - Beispiel: `conda list`
- `conda install`: Installiert ein Paket in die aktuelle Umgebung.
    - Beispiel: `conda install numpy`
- `conda update`: Aktualisiert ein Paket in der aktuellen Umgebung.
    - Beispiel: `conda update numpy`
- `conda remove`: Entfernt ein Paket aus der aktuellen Umgebung.
    - Beispiel: `conda remove numpy`
- `conda search`: Sucht nach einem Paket in den Conda-Paketkanälen.
    - Beispiel: `conda search numpy`
- `conda info`: Zeigt Informationen über Conda.
    - Beispiel: `conda info`
