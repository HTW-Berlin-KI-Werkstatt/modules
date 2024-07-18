---
title: "Retrieval Augement Generation"
layout: single
author_profile: true
author: Christina Kratsch
lecture_name: "Data Science"
lecture_desc: "Grundlegende Methoden für die Exploration und das Management von Daten."
licence: "CC-BY"
licence_desc: 2024 | HTW Berlin 
classes: wide
---


## Die Idee hinter RAG

Der Umgang mit Large Language Models (LLMs) wie ChatGPT ist Ihnen sicherlich aus dem Alltagsgebrauch bekannt. Diese Modelle stellen aktuell Anwendungsfälle für die größten weltweit bekannten neuronalen Netze dar. Derzeit wird sehr viel Forschung betrieben, um die Qualität des Outputs dieser Netze zu verbessern.

Ein wesentlicher Nachteil von LLMs ist, dass sie zeitlich fixiert sind. Da ihre Parameter (wie bei jedem neuronalen Netz) nach dem Training unveränderlich sind, verfügt ein LLM nur über Weltwissen bis zum Zeitpunkt des Trainings. Eine einfache Methode, um damit umzugehen, ist **Retrieval Augmented Generation** (RAG)-

Retrieval Augmented Generation (RAG) ist ein fortschrittlicher Ansatz im Bereich der künstlichen Intelligenz, der darauf abzielt, die Qualität und Relevanz von generierten Texten zu verbessern. Diese Methode kombiniert zwei wichtige Techniken: Information Retrieval (IR) und Textgenerierung (Generation).

Die Idee ist einfach:

![rag](img/rag.png)

Information Retrieval bezieht sich auf den Prozess des Abrufens relevanter Informationen aus großen Datensätzen. Im Kontext von RAG werden große Wissensdatenbanken oder Informationsquellen verwendet, um relevante Informationen für eine gegebene Eingabeabfrage zu finden (siehe Grafik, zweiter Schritt). Dies können beispielsweise Wikipedia-Artikel, Fachbücher oder Online-Datenbanken oder eben Wissensquellen im unternehmerischen Kontext sein. 

Der (1) einfache Query eines Nutzers soll durch RAG mit zusätzlichen Informationen angereichert (augmentiert) werden. Dafür werden diese von einem Orchestrieurungssystem (d.h. einer individuellen Softwarelösung) aus den verfügbaren Wissensquellen gesucht und ggf. zusammengefasst. Dies passiert mit herkömmlichen Methoden des Information Retrievals (2). Das LLM erhält stattdessen ein deutlich größeres, informationshaltigeres Prompt (3) und kann dadurch eine verbesserte Antwort generieren (4).

Die Vorteile dieser Methode sind vielfältig:
* das LLM erhält zusätzliche (aktuelle) Fakten für die Formulierung seiner Antworten
* Prompts an das LLM können mit zusätzlichen Informationen ausgestattet werden 
* spart Re-Training oder Finetuning des LLMs
* reduziert Halluzinationen
* ermöglicht dem LLM das Nennen von Quellen

## Implementierung

Wenig überraschend stellen Vektordatenbanken eine exzellente Lösung dar, um das IR-System (Schritt 2) zu unterstützen. Das Python-Paket `llama-index` nutzt eine eigene Implementierung von Vektordatenbanken. Die genaue Umsetzung können Sie in der [Dokumentation](https://docs.llamaindex.ai/en/stable/getting_started/concepts/) nachvollziehen. Grundsätzlich bietet die Library die Möglichkeit, Daten in verdauliche Informationshappen ("chunks") zu zerlegen und diese in einer Datenbank (index) abzulegen:

![RAG mit llama-index](https://docs.llamaindex.ai/en/stable/_static/getting_started/basic_rag.png)


Ein vollständiges RAG-System können Sie in wenigen Zeilen Code implementieren! Hier ist ein einfaches Starter Example aus den [llama docs](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/):
```python
# pip install llama-index
# export OPENAI_API_KEY=your key here
import os.path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What is the meaning of life?")
print(response)
```

Anmerkung: Stand Sommer 2024 nutzt `llama-index` default-mäßig das Modell `gpt-3.5-turbo` von OpenAI als LLM zur Antwortgenerierung und das Modell `text-embedding-ada-002` für Embedding und Retrieval. Das heißt insbesondere, dass Sie für die Arbeit mit `llama-index` in der Default-Einstellung einen (kostenpflichtigen) OpenAI API key benötigen. Alternativ können Sie in der Konfiguration aber auf Open Source Modelle wechseln. {: .notice--info}