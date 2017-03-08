Zusatzmaterial fuer das Tutorial DataScience in Python
======================================================

Um beim Tutorial DataScience in Python mitzumachen muessen sie einfach dieses Repository
auschecken und sich eine passende Python Umgebung aufsetzten.

Vorbereitung Linux Systeme
--------------------------

Unter Linux ist es am einfachsten sich dazu eine *Virtualenv* einzurichten.
Dazu oeffnen man einfach einen Terminal und fuehrt folgende Befehle aus.

```
> virtualenv myenv
> source myenv/bin/activate
> pip install -r requirements.txt
> ipython notebook
```

Vorbereitung andere Betriebssysteme
-----------------------------------

Unter anderen Betriebssystem ist es am einfachsten sich mit Hilfe von anaconda eine Python
Umbgebung mit den noetigen Abhaengigkeiten einzurichten (dies ist prinzipiell auch unter Linux
moeglich).

* Als erstes muss dafuer die fuer das Betriebssystem korrekte Version von Anaconda von der Seite
  https://www.continuum.io/downloads heruntergeladen werden.

* Dann folgt man einfach dem Installer und installiert Anaconda (Erweiterungen wie AnacondaCload
  sind nicht noetig).

* Nun oeffnen wir den AnacondaNavigator und dort das Jupyter Notebook

* Jetzt muessen wir nur noch das richtige Notebook auswaehlen und schon kann es losgehen.