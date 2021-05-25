# Beispielprojekt

Dieses Projekt dient als Beispiel zum Aufsetzen eines [AI Overflow](https://ai-overflow.github.io/ai_playground/) 
basierten Projekts.

## Start

Zum Starten des Projekts ist der folgende Befehl ausreichend:

```bash
~/project_root $> docker compose up
```

**Achtung:** Zum Herunterfahren des Containers **muss** der Befehl `docker compose down` verwendet werden, da sonst 
ein Fehler auftritt, welcher ein Erneutes starten verhindert.

## Anpassungen

Wenn Sie kein Alpine verwenden wollen, so ist es möglich die erste Zeile in der Datei `Dockerfile` durch folgendes
zu ersetzen:

```Dockerfile
FROM tiangolo/uwsgi-nginx-flask:python3.8
# RUN apk --update add bash nano
RUN apk update
# ...
```
(tiangolo/uwsgi-nginx-flask:python3.8-**alpine** → tiangolo/uwsgi-nginx-flask:python3.8)

Dies kann notwendig sein, falls es zu Fehlern bei der installation von Python Packages kommt.