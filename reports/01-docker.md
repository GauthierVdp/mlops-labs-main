# Lab Rapport: Docker en Triton MLOps

## Studentinformatie

- Student naam: 
- Student code: 

## Opdrachtbeschrijving

In dit labo was het de bedoeling om Docker effectief te leren gebruiken binnen een MLOps-context. De taken omvatten het installeren van Docker, het containerizen van een Flask-gebaseerd ML-model, het deployen van modellen met NVIDIA Triton Inference Server, het pushen en pullen van images naar een container registry en het orkestreren van services met Docker Compose.

## Bewijs van uitgevoerde taken

_Voeg hier screenshots, terminal output, codefragmenten en links naar repositories toe._

## ✅ Evaluatiecriteria + Bewijs

### Deel 1: Docker Basics in ML Context

| Taak | Bewijs / Screenshot van | Commando / Actie |
|------|------------------------|------------------|
| Docker en Docker Compose geïnstalleerd | Screenshot van terminalversie | `docker --version` + `docker compose --version` |
| Dockerfile aangemaakt voor Flask-model | Screenshot van Dockerfile inhoud | Toon inhoud met `cat Dockerfile` |
| Docker image gebouwd | Terminal output van succesvolle build | `docker build -t my-ml-app .` (start docker desktop op anders werkt dit niet) |
| Container gestart | `docker ps` met draaiende container zichtbaar | `docker run -p 5000:5000 my-ml-app` |
| Inference uitgevoerd via Flask endpoint | Screenshot Postman / curl output | `curl http://localhost:5000/predict -X POST -d '{"data": ...}'` |
| Image getagd en gepusht naar registry | Terminaloutput push en tag zichtbaar | `docker tag my-ml-app <username>/my-ml-app:v1` + `docker push <username>/my-ml-app:v1` |
| Image gepulled en aanwezigheid bevestigd | Screenshot van `docker images` na pull | `docker pull <username>/my-ml-app:v1` |

---

### Deel 2: Triton Serving

| Taak | Bewijs / Screenshot van | Commando / Actie |
|------|------------------------|------------------|
| Triton-container gestart met TensorFlow-model | `docker ps` met `nvcr.io/nvidia/tritonserver` zichtbaar | `docker run --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v ./models:/models nvcr.io/nvidia/tritonserver:xx.yy-py3 tritonserver --model-repository=/models` |
| Inference via Triton HTTP endpoint | Screenshot van curl output | `curl -v localhost:8000/v2/models/<model>/infer` |
| Publiek model gedownload en gebruikt | Screenshot van mapstructuur `models/<model>/1/` | `wget <model-url>` of GitHub clone |
| Structuur model repository + `config.pbtxt` toegelicht | Screenshot van `tree models` + inhoud `config.pbtxt` | `tree models -L 3` + `cat models/<model>/config.pbtxt` |

---

### Deel 3: Docker Compose

| Taak | Bewijs / Screenshot van | Commando / Actie |
|------|------------------------|------------------|
| `docker-compose.yml` aangemaakt | Toon bestand in rapport (codeblok) | `cat docker-compose.yml` |
| Services gestart via Docker Compose | Terminaloutput `docker compose up` en `docker compose ps` | `docker compose up` + `docker compose ps` |
| Logs bekeken | Screenshot met logoutput | `docker compose logs -f` |
| Services gestopt en opgeschoond | Terminaloutput stop/verwijdering | `docker compose down` + eventueel `docker system prune` |

---

### Algemeen

| Taak | Bewijs / Screenshot van | Actie |
|------|------------------------|-------|
| Markdown labrapport in repo | Screenshot GitHub commit | Laat `README.md` / `REPORT.md` zien op GitHub |
| Alle vragen beantwoord | Toon sectie met `:question:` antwoorden | Voeg sectie "Antwoorden op vragen" toe |
| Screenshots van alle belangrijke stappen | Screenshots gelabeld (bijv. `docker_build.png`, `triton_infer.png`) | Upload screenshots in `/docs/screenshots/` map |
| Command cheat sheet bijgewerkt | Toon tabel met alle gebruikte commando’s | Voeg sectie **"Command Cheat Sheet"** toe onderaan |

---

### 💡 Tip voor structuur in rapport

Je kan **per onderdeel** een blok maken zoals:

```md
#### Bewijs: Docker versie

```bash
$ docker --version
Docker version 27.0.1, build xxxxxxx


## Issues

_Beschrijf problemen die je tegenkwam en hoe je deze opgelost hebt. Indien geen problemen: schrijf "geen"._

## Reflectie

_Schrijf hier je reflectie: wat heb je geleerd, welke stappen waren uitdagend, wat zou je anders aanpakken in een volgend project._

## Resources

_Lijst enkel echte bronnen zoals Docker Docs, NVIDIA Triton documentatie, GitHub repositories, officiële tutorials. Geen AI-bronnen._
