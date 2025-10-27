# Lab Rapport: Machine Learning Workflow

## Student information

- Student naam: Gauthier Vandeputte
- Student code: 202397621

## Assignment description

In dit labo leren we een Simpele Machine Learning workflow optestellen met Prefect. We loggen metrics en artifects met MLFlow + het maken van voorspellingen met een gemaakt model.

## Proof of work done
| Taak | Bewijs / Screenshot van | Commando / Actie |
|------|------------------------|------------------|
|naar de juiste directory gaan|x|`cd resources/02-ml-workflow`|
|opstellen van een virtuele enviroment|x|`python -m venv venv`|
|opstarten van ded virtuele enviroment|x|`venv\Scripts\Activate.ps1`|
|installeren van lle dependencies|x|`pip install -r requirements.txt`|
|opstarten van de prefect server|x|`prefect server start`|
|installeren van mlflow|x|`pip install mlflow`|
|opstarten van de mlflow server|x|`python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ".\resources\02-ml-workflow\mlruns" --host 127.0.0.1 --port 5000`|
|trainen van het model|x|python .\resources\`02-ml-workflow\ml_workflow.py`|
|testen van het model|x|`python .\resources\02-ml-workflow\predict.py --image ".\resources\02-ml-workflow\dataset\test\oranges\orange9.jpeg"`|
||||

### vragen
### Lab Questions and Answers

# Answers written as a student (questions blijven in het Engels)

## 1. ** Why do we need to set the PREFECT_HOME environment variable?**
Het bepaalt waar Prefect zijn configuratie, metadata en databank opslaat. zo blijven prefect bestanden binnen in het project, wordt het hergebruikbaar in andere projecten en voorkom je conflicten met andere prefect instanties op dezelfde computer.

## 2. **Why do you need to use the registered model from MLFlow and not the model file directly? **
Omdat het losse modelbestand alleen de modelgewichten bezit maar niet de juiste omgeving, de exacte metadata en versiegeschiedenis. Een geregistreerd model in MLFlow beat al deze informatie wel.

## 3. **What's the purpose of the MLFlow Model Registry? **
Het is een centrale plaats om ML-modellen te beheren. je kan er je modellen in zien, de versies er van, de status, documentaties en nog veel meer.

### Conclusion:
Aan de hand van deze oefeningen hebben we geleerd om een Machine Learning workflow te maken met Prefect, deze modellen op MLFlow te stellen en kleine voorspellingen te maken met het gemaakte fruit model.

## Reflection

### What was difficult?
Ik had een groot probleem met het probleem te vinden waarom tensorflow en andere libraries te installeren. Ook een aantal bugfixes nodig voor de predictie en het trainen van het model gaf soms verkeerde outputs maar dat is opgelost.

### What was easy?
Het opstellen van de MLFlow ging redelijk vlot.

### What did I learn?
Ik heb vooral geleerd hoe je MLFlow moest gebruiken en dat het een handig tool is voor later.

### What would I do differently?
/