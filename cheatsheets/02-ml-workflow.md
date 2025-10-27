# Cheat sheets and checklists

- Student: Gauthier Vandeputte
- GitHub repo: [URL](https://github.com/GauthierVdp/mlops-labs-main)

---


## Commands

|naar de juiste directory gaan|x|`cd resources/02-ml-workflow`|
|opstellen van een virtuele enviroment|x|`python -m venv venv`|
|opstarten van ded virtuele enviroment|x|`venv\Scripts\Activate.ps1`|
|installeren van alle dependencies|x|`pip install -r requirements.txt`|
|opstarten van de prefect server|x|`prefect server start`|
|installeren van mlflow|x|`pip install mlflow`|
|opstarten van de mlflow server|x|`python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ".\resources\02-ml-workflow\mlruns" --host 127.0.0.1 --port 5000`|
|trainen van het model|x|python .\resources\`02-ml-workflow\ml_workflow.py`|
|testen van het model|x|`python .\resources\02-ml-workflow\predict.py --image ".\resources\02-ml-workflow\dataset\test\oranges\orange9.jpeg"`|