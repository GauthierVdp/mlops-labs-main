# Cheat sheets and checklists

- Student: Gauthier Vandeputte
- GitHub repo: URL

---


## Docker Commands

| Actie                                 | Command |
| :----------------------------------- | :------ |
| Toon Docker versie                   | `docker --version` |
| Toon Docker Compose versie           | `docker compose --version` |
| Build image                          | `docker build -t naam:tag .` |
| Toon images                          | `docker images` |
| Run container                        | `docker run -p HOST:CONTAINER naam:tag` |
| Run container met GPU (Triton)       | `docker run --gpus all -p 8000:8000 -v "PAD:/models" nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models` |
| Stop container                       | `docker stop CONTAINER` |
| Verwijder container                  | `docker rm CONTAINER` |
| Push naar Docker Hub                 | `docker push USER/NAAM:tag` |
| Pull image                           | `docker pull USER/NAAM:tag` |

---

## Docker Compose

| Actie                            | Command |
| :----------------------------- | :------ |
| Start services                 | `docker compose up` |
| Start in achtergrond (`detached`) | `docker compose up -d` |
| Toon draaiende services         | `docker compose ps` |
| Bekijk logs                     | `docker compose logs -f` |
| Stop en verwijder alles        | `docker compose down` |

---

## Triton Inference Server

| Actie                                      | Command |
| :---------------------------------------- | :------ |
| Check of model geladen is                 | `curl localhost:8000/v2/models/example_model` |
| Inference HTTP POST (PowerShell)          | ```powershell $body='{"inputs":[{"name":"keras_tensor","shape":[1,4],"datatype":"FP32","data":[[1.2,3.4,5.6,7.8]]}]}'```  
| Verstuur request                          | `Invoke-WebRequest http://localhost:8000/v2/models/example_model/infer -Method POST -ContentType "application/json" -Body $body` |
| Model repository tonen                    | `tree model_repository -L 3` |
| Config tonen                              | `cat model_repository/example_model/config.pbtxt` |

---


