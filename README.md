# Application de prédiction de la race d'un chien présent sur une image
Cette application a été codée dans le cadre de la formation ingénieur Machine Learning dispensée par OpenClassrooms en partenariat avec CentraleSupelec.
L'objectif était d'entraîner un modèle de deeplearning de classification d'images de chiens en fonction de leurs races. L'algorithme classifie 120 races de chiens différentes avec une précision supérieure à 90%.

## Pré-requis
 - Docker
 - Docker compose

## Lancer l'application
L'application utilise Docker et Docker Compose
Dans le terminal :
```sh
git clone https://github.com/JMFrmg/dog_breed_classifier.git
```
```sh
cd dog_breed_classifier
```
```sh
docker compose up -d --build
```

## Choix technologiques
 - Fastapi (backend)
 - Pytorch
 - Streamlit (frontend)

