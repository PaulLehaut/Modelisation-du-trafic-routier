# Utilisation des modèles ResNet

## Entraînement d’un modèle

Lancer `ResNet_training.py` en vérifiant les éléments suivants :

- Vérifier que vous avez les données synthétiques générées par Colleau (`bastien/datasets`)
- Vérifier le chemin `DATA_PATH`
- Choisir les hyperparamètres

---

## Visualisation des résultats

Lancer `ResNet_visualisation.py` en vérifiant :

- que vous utilisez les mêmes hyperparamètres que pour l’entraînement ;
- que les chemins sont corrects.

Les visualisations sont disponibles dans :

```text
experimentations/charles/ResNet_alpha/graphics
```

## Comparaison de plusieurs modèles ResNet

Lancer simplement `ResNet_comparison.py`. Ce script compare l'évolution
des loss d'entrainement de tous les modèles déjà entrainés
