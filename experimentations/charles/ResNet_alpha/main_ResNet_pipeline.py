import argparse
import os
import re
import torch
import matplotlib.pyplot as plt

# Import de tes modules existants
# Assure-toi que ces fichiers sont bien dans le même dossier
import training
import visualisation
import compare_models


def extract_mode_from_path(path):
    """Extrait le mode (ex: 'rarefaction') du nom de fichier."""
    filename = os.path.basename(path)
    # Cherche le texte après 'N1000_' et avant '.csv'
    match = re.search(r"N\d+_(.+)\.csv", filename)
    return match.group(1) if match else "unknown"


def main():
    parser = argparse.ArgumentParser(description="Pipeline TrafficResNet")
    parser.add_argument(
        "action",
        choices=["train", "visualize", "compare", "all"],
        help="Action à réaliser",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=r"data\reconstruction_modele_imi\traffic_N1000_rarefaction.csv",
    )
    parser.add_argument("--portion", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.5)

    args = parser.parse_args()

    # Extraction du mode pour le nommage du fichier modèle
    mode = extract_mode_from_path(args.dataset)
    model_filename = f"ResNet_{mode}_probe{args.portion}_ep{args.epochs}_lr{args.lr}.pt"

    # 1. ENTRAÎNEMENT
    if args.action in ["train", "all"]:
        print(f"--- Lancement de l'entraînement : {model_filename} ---")
        training.run_training(
            csv_path=args.dataset,
            portion=args.portion,
            epochs=args.epochs,
            lr=args.lr,
            save_name=model_filename,
        )

    # 2. VISUALISATION
    if args.action in ["visualize", "all"]:
        print(f"--- Lancement de la visualisation : {model_filename} ---")
        visualisation.run_viz(model_filename)

    # 3. COMPARAISON
    if args.action in ["compare", "all"]:
        print("--- Lancement de la comparaison globale ---")
        compare_models.run_comparison()


if __name__ == "__main__":
    main()
