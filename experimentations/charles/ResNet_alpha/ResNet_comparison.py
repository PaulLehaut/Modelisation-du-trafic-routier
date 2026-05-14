import os
import torch
import matplotlib.pyplot as plt

# =====================================================================
# CONFIGURATION
# =====================================================================
BASE_DIR = r"C:\Users\charl\OneDrive\Documents\PontsEtChaussees\2A\PROJET\code-projet-IMI\Modelisation-du-trafic-routier\experimentations\charles\ResNet_alpha"
RESULTS_DIR = os.path.join(BASE_DIR, "training_results")
GRAPHICS_DIR = os.path.join(BASE_DIR, "graphics")


def main():
    # Vérification de l'existence du dossier de résultats
    if not os.path.exists(RESULTS_DIR):
        raise FileNotFoundError(
            f"Le dossier contenant les résultats est introuvable:\n{RESULTS_DIR}"
        )

    os.makedirs(GRAPHICS_DIR, exist_ok=True)

    # Récupération de tous les fichiers dont le nom commence par "ResNet" et finit par ".pt"
    fichiers_modeles = [
        f
        for f in os.listdir(RESULTS_DIR)
        if f.startswith("ResNet") and f.endswith(".pt")
    ]

    if not fichiers_modeles:
        print(
            "Aucun fichier correspondant aux modèles ResNet n'a été trouvé dans le dossier."
        )
        return

    print(
        f"[{len(fichiers_modeles)}] fichiers trouvés. Génération du graphique comparatif..."
    )

    # Configuration du style visuel
    plt.style.use("seaborn-v0_8-darkgrid")

    # Utilisation des standards de taille du fichier de référence
    fig, ax = plt.subplots(figsize=(8, 5))

    # Parcours et extraction des données pour chaque fichier
    for fichier in fichiers_modeles:
        chemin_fichier = os.path.join(RESULTS_DIR, fichier)

        try:
            results = torch.load(chemin_fichier, weights_only=False)

            loss_history = results.get("loss_history")
            portion = results.get("PORTION_PROBE")
            lr = results.get("LEARNING_RATE")

            # Vérification que l'historique existe bien dans le dictionnaire
            if loss_history is None:
                print(
                    f"Attention: Aucune 'loss_history' trouvée dans {fichier}. Fichier ignoré."
                )
                continue

            # Construction du label pour la légende (Si les params manquent, on utilise le nom du fichier)
            if portion is not None and lr is not None:
                label = f"PV: {portion * 100:.0f}%, LR: {lr}"
            else:
                label = fichier.replace(".pt", "")

            # Tracé de la courbe
            ax.plot(loss_history, lw=1.5, alpha=0.8, label=label)

        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {fichier} : {e}")

    # Mise en forme du graphique selon tes standards
    ax.set_yscale("log")
    ax.set_title(
        "Comparaison de l'évolution des Loss (MSE) entre les modèles", fontsize=11
    )
    ax.set_xlabel("Epochs", fontsize=11)
    ax.set_ylabel("Erreur normalisée [Log]", fontsize=11)

    # Ajout de la légende et de la grille
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.3, 1))
    ax.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()

    # Sauvegarde
    save_path = os.path.join(GRAPHICS_DIR, "ResNet_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Graphique comparatif sauvegardé avec succès dans :\n{save_path}")


if __name__ == "__main__":
    main()
