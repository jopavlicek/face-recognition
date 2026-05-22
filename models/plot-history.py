import os
import json
import matplotlib.pyplot as plt

# Skript se podívá, kde reálně leží, a od toho odvodí správnou cestu k souborům
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_HISTORY_PATH = os.path.join(BASE_DIR, "custom-history.json")
MOBILENET_HISTORY_PATH = os.path.join(BASE_DIR, "mobilenet-history.json")

def load_history(path):
    """Bezpečně načte historii z JSON souboru, pokud existuje."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def main():
    # Načtení dat
    custom_hist = load_history(CUSTOM_HISTORY_PATH)
    mobilenet_hist = load_history(MOBILENET_HISTORY_PATH)
    
    if not custom_hist and not mobilenet_hist:
        print(f"Chyba: Nenašel jsem ani jeden JSON soubor s historií.")
        print(f"Hledal jsem v:\n  - {CUSTOM_HISTORY_PATH}\n  - {MOBILENET_HISTORY_PATH}")
        return

    # Nastavení vzhledu grafů
    plt.figure(figsize=(16, 7))
    
    # =========================================================================
    # LEVÝ GRAF: LOSS (CHYBOVOST)
    # =========================================================================
    plt.subplot(1, 2, 1)
    
    if custom_hist:
        epochs = range(1, len(custom_hist["loss"]) + 1)
        plt.plot(epochs, custom_hist["loss"], '--', color="#1f77b4", label="Custom: Trénovací Loss", alpha=0.5)
        plt.plot(epochs, custom_hist["val_loss"], '-', color="#1f77b4", label="Custom: Validační Loss", linewidth=2)
        
    if mobilenet_hist:
        epochs_mb = range(1, len(mobilenet_hist["loss"]) + 1)
        plt.plot(epochs_mb, mobilenet_hist["loss"], '--', color="#ff7f0e", label="MobileNet: Trénovací Loss", alpha=0.5)
        plt.plot(epochs_mb, mobilenet_hist["val_loss"], '-', color="#ff7f0e", label="MobileNet: Validační Loss", linewidth=2)
        
    plt.title("Srovnání chybovosti (Loss Profile)", fontsize=14, fontweight='bold')
    plt.xlabel("Epocha", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    # =========================================================================
    # PRAVÝ GRAF: ACCURACY (PŘESNOST)
    # =========================================================================
    plt.subplot(1, 2, 2)
    
    if custom_hist:
        epochs = range(1, len(custom_hist["accuracy"]) + 1)
        plt.plot(epochs, custom_hist["accuracy"], '--', color="#2ca02c", label="Custom: Trénovací Acc", alpha=0.5)
        plt.plot(epochs, custom_hist["val_accuracy"], '-', color="#2ca02c", label="Custom: Validační Acc", linewidth=2)
        
        # Zvýraznění nejlepšího bodu u Custom modelu
        best_val_acc = max(custom_hist["val_accuracy"])
        best_epoch = custom_hist["val_accuracy"].index(best_val_acc) + 1
        plt.scatter(best_epoch, best_val_acc, color="#d62728", s=100, zorder=5)
        plt.annotate(f"Custom Max: {best_val_acc:.2%}", 
                     xy=(best_epoch, best_val_acc), 
                     xytext=(best_epoch, best_val_acc - 0.04),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                     ha='center')
        
    if mobilenet_hist:
        epochs_mb = range(1, len(mobilenet_hist["accuracy"]) + 1)
        plt.plot(epochs_mb, mobilenet_hist["accuracy"], '--', color="#9467bd", label="MobileNet: Trénovací Acc", alpha=0.5)
        plt.plot(epochs_mb, mobilenet_hist["val_accuracy"], '-', color="#9467bd", label="MobileNet: Validační Acc", linewidth=2)
        
        # Zvýraznění nejlepšího bodu u MobileNetu
        best_val_acc_mb = max(mobilenet_hist["val_accuracy"])
        best_epoch_mb = mobilenet_hist["val_accuracy"].index(best_val_acc_mb) + 1
        plt.scatter(best_epoch_mb, best_val_acc_mb, color="#d62728", s=100, zorder=5)
        plt.annotate(f"MobileNet Max: {best_val_acc_mb:.2%}", 
                     xy=(best_epoch_mb, best_val_acc_mb), 
                     xytext=(best_epoch_mb, best_val_acc_mb + 0.03),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                     ha='center')
        
    plt.title("Srovnání přesnosti (Accuracy Profile)", fontsize=14, fontweight='bold')
    plt.xlabel("Epocha", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) # Procenta na ose Y
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Zobrazení
    plt.tight_layout()
    print("Vykresluji grafy...")
    plt.show()

if __name__ == "__main__":
    main()