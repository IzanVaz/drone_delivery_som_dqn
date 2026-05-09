import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def dibujar_mapa():
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Cuadrícula 10x10
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(0, 11, 1))
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    
    # Edificios (x, y) y su altura (z) extraídos de drone_env.py
    edificios = {
        (0, 4): 4, (1, 4): 5, (2, 4): 3, 
        (7, 4): 3, (8, 4): 5, (9, 4): 4,
        (7, 2): 3, (2, 6): 2
    }
    
    # Dibujar edificios
    for (x, y), h in edificios.items():
        rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='black', facecolor='slategray')
        ax.add_patch(rect)
        ax.text(x + 0.5, y + 0.5, f"h={h}", color='white', ha='center', va='center', fontweight='bold')
        
    # Dibujar Dron (Origen)
    dron = patches.Circle((5.5, 9.5), 0.3, linewidth=1, edgecolor='black', facecolor='cyan')
    ax.add_patch(dron)
    ax.text(5.5, 9.5, "S", color='black', ha='center', va='center', fontweight='bold')
    
    # Dibujar Meta (Destino)
    meta = patches.Rectangle((8.1, 1.1), 0.8, 0.8, linewidth=2, edgecolor='gold', facecolor='yellow', alpha=0.5)
    ax.add_patch(meta)
    ax.text(8.5, 1.5, "G", color='black', ha='center', va='center', fontweight='bold')
    
    # Viento
    ax.annotate('', xy=(1, 8.5), xytext=(4, 8.5), arrowprops=dict(facecolor='blue', shrink=0.05, width=2))
    ax.text(2.5, 8.8, "Viento", color='blue', ha='center')

    ax.set_aspect('equal')
    ax.set_title("Topología del Entorno (10x10)", fontsize=14, fontweight='bold')
    
    # Invertir el eje Y para que coincida con la vista de matriz clásica
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig("figures/mapa_entorno.pdf")
    print("-> Imagen del mapa generada en figures/mapa_entorno.pdf")

if __name__ == "__main__":
    dibujar_mapa()