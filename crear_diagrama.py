import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def crear_diagrama():
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Cajas principales
    entorno = patches.Rectangle((0.1, 0.4), 0.2, 0.2, facecolor='#e6f2ff', edgecolor='black', linewidth=2)
    som = patches.Rectangle((0.4, 0.6), 0.2, 0.2, facecolor='#e6ffe6', edgecolor='black', linewidth=2)
    dqn = patches.Rectangle((0.7, 0.4), 0.2, 0.2, facecolor='#ffe6e6', edgecolor='black', linewidth=2)
    memoria = patches.Rectangle((0.4, 0.2), 0.2, 0.2, facecolor='#fff2e6', edgecolor='black', linewidth=2)

    ax.add_patch(entorno)
    ax.add_patch(som)
    ax.add_patch(dqn)
    ax.add_patch(memoria)

    # Textos de las cajas
    ax.text(0.2, 0.5, 'Entorno\n(Gymnasium)\nDron 10x10', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.7, 'Codificador\nBioinspirado\n(SOM 20x20)', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.8, 0.5, 'Agente RL\n(DQN)', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.3, 'Replay Buffer\n(Memoria)', ha='center', va='center', fontsize=12, fontweight='bold')

    # Flechas y flujo de datos
    ax.annotate('', xy=(0.4, 0.7), xytext=(0.3, 0.5), arrowprops=dict(arrowstyle="->", lw=2, connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    ax.text(0.3, 0.65, 'Observación (7 floats)', ha='right', fontsize=10)

    ax.annotate('', xy=(0.7, 0.55), xytext=(0.6, 0.7), arrowprops=dict(arrowstyle="->", lw=2, connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    ax.text(0.7, 0.65, 'Índice BMU\n(Estado discreto)', ha='left', fontsize=10)

    ax.annotate('', xy=(0.1, 0.45), xytext=(0.8, 0.4), arrowprops=dict(arrowstyle="->", lw=2, connectionstyle="bar,fraction=-0.2"))
    ax.text(0.5, 0.15, 'Acción seleccionada (0-5)', ha='center', fontsize=10)

    ax.annotate('', xy=(0.7, 0.45), xytext=(0.3, 0.45), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.5, 0.48, 'Recompensa ($R_t$)', ha='center', fontsize=10)
    
    ax.annotate('', xy=(0.5, 0.4), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle="<->", lw=2))
    ax.text(0.52, 0.45, 'Transiciones\n$(S, A, R, S_{next})$', ha='left', fontsize=9)

    plt.title("Diagrama de Arquitectura: Acoplamiento Híbrido SOM + DQN", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("figures/diagrama_arquitectura.pdf", bbox_inches='tight')
    plt.close()
    print("Diagrama guardado en figures/diagrama_arquitectura.pdf")

if __name__ == "__main__":
    crear_diagrama()