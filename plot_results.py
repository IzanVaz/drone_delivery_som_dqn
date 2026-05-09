import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

def load_data(directory, prefix, num_seeds=5):
    """Carga los datos de todas las semillas y devuelve una matriz (semillas, episodios)"""
    seeds_data = []
    # Buscamos los archivos específicos generados
    seeds = [42, 123, 7, 99, 1]
    
    for s in seeds:
        filepath = os.path.join(directory, f"{prefix}_seed_{s}.npz")
        if os.path.exists(filepath):
            data = np.load(filepath)['returns']
            seeds_data.append(data)
        else:
            print(f"Advertencia: No se encontró {filepath}")
            
    if not seeds_data:
        raise FileNotFoundError(f"No hay datos en {directory} con el prefijo {prefix}")
        
    return np.array(seeds_data)

def smooth_curve(data, window=100):
    """Aplica una media móvil para suavizar las gráficas"""
    smoothed = np.zeros_like(data, dtype=float)
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed[i] = np.mean(data[start:i+1])
    return smoothed

def main():
    # 1. Crear carpeta para las figuras
    os.makedirs("figures", exist_ok=True)
    print("Cargando datos experimentales...")

    try:
        # Cargar datos del Híbrido (SOM+DQN) y suavizarlos
        hybrid_data = load_data("results", "raw_data")
        hybrid_mean = np.mean(hybrid_data, axis=0)
        hybrid_std = np.std(hybrid_data, axis=0)
        hybrid_mean_smooth = smooth_curve(hybrid_mean)
        hybrid_std_smooth = smooth_curve(hybrid_std)

        # Cargar datos de la Ablación (Solo DQN) y suavizarlos
        baseline_data = load_data("results_ablacion", "baseline_data")
        baseline_mean = np.mean(baseline_data, axis=0)
        baseline_std = np.std(baseline_data, axis=0)
        baseline_mean_smooth = smooth_curve(baseline_mean)
        baseline_std_smooth = smooth_curve(baseline_std)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Asegúrate de haber ejecutado train.py y train_ablacion.py primero.")
        return

    # 2. Generar y guardar la gráfica principal
    print("Generando curva de aprendizaje...")
    episodes = np.arange(len(hybrid_mean))
    
    plt.figure(figsize=(10, 6))
    
    # Dibujar Híbrido
    plt.plot(episodes, hybrid_mean_smooth, label="Híbrido (SOM + DQN)", color="blue", linewidth=2)
    plt.fill_between(episodes, hybrid_mean_smooth - hybrid_std_smooth, hybrid_mean_smooth + hybrid_std_smooth, color="blue", alpha=0.2)
    
    # Dibujar Ablación
    plt.plot(episodes, baseline_mean_smooth, label="Línea Base (Solo DQN)", color="red", linewidth=2, linestyle="--")
    plt.fill_between(episodes, baseline_mean_smooth - baseline_std_smooth, baseline_mean_smooth + baseline_std_smooth, color="red", alpha=0.2)

    plt.title("Curva de Aprendizaje: Híbrido vs Línea Base", fontsize=14)
    plt.xlabel("Episodios", fontsize=12)
    plt.ylabel("Retorno Medio (Ventana = 100)", fontsize=12)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    
    plt.savefig("figures/learning_curve.pdf") # Se pide .pdf o .png en el boletín
    plt.savefig("figures/learning_curve.png")
    print("-> Gráfica guardada en 'figures/learning_curve.pdf'")

    # 3. Análisis Estadístico y Métricas (Último 10% de episodios = últimos 1000)
    print("\n" + "="*50)
    print(" RESULTADOS ESTADÍSTICOS PARA EL INFORME LATEX ")
    print("="*50)
    
    # Promedio del retorno de cada semilla en los últimos 1000 episodios
    hybrid_final_scores = np.mean(hybrid_data[:, -1000:], axis=1)
    baseline_final_scores = np.mean(baseline_data[:, -1000:], axis=1)

    r_hibrido = np.mean(hybrid_final_scores)
    r_base = np.mean(baseline_final_scores)
    
    print(f"Puntuación de Convergencia (Híbrido): {r_hibrido:.2f} ± {np.std(hybrid_final_scores):.2f}")
    print(f"Puntuación de Convergencia (Base):    {r_base:.2f} ± {np.std(baseline_final_scores):.2f}")

    # Mejora relativa (Delta)
    if r_base != 0:
        delta = (r_hibrido - r_base) / abs(r_base)
        print(f"Mejora Relativa (\u0394):               {delta * 100:.2f}%")
    
    # Prueba U de Mann-Whitney
    stat, p_value = mannwhitneyu(hybrid_final_scores, baseline_final_scores, alternative='two-sided')
    print(f"Prueba U de Mann-Whitney (p-value):   {p_value:.5f}")
    
    if p_value < 0.05:
        print("-> CONCLUSIÓN: La diferencia ES estadísticamente significativa (p < 0.05).")
    else:
        print("-> CONCLUSIÓN: La diferencia NO es estadísticamente significativa (p >= 0.05).")
    print("="*50)

if __name__ == "__main__":
    main()