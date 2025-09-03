import pandas as pd
import matplotlib.pyplot as plt

# Nome do arquivo de histórico
HISTORICO_CSV = "historico_metricas.csv"

def analisar_historico():
    # Carregar histórico
    try:
        df = pd.read_csv(HISTORICO_CSV)
    except FileNotFoundError:
        print(f"Arquivo {HISTORICO_CSV} não encontrado. Rode treino.py primeiro.")
        return

    # Converter timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("\n=== Histórico de Treinamentos ===")
    print(df.tail(10))  # Mostra os últimos 10 registros

    # Gráficos de evolução
    metricas = ["precision", "recall", "f1", "roc_auc"]
    for metrica in metricas:
        plt.figure(figsize=(10, 5))
        for modelo in df["modelo"].unique():
            df_modelo = df[df["modelo"] == modelo]
            plt.plot(df_modelo["timestamp"], df_modelo[metrica], marker="o", label=modelo)
        plt.title(f"Evolução da Métrica: {metrica.upper()}")
        plt.xlabel("Execuções no tempo")
        plt.ylabel(metrica.upper())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analisar_historico()
