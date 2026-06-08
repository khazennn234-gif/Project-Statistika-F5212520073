import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    print("Folder saat ini :", os.getcwd())

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "rea_games_dataset_updated.csv")

    print("Mencari file :", file_path)

    try:
        df = pd.read_csv(file_path)
        print("File berhasil dibaca!")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' tidak ditemukan.")
        return

    print("\nDaftar Kolom:")
    print(df.columns.tolist())

    # Filter wilayah Asia
    df_asia = df[df['Region'] == 'Asia'].head(100)

    if len(df_asia) == 0:
        print("Tidak ada data dengan Region = 'Asia'")
        return

    X = df_asia['Player Count (Million)'].values
    Y = df_asia['Sales (Million $)'].values

    n = len(X)
    sum_x = np.sum(X)
    sum_y = np.sum(Y)
    sum_xy = np.sum(X * Y)
    sum_x_square = np.sum(X**2)

    denominator = (n * sum_x_square - sum_x**2)

    if denominator == 0:
        print("Error: Tidak dapat menghitung regresi karena penyebut bernilai 0.")
        return

    b = (n * sum_xy - sum_x * sum_y) / denominator
    a = (sum_y - b * sum_x) / n

    print("\n" + "="*50)
    print(" HASIL ANALISIS REGRESI LINEAR SEDERHANA ")
    print("="*50)
    print(f"Jumlah Sampel Data (n)  : {n}")
    print(f"Nilai Konstanta (a)     : {a:.4f}")
    print(f"Nilai Koefisien (b)     : {b:.4f}")
    print(f"Persamaan Regresi       : Y = {a:.4f} + {b:.4f}X")
    print("="*50)
    
    idx = np.argsort(X)

    plt.figure(figsize=(9, 5.5))
    plt.scatter(X, Y, alpha=0.7, label='Data Riil Video Game (Asia)')
    plt.plot(
        X[idx],
        (a + b * X)[idx],
        linewidth=2,
        label=f'Garis Regresi: Y = {a:.2f} + {b:.2f}X'
    )

    plt.title('Analisis Regresi: Hubungan Player Count terhadap Total Sales di Asia')
    plt.xlabel('Player Count (Juta Pemain)')
    plt.ylabel('Sales (Juta Dollar $)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    output_chart = os.path.join(base_dir, 'hasil_regresi_game.png')
    plt.savefig(output_chart, dpi=300, bbox_inches='tight')

    print(f"\nGrafik berhasil disimpan:")
    print(output_chart)

    plt.show()

if __name__ == "__main__":
    main()