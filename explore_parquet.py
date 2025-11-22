
import os
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print("=" * 80)
    print("KLAIDA: Trūksta reikalingų bibliotekų!")
    print("=" * 80)
    print("\nPrašome įdiegti:")
    print("  pip install pandas pyarrow matplotlib seaborn")
    print("\nArba:")
    print("  pip install -r requirements.txt")
    print("\nKlaida:", e)
    sys.exit(1)

# Nustatome stilių
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def explore_parquet_file(file_path: str, output_dir: str = 'parquet_analysis'):
    """
    Atidaro .parquet failą, parodo informaciją ir sukuria vizualizacijas
    
    Args:
        file_path: Kelias į .parquet failą
        output_dir: Aplankas rezultatams
    """
    print("=" * 80)
    print(f"Analizuojamas failas: {file_path}")
    print("=" * 80)
    
    # Patikriname ar failas egzistuoja
    if not os.path.exists(file_path):
        print(f"✗ Klaida: Failas {file_path} nerastas!")
        return
    
    # Sukuriame išvesties aplanką
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Užkrauname duomenis
    print("\n[1/5] Užkraunami duomenys...")
    try:
        df = pd.read_parquet(file_path)
        print(f"✓ Duomenys užkrauti: {len(df)} eilučių, {len(df.columns)} stulpelių")
    except Exception as e:
        print(f"✗ Klaida užkraunant duomenis: {e}")
        return
    
    # 2. Pagrindinė informacija
    print("\n[2/5] Pagrindinė informacija apie duomenis...")
    
    print("\n--- Duomenų forma ---")
    print(f"Eilučių skaičius: {len(df)}")
    print(f"Stulpelių skaičius: {len(df.columns)}")
    print(f"Duomenų tipas: {df.dtypes.value_counts().to_dict()}")
    
    print("\n--- Stulpelių sąrašas ---")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"{i:2d}. {col:30s} | Tipas: {str(dtype):15s} | Null: {null_count:6d} ({null_pct:5.2f}%)")
    
    print("\n--- Pirmos 10 eilučių ---")
    print(df.head(10))
    
    print("\n--- Paskutinės 5 eilutės ---")
    print(df.tail(5))
    
    print("\n--- Statistinė informacija (skaitiniai stulpeliai) ---")
    # Randame timestamp stulpelius prieš numeric_cols apibrėžimą
    timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'time' in col.lower() or 'date' in col.lower()]
    # Pašaliname timestamp iš numeric_cols, jei jis ten yra
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                    if col not in timestamp_cols]
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("Nėra skaitinių stulpelių")
    
    # Išsaugome informaciją į failą
    info_file = os.path.join(output_dir, 'data_info.txt')
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"DUOMENŲ INFORMACIJA: {os.path.basename(file_path)}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Eilučių skaičius: {len(df)}\n")
        f.write(f"Stulpelių skaičius: {len(df.columns)}\n\n")
        f.write("Stulpeliai:\n")
        for col in df.columns:
            f.write(f"  - {col}: {df[col].dtype}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Statistinė informacija:\n")
        f.write("=" * 80 + "\n")
        f.write(df.describe().to_string())
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("Pirmos eilutės:\n")
        f.write("=" * 80 + "\n")
        f.write(df.head(20).to_string())
    print(f"  ✓ Informacija išsaugota į {info_file}")
    
    # 3. Timestamp analizė (jei yra timestamp stulpelis)
    print("\n[3/5] Laiko serijų analizė...")
    
    # Jei timestamp_cols dar neapibrėžtas, apibrėžiame
    if 'timestamp_cols' not in locals():
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'time' in col.lower() or 'date' in col.lower()]
    
    if timestamp_cols:
        for ts_col in timestamp_cols:
            print(f"\nAnalizuojamas: {ts_col}")
            try:
                # Sukuriame kopiją timestamp stulpelio konvertavimui
                ts_copy = df[ts_col].copy()
                
                if not pd.api.types.is_datetime64_any_dtype(ts_copy):
                    # Jei int64, galbūt tai milisekundės arba sekundės
                    if ts_copy.dtype == 'int64':
                        # Tikriname ar tai milisekundės (13 skaitmenų)
                        max_val = ts_copy.max()
                        if max_val > 1e12:
                            # Milisekundės
                            ts_copy = pd.to_datetime(ts_copy, unit='ms', errors='coerce')
                        elif max_val > 1e9:
                            # Sekundės
                            ts_copy = pd.to_datetime(ts_copy, unit='s', errors='coerce')
                        else:
                            ts_copy = pd.to_datetime(ts_copy, errors='coerce')
                    else:
                        ts_copy = pd.to_datetime(ts_copy, errors='coerce')
                
                # Atnaujiname originalų DataFrame
                df[ts_col] = ts_copy
                
                print(f"  Pirmas įrašas: {df[ts_col].min()}")
                print(f"  Paskutinis įrašas: {df[ts_col].max()}")
                print(f"  Trukmė: {df[ts_col].max() - df[ts_col].min()}")
                print(f"  Vidutinis intervalas: {df[ts_col].diff().mean()}")
                
                # Vizualizacija - duomenų pasiskirstymas pagal laiką
                fig, axes = plt.subplots(2, 1, figsize=(14, 8))
                
                # Histograma - konvertuojame į numerinę reikšmę
                df_numeric = pd.to_numeric(df[ts_col], errors='coerce')
                df_numeric = df_numeric.dropna()
                if len(df_numeric) > 0:
                    axes[0].hist(df_numeric.values, bins=50)
                    axes[0].set_title(f'{ts_col} - Duomenų pasiskirstymas', fontsize=12, fontweight='bold')
                    axes[0].set_xlabel('Data/Laikas (numerinis)')
                    axes[0].set_ylabel('Dažnis')
                    axes[0].tick_params(axis='x', rotation=45)
                else:
                    axes[0].text(0.5, 0.5, 'Nėra duomenų vizualizacijai', 
                                ha='center', va='center', transform=axes[0].transAxes)
                
                # Laiko serija (kiek įrašų per laiką)
                try:
                    # Bandoma grupuoti pagal valandas
                    hourly_counts = df.groupby(df[ts_col].dt.floor('h')).size()
                    axes[1].plot(hourly_counts.index, hourly_counts.values, linewidth=1.5)
                    axes[1].set_title(f'{ts_col} - Įrašų skaičius per valandą', fontsize=12, fontweight='bold')
                    axes[1].set_xlabel('Data/Laikas')
                    axes[1].set_ylabel('Įrašų skaičius')
                    axes[1].tick_params(axis='x', rotation=45)
                    axes[1].grid(True, alpha=0.3)
                except:
                    # Jei nepavyko, tiesiog parodome dažnių skaičių
                    axes[1].text(0.5, 0.5, f'Įrašų skaičius: {len(df)}', 
                                ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
                
                plt.tight_layout()
                timestamp_plot_file = os.path.join(output_dir, f'timestamp_{ts_col}_analysis.png')
                plt.savefig(timestamp_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Vizualizacija išsaugota į {timestamp_plot_file}")
                
            except Exception as e:
                print(f"  ⚠ Klaida analizuojant {ts_col}: {e}")
    
    # 4. Skaitinių stulpelių vizualizacija
    print("\n[4/5] Skaitinių stulpelių vizualizacija...")
    
    if len(numeric_cols) > 0:
        # Pasirenkame svarbiausius stulpelius (jei per daug)
        cols_to_plot = numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
        
        # Histogramos
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx] if len(cols_to_plot) > 1 else axes[0]
            
            # Patikriname ar stulpelis gali būti histogramuojamas
            try:
                # Pašaliname NaN
                col_data = df[col].dropna()
                
                if len(col_data) == 0:
                    ax.text(0.5, 0.5, 'Nėra duomenų', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(f'{col}\n(Nėra duomenų)', fontsize=10)
                    continue
                
                # Tikriname ar yra skaitinių duomenų
                if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_datetime64_any_dtype(col_data):
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    
                    col_data.hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
                    ax.set_title(f'{col}\n(mean={mean_val:.2f}, std={std_val:.2f})', 
                               fontsize=10, fontweight='bold')
                    ax.set_xlabel('Reikšmė')
                    ax.set_ylabel('Dažnis')
                    ax.grid(True, alpha=0.3, axis='y')
                else:
                    # Jei ne skaitinis, parodome unikalių reikšmių dažnį
                    value_counts = col_data.value_counts().head(20)
                    value_counts.plot(kind='bar', ax=ax)
                    ax.set_title(f'{col}\n(Unikalios reikšmės)', fontsize=10, fontweight='bold')
                    ax.set_xlabel('Reikšmė')
                    ax.set_ylabel('Dažnis')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3, axis='y')
            except Exception as e:
                ax.text(0.5, 0.5, f'Klaida:\n{str(e)[:50]}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8)
                ax.set_title(f'{col}\n(Klaida)', fontsize=10)
        
        # Paslėpiame tuščius subplotus
        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        hist_file = os.path.join(output_dir, 'numeric_histograms.png')
        plt.savefig(hist_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Histogramos išsaugotos į {hist_file}")
        
        # Koreliacijų matrica (jei yra daugiau nei 1 skaitinis stulpelis)
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(max(12, len(numeric_cols)), max(10, len(numeric_cols))))
            
            # Riboje koreliacijų matricą, jei per didelė
            if len(numeric_cols) > 15:
                # Pasirenkame tik didžiausias koreliacijas
                top_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        top_corr_pairs.append((
                            corr_matrix.iloc[i, j],
                            corr_matrix.columns[i],
                            corr_matrix.columns[j]
                        ))
                top_corr_pairs.sort(key=lambda x: abs(x[0]), reverse=True)
                top_cols = set()
                for _, col1, col2 in top_corr_pairs[:20]:
                    top_cols.add(col1)
                    top_cols.add(col2)
                corr_matrix = corr_matrix.loc[list(top_cols), list(top_cols)]
            
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={'label': 'Koreliacija'}, ax=ax)
            ax.set_title('Skaitinių Stulpelių Koreliacijų Matrica', fontsize=14, fontweight='bold')
            plt.tight_layout()
            corr_file = os.path.join(output_dir, 'correlation_matrix.png')
            plt.savefig(corr_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Koreliacijų matrica išsaugota į {corr_file}")
        
        # Laiko serijos (jei yra timestamp)
        if timestamp_cols and len(numeric_cols) > 0:
            ts_col = timestamp_cols[0]
            try:
                if pd.api.types.is_datetime64_any_dtype(df[ts_col]):
                    # Pasirenkame pirmus kelis skaitinius stulpelius
                    cols_to_plot_ts = numeric_cols[:5]
                    
                    fig, axes = plt.subplots(len(cols_to_plot_ts), 1, figsize=(14, 4*len(cols_to_plot_ts)))
                    if len(cols_to_plot_ts) == 1:
                        axes = [axes]
                    
                    for idx, col in enumerate(cols_to_plot_ts):
                        # Agreguojame pagal valandas
                        try:
                            df_agg = df.groupby(df[ts_col].dt.floor('h'))[col].mean()
                            
                            axes[idx].plot(df_agg.index, df_agg.values, linewidth=1.5)
                            axes[idx].set_title(f'{col} per laiką (valandų vidurkiai)', 
                                              fontsize=12, fontweight='bold')
                            axes[idx].set_xlabel('Data/Laikas')
                            axes[idx].set_ylabel(col)
                            axes[idx].tick_params(axis='x', rotation=45)
                            axes[idx].grid(True, alpha=0.3)
                        except Exception as e:
                            axes[idx].text(0.5, 0.5, f'Klaida kurdant {col}', 
                                          ha='center', va='center', 
                                          transform=axes[idx].transAxes)
                    
                    plt.tight_layout()
                    ts_file = os.path.join(output_dir, 'time_series.png')
                    plt.savefig(ts_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  ✓ Laiko serijos išsaugotos į {ts_file}")
            except Exception as e:
                print(f"  ⚠ Nepavyko sukurti laiko serijų: {e}")
    
    # 5. Unikalūs įrašai ir kiti įdomūs faktai
    print("\n[5/5] Papildoma analizė...")
    
    print("\n--- Unikalios reikšmės ---")
    for col in df.columns[:10]:  # Pirmi 10 stulpelių
        unique_count = df[col].nunique()
        unique_pct = (unique_count / len(df)) * 100
        print(f"{col}: {unique_count} unikalių reikšmių ({unique_pct:.2f}%)")
        if unique_count <= 20 and unique_count > 0:
            print(f"  Reikšmės: {sorted(df[col].unique())[:20]}")
    
    # Išsaugome visą DataFrame į CSV
    print("\n[6/6] Eksportuojami duomenys į CSV...")
    
    csv_file = os.path.join(output_dir, 'data_export.csv')
    
    try:
        # Išsaugome visą DataFrame
        print(f"  Išsaugomas CSV failas: {csv_file}")
        print(f"  Eilučių: {len(df)}, Stulpelių: {len(df.columns)}")
        
        # Konvertuojame timestamp stulpelius į skaitomus formatus prieš eksportą
        df_export = df.copy()
        for ts_col in timestamp_cols:
            if ts_col in df_export.columns:
                if not pd.api.types.is_datetime64_any_dtype(df_export[ts_col]):
                    # Jei int64, konvertuojame
                    if df_export[ts_col].dtype == 'int64':
                        max_val = df_export[ts_col].max()
                        if max_val > 1e12:
                            df_export[ts_col] = pd.to_datetime(df_export[ts_col], unit='ms', errors='coerce')
                        elif max_val > 1e9:
                            df_export[ts_col] = pd.to_datetime(df_export[ts_col], unit='s', errors='coerce')
                # Formatavimas kaip string CSV formatui
                if pd.api.types.is_datetime64_any_dtype(df_export[ts_col]):
                    df_export[ts_col] = df_export[ts_col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Jei per daug duomenų, galime naudoti chunking, bet pabandysime išsaugoti viską
        df_export.to_csv(csv_file, index=False)
        
        # Patikriname failo dydį
        file_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
        print(f"\n✓ Visi duomenys eksportuoti į CSV: {csv_file}")
        print(f"  Failo dydis: {file_size_mb:.2f} MB")
        
        # Taip pat išsaugome tik pavyzdį (pirmas 1000 eilučių) lengvam peržiūrėjimui
        sample_file = os.path.join(output_dir, 'data_sample_1000.csv')
        df_export.head(1000).to_csv(sample_file, index=False)
        print(f"✓ Pavyzdys (1000 eilučių) išsaugotas į: {sample_file}")
        
    except Exception as e:
        print(f"  ⚠ Klaida eksportuojant CSV: {e}")
        # Bandoma su chunking
        try:
            print("  Bandoma išsaugoti po dalimis...")
            chunk_size = 50000
            chunks = []
            for i in range(0, len(df), chunk_size):
                chunks.append(df.iloc[i:i+chunk_size])
            
            # Sujungiame ir išsaugome
            df_export_chunks = pd.concat(chunks)
            df_export_chunks.to_csv(csv_file, index=False)
            print(f"  ✓ Duomenys išsaugoti po dalimis į: {csv_file}")
        except Exception as e2:
            print(f"  ✗ Nepavyko išsaugoti: {e2}")
            # Išsaugome tik pavyzdį
            try:
                sample_file = os.path.join(output_dir, 'data_sample.csv')
                sample_size = min(10000, len(df))
                df.sample(sample_size).to_csv(sample_file, index=False)
                print(f"  ✓ Išsaugotas pavyzdys ({sample_size} eilučių) į: {sample_file}")
            except:
                print(f"  ✗ Nepavyko išsaugoti net pavyzdžio")
    
    print("\n" + "=" * 80)
    print("Analizė baigta!")
    print("=" * 80)
    print(f"\nVisi rezultatai išsaugoti į: {output_dir}/")
    print(f"  - data_info.txt - Tekstinė informacija")
    print(f"  - *.png - Vizualizacijos")
    print(f"  - *.csv - Eksportuoti duomenys")


def main():
    """Pagrindinė funkcija"""
    # Patikriname ar pateiktas failas
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Ieškome .parquet failų dabartiniame aplanke
        parquet_files = list(Path('.').glob('*.parquet'))
        if parquet_files:
            file_path = str(parquet_files[0])
            print(f"Rastas .parquet failas: {file_path}")
        else:
            print("Naudojimas: python explore_parquet.py <failo_kelias>")
            print("Arba patalpinkite .parquet failą į tą patį aplanką")
            return
    
    explore_parquet_file(file_path)


if __name__ == "__main__":
    main()

