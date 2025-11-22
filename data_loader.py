"""
Modulis duomenų parsisiuntimui iš Google Drive ir apdorojimui
"""

import os
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Google Drive API nustatymai
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def authenticate_google_drive():
    """
    Autentifikuojasi su Google Drive API
    Reikia credentials.json failo (gali būti gautas iš Google Cloud Console)
    """
    creds = None
    
    # Patikriname ar yra token.pickle su išsaugotais credentials
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # Jei nėra galiojančių credentials, prašome vartotojo prisijungti
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                print("ERROR: credentials.json failas nerastas!")
                print("Prašome sukurti credentials.json failą iš Google Cloud Console")
                print("Arba naudokite alternatyvų metodą - parsisiųskite duomenis rankiniu būdu")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Išsaugome credentials kitiems kartams
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)


def download_folder_files(service, folder_id: str, local_folder: str):
    """
    Parsisiunčia visus failus iš Google Drive folderio
    
    Args:
        service: Google Drive API serviso objektas
        folder_id: Folderio ID
        local_folder: Vietinis aplankas, kur išsaugoti failus
    """
    os.makedirs(local_folder, exist_ok=True)
    
    # Gauname visus failus iš folderio
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    items = results.get('files', [])
    
    if not items:
        print(f"  Įspėjimas: Folderis {folder_id} tuščias")
        return
    
    for item in items:
        file_id = item['id']
        file_name = item['name']
        mime_type = item.get('mimeType', '')
        
        # Jei tai kitas folderis, rekursiškai parsisiunčiame
        if 'folder' in mime_type.lower():
            subfolder_path = os.path.join(local_folder, file_name)
            download_folder_files(service, file_id, subfolder_path)
        else:
            # Parsisiunčiame failą
            file_path = os.path.join(local_folder, file_name)
            try:
                request = service.files().get_media(fileId=file_id)
                with open(file_path, 'wb') as f:
                    from io import BytesIO
                    import io
                    from googleapiclient.http import MediaIoBaseDownload
                    downloader = MediaIoBaseDownload(f, request)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
            except Exception as e:
                print(f"  Klaida parsisiunčiant {file_name}: {e}")


def load_crypto_data_from_local(data_dir: str, currency_pair: str) -> pd.DataFrame:
    """
    Užkrauna kriptovaliutos duomenis iš vietinių failų
    Struktūra: currency_pair/ -> dienos folderiai/ -> CSV failai su bid_wwap, ask_wwap, timestamp, local_timestamp
    
    Args:
        data_dir: Aplankas su duomenimis
        currency_pair: Valiutos poros pavadinimas (pvz., 'BTCUSDT')
    
    Returns:
        DataFrame su duomenimis (open, high, low, close, volume apskaičiuoti iš bid/ask)
    """
    folder_path = os.path.join(data_dir, currency_pair)
    
    if not os.path.exists(folder_path):
        print(f"  Įspėjimas: Aplankas {folder_path} nerastas")
        return pd.DataFrame()
    
    # Ieškome dienų folderių ir CSV failų
    all_data = []
    
    # Ignoruojame '1m daily' folderį pagal reikalavimus
    ignored_folders = ['1m daily', 'daily']
    
    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_name)
        
        # Patikriname ar tai ignoruotinas folderis
        if any(ignore in item_name.lower() for ignore in ignored_folders):
            continue
        
        # Jei tai folderis (diena), peržiūrime jo turinį
        if os.path.isdir(item_path):
            # Ieškome CSV failų dienos folderyje
            for file_name in os.listdir(item_path):
                file_path = os.path.join(item_path, file_name)
                
                if not file_name.endswith('.csv'):
                    continue
                
                try:
                    df = pd.read_csv(file_path)
                    
                    # Standartizuojame stulpelių pavadinimus
                    df.columns = df.columns.str.lower().str.strip()
                    
                    # Patikriname ar yra reikiami stulpeliai: bid_wwap, ask_wwap, timestamp
                    if 'bid_wwap' not in df.columns or 'ask_wwap' not in df.columns:
                        # Bandoma rasti alternatyvius pavadinimus
                        col_mapping = {}
                        for col in df.columns:
                            if 'bid' in col and ('wwap' in col or 'vwap' in col):
                                col_mapping[col] = 'bid_wwap'
                            elif 'ask' in col and ('wwap' in col or 'vwap' in col):
                                col_mapping[col] = 'ask_wwap'
                            elif 'timestamp' in col or 'time' in col:
                                if 'local' not in col.lower():
                                    col_mapping[col] = 'timestamp'
                    
                        if 'bid_wwap' in col_mapping or 'ask_wwap' in col_mapping:
                            df = df.rename(columns=col_mapping)
                        else:
                            print(f"  Įspėjimas: Nepavyko rasti bid_wwap/ask_wwap {file_path}")
                            continue
                    
                    # Patikriname ar yra timestamp
                    if 'timestamp' not in df.columns:
                        print(f"  Įspėjimas: Nerastas timestamp {file_path}")
                        continue
                    
                    # Apskaičiuojame mid price (vidutinė iš bid ir ask)
                    df['mid_price'] = (df['bid_wwap'] + df['ask_wwap']) / 2
                    
                    # Apskaičiuojame spread
                    df['spread'] = df['ask_wwap'] - df['bid_wwap']
                    
                    # Konvertuojame timestamp
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Rūšiuojame pagal timestamp
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    # Sukuriame OHLC struktūrą iš bid/ask duomenų
                    # Naudojame mid price kaip close
                    df['close'] = df['mid_price']
                    
                    # Grupuojame pagal laiko intervalus (pvz., minutės arba valandos) ir apskaičiuojame OHLC
                    # Jei duomenys jau yra agreguoti, naudojame tiesioginį metodą
                    # High = maksimalus ask_wwap, Low = minimalus bid_wwap per intervalą
                    
                    # Jei duomenys yra minutės lygyje, agreguojame į valandas arba dienas
                    # Bet pirmiausia patikrinkime, ar duomenys jau agreguoti
                    
                    # Naudojame rolling window metodą arba grupuojame pagal dienas
                    df['date'] = df['timestamp'].dt.date
                    
                    # Sukuriame OHLC pagal dienas
                    daily_ohlc = df.groupby('date').agg({
                        'ask_wwap': 'max',  # High = maksimalus ask
                        'bid_wwap': 'min',  # Low = minimalus bid
                        'mid_price': ['first', 'last']  # Open = pirmas mid_price, Close = paskutinis mid_price
                    }).reset_index()
                    
                    daily_ohlc.columns = ['date', 'high', 'low', 'open', 'close']
                    
                    # Apskaičiuojame volume (naudojame spread kaip proxy, arba 0 jei nėra)
                    daily_ohlc['volume'] = df.groupby('date')['spread'].sum()  # Spread kaip volume proxy
                    
                    # Sukuriame timestamp iš datų
                    daily_ohlc['timestamp'] = pd.to_datetime(daily_ohlc['date'])
                    
                    # Pridedame sesijos informaciją (valandos lygmenyje)
                    # Išsaugome originalius duomenis su timestamp ir mid_price sesijoms
                    all_data.append(df[['timestamp', 'mid_price', 'bid_wwap', 'ask_wwap', 'spread']].copy())
                    
                except Exception as e:
                    print(f"  Įspėjimas: Nepavyko nuskaityti {file_path}: {e}")
                    continue
        
        # Jei tiesiogiai CSV failas (alternatyvi struktūra)
        elif item_name.endswith('.csv'):
            file_path = item_path
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.lower().str.strip()
                
                if 'bid_wwap' in df.columns and 'ask_wwap' in df.columns and 'timestamp' in df.columns:
                    df['mid_price'] = (df['bid_wwap'] + df['ask_wwap']) / 2
                    df['spread'] = df['ask_wwap'] - df['bid_wwap']
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    all_data.append(df[['timestamp', 'mid_price', 'bid_wwap', 'ask_wwap', 'spread']].copy())
            except Exception as e:
                print(f"  Įspėjimas: Nepavyko nuskaityti {file_path}: {e}")
                continue
    
    if not all_data:
        return pd.DataFrame()
    
    # Sujungiame visus duomenis
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Rūšiuojame pagal timestamp
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    # Sukuriame OHLC struktūrą iš agreguotų duomenų
    # Grupuojame pagal valandas (arba dienas priklausomai nuo duomenų tankumo)
    combined_df['date_hour'] = combined_df['timestamp'].dt.floor('H')  # Valandos intervalas
    
    # Apskaičiuojame OHLC kiekvienai valandai
    hourly_ohlc = combined_df.groupby('date_hour').agg({
        'ask_wwap': 'max',      # High
        'bid_wwap': 'min',      # Low
        'mid_price': ['first', 'last'],  # Open, Close
        'spread': 'sum'         # Volume proxy
    }).reset_index()
    
    hourly_ohlc.columns = ['timestamp', 'high', 'low', 'open', 'close', 'volume']
    
    # Jei duomenys tankūs, galime naudoti minutės intervalus
    # Bet OHLC valandų lygmenyje turėtų užtekti
    hourly_ohlc = hourly_ohlc.sort_values('timestamp').reset_index(drop=True)
    
    return hourly_ohlc


def download_all_data_from_drive(drive_links: Dict[str, str], data_dir: str = 'data'):
    """
    Parsisiunčia visus duomenis iš Google Drive
    
    Args:
        drive_links: Žodynas {valiutos_poros: folderio_id}
        data_dir: Vietinis aplankas duomenims
    """
    print("Prisijungimas prie Google Drive...")
    service = authenticate_google_drive()
    
    if service is None:
        print("Nepavyko autentifikuotis. Naudokite rankinį duomenų parsisiuntimą.")
        return
    
    print(f"\nParsisiunčiami duomenys iš {len(drive_links)} folderių...")
    
    for currency_pair, folder_id in drive_links.items():
        print(f"\nParsisiunčiami {currency_pair} duomenys...")
        local_folder = os.path.join(data_dir, currency_pair)
        download_folder_files(service, folder_id, local_folder)
        print(f"  ✓ {currency_pair} duomenys parsisiųsti")
    
    print("\n✓ Visi duomenys parsisiųsti!")


def load_all_crypto_data(data_dir: str, drive_links: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Užkrauna visų kriptovaliutų duomenis
    
    Returns:
        Žodynas {valiutos_poros: DataFrame}
    """
    all_data = {}
    
    print(f"\nUžkraunami duomenys iš {data_dir}...")
    
    for currency_pair in drive_links.keys():
        print(f"  Užkraunami {currency_pair} duomenys...")
        df = load_crypto_data_from_local(data_dir, currency_pair)
        
        if not df.empty:
            all_data[currency_pair] = df
            print(f"    ✓ Užkrauta {len(df)} įrašų")
        else:
            print(f"    ✗ Nepavyko užkrauti {currency_pair} duomenų")
    
    print(f"\n✓ Iš viso užkrauta {len(all_data)} valiutų porų duomenys")
    
    return all_data

