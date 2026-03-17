import pandas as pd
from pathlib import Path

class Olist:
    def __init__(self):
        # Notebook'taki ile aynı yolu tanımlıyoruz
        self.csv_path = Path("~/.workintech/olist/data/csv").expanduser()

    def get_data(self):
        """
        9 CSV dosyasını okur, isimlerini temizler ve bir dict içinde döndürür.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Veri yolu bulunamadı: {self.csv_path}")

        # Dosya yollarını listele
        file_paths = list(self.csv_path.iterdir())

        # Dosya isimlerini al
        file_names = [path.name for path in file_paths if path.suffix == '.csv']

        # Anahtar isimlerini temizle
        key_names = [
            name.replace('olist_', '').replace('_dataset.csv', '').replace('.csv', '')
            for name in file_names
        ]

        # Sözlüğü oluştur (Notebook'ta yaptığımız mantıkla)
        data = {
            key: pd.read_csv(path)
            for key, path in zip(key_names, file_paths)
        }

        return data
