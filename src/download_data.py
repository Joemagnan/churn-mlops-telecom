"""Télécharger le dataset pédagogique public depuis le dépôt d'IBM."""
from urllib.request import urlretrieve
from src.config import DATA_PATH
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def main():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(URL, DATA_PATH)
    print(f"Fichier téléchargé : {DATA_PATH}")

if __name__ == "__main__":
    main()
