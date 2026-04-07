import pandas as pd

def load_data():
    df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="latin1")
    return df