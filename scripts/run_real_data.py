from app.data_loader import load_data

df = load_data()

print("Dataset loaded successfully!")
print(df.head())