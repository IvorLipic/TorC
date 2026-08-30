import pandas as pd
from PIL import Image
import io
import os

def convert_parquet_to_csv(parquet_path, csv_path):
    df = pd.read_parquet(parquet_path)
    with open(csv_path, 'w') as f:
        for _, row in df.iterrows():
            label = row['label']
            img_bytes = row['image']['bytes']
            img = Image.open(io.BytesIO(img_bytes))
            pixels = list(img.getdata())
            row_data = [str(label)] + [str(p / 255.0) for p in pixels]
            f.write(','.join(row_data) + '\n')
    print(f"Converted {parquet_path} -> {csv_path} ({len(df)} samples)")

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    convert_parquet_to_csv(
        os.path.join(base, 'train-00000-of-00001.parquet'),
        os.path.join(base, 'mnist_train.csv')
    )
    convert_parquet_to_csv(
        os.path.join(base, 'test-00000-of-00001.parquet'),
        os.path.join(base, 'mnist_test.csv')
    )
