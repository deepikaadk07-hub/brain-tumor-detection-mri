import pandas as pd

data = [
    [1200, 15, 200, "Small"],
    [5000, 40, 600, "Medium"],
    [12000, 80, 1500, "Large"]
]

df = pd.DataFrame(data, columns=[
    "volume", "slices", "max_area", "label"
])

df.to_csv("tumor_features.csv", index=False)
print("Dataset created")