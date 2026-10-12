import pandas as pd

raw = pd.read_csv("data/raw/dataset.csv", index_col=0)

song_key = raw["artists"].astype(str) + " | " + raw["track_name"].astype(str)
test_songs = song_key.drop_duplicates().sample(frac=0.2, random_state=42)

is_test = song_key.isin(test_songs)
train_data = raw[~is_test]
test_data = raw[is_test]

def to_sheet_format(frame):
    formatted = (frame.rename(columns={"track_genre": "TARGET"})
                        .rename_axis("SAMPLE ID")
                        .reset_index())
    formatted.insert(1, "TARGET", formatted.pop("TARGET"))
    return formatted

with pd.ExcelWriter("notebooks/Data.xlsx") as writer:
    to_sheet_format(train_data).to_excel(writer, sheet_name="Training", index=False)
    to_sheet_format(test_data).to_excel(writer, sheet_name="Testing", index=False)

print(f"Training rows: {len(train_data)}, Testing Rows: {len(test_data)}")