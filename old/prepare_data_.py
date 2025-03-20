from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import zipfile
import skimage
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

csv_path = Path("../gt.csv")
zip_path = Path("../jpegs")
save_path = Path("patches")

save_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)


def split_train_val_test(df, seed:int=42, test_size:int=0.2, validation_size:int=0.25):
    df["split"] = None
    df_tracks = df.groupby(by="track_id").first().reset_index()
    trainval, test = train_test_split(df_tracks, stratify=df_tracks.label_id,random_state=seed,test_size=test_size)
    train, validation = train_test_split(trainval, stratify=trainval.label_id,random_state=seed,test_size=validation_size)
    for idx, val in train.iterrows():
        df.loc[df.track_id == val.track_id, "split"] = "train"
    for idx, val in validation.iterrows():
        df.loc[df.track_id == val.track_id, "split"] = "validation"
    for idx, val in test.iterrows():
        df.loc[df.track_id == val.track_id, "split"] = "test"
    return

# ------------- Merge varieties------------
# Varieties Before
# print("Before merge")
# print((np.unique(df.label_id)))

# Merge varieties
df.loc[df.label_id.isin(["SORFR", "SORHA", "SORKM", "SORKS", "SORRS", "SORSA"]), "label_id"] = "SORVU"
df.loc[df.label_id.isin(["ZEAKJ", "ZEALP"]), "label_id"] = "ZEAMX"

# Varieties after filtering
# print("After merge")
# print((np.unique(df.label_id)))


#------------- Remove unknown weeds--------------

df = df[df.label_id != "Weed"]

# print("Remove unknown weeds")
# print((np.unique(df.label_id)))


# -------------- remove specific species

df = df[~df.label_id.isin(["VICVI", "POLAV"])]

# print("Remove Specific species")
# print((np.unique(df.label_id)))

split_train_val_test(df)

zips = sorted(zip_path.glob("*/*.zip"))

tqdm_ = tqdm(zips, total=len(zips))
for zip_file in tqdm_:
    tqdm_.set_description_str(f"{zip_file}")
    with zipfile.ZipFile(zip_file, mode="r") as archive:
        for file in sorted(archive.namelist()[1:]):
            # print(f"Current file: {file}")
            img = skimage.io.imread(archive.open(file))  # Replace with your image path
            rescale = df.loc[df.filename.str.contains(Path(file).stem)]

            if len(rescale) > 0:
                for idx, val in rescale.iterrows():
                    patch = img[val.ymin:val.ymax, val.xmin:val.xmax, :]
                    save_folder_path = save_path / val.split / val.label_id
                    save_folder_path.mkdir(parents=True, exist_ok=True)
                    skimage.io.imsave(f"{save_folder_path}/{val.tray_id}_{val.bbox_id}.jpeg", patch, check_contrast=False)