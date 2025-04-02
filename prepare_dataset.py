from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

jpeg_p = Path("jpegs_small")
mask_p = Path("masks_small")

df = pd.DataFrame(columns=['masks', 'label'])

X = []
Y = []

dt = 4 # Past images
dl = 4 # Future images to predict

for folder in mask_p.glob("*/*"):
    imgs = list(folder.glob("*"))
    for i in range(len(imgs) - dt - dl + 1):
        x = ",".join(str(img) for img in imgs[i: i+dt])
        y = imgs[i + dt + dl - 1]
        X.append(x)
        Y.append(y)

new_row = pd.DataFrame({'masks': X, 'label': Y})
df = pd.concat([df, new_row], ignore_index=True)

Path(f"dataset_{dt}_{dl}").mkdir(parents=True, exist_ok=True)

trainval, test = train_test_split(df, random_state=42, test_size=0.2)
train, validation = train_test_split(df, random_state=42, test_size=0.25)

train.to_csv(f'dataset_{dt}_{dl}/train_{dt}_{dl}.csv', index=False)
test.to_csv(f'dataset_{dt}_{dl}/test_{dt}_{dl}.csv', index=False)
validation.to_csv(f'dataset_{dt}_{dl}/val_{dt}_{dl}.csv', index=False)