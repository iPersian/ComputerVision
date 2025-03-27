from pathlib import Path
import pandas as pd

jpeg_p = Path("jpegs_small")
mask_p = Path("masks_small")

df = pd.DataFrame(columns=['masks', 'label'])

X = []
Y = []

dt = 4 # Past images
dl = 1 # Future images to predict

for folder in mask_p.glob("*/*"):
    imgs = list(folder.glob("*"))
    for i in range(len(imgs) - dt - dl + 1):
        x = ",".join(str(img) for img in imgs[i: i+dt])
        y = imgs[i + dt + dl - 1]
        X.append(x)
        Y.append(y)

new_row = pd.DataFrame({'masks': X, 'label': Y})
df = pd.concat([df, new_row], ignore_index=True)
df.to_csv('training.csv', index=False)