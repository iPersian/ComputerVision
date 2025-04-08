from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np


src_path = Path("jpegs_small")
dest_path = Path("masks_small")
dest_path.mkdir(parents=True, exist_ok=True)

folders = list(src_path.glob("*"))
tqdm_ = tqdm(folders, total=len(folders))

if __name__ == "__main__":
    for species in tqdm_:
        tqdm_.set_description_str(f"Processing {species}")
        for image in list(species.glob("*/*")):
            img = cv2.imread(image)
            lower = np.array([30, 87, 86])
            upper = np.array([50, 255, 255])
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(imgHSV, lower, upper)
            mask_fil = cv2.bilateralFilter(mask, 20, 121, 34)
            # Ensure mask remains binary by thresholding
            _, binary_mask_fil = cv2.threshold(mask_fil, 127, 255, cv2.THRESH_BINARY)
            new_path = (dest_path / image.relative_to(image.parts[0])).with_suffix(".png")
            new_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(new_path.resolve(), binary_mask_fil)