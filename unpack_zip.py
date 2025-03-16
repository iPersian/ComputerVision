import zipfile
from pathlib import Path
from tqdm import tqdm
from PIL import Image


path = Path("jpegs")
save_path = Path("jpegs_small")
save_path.mkdir(parents=True, exist_ok=True)


def resize_image(image_path, size=(224, 224)):
    with Image.open(image_path) as img:
        img_resized = img.resize(size)
        img_resized.save(image_path)


if __name__ == "__main__":
    zips = list(path.glob("*/*.zip"))
    tqdm_ = tqdm(zips, total=len(zips))
    for zipfile_ in tqdm_:
        tqdm_.set_description_str(f"Unpacking: {zipfile_}")
        with zipfile.ZipFile(zipfile_, mode='r') as archive:
            save_p = save_path / Path(*zipfile_.parent.parts[1:])
            archive.extractall(save_p)
        images = list(save_p.glob("*/*.jpeg"))
        for i in images:
            resize_image(i)