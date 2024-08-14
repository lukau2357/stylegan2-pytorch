import os
import math
import cv2
import tqdm

def preprocess(img_size):
    """
    CelebA (at least the original form) contains images of size 218x178. 218 = 2 x 89, 178 = 2 x 109, since layers inside
    generator and critic should double/shrink resolutions by factor of 2, this is inconvenient. Hence, we resize all images of CelebA to 128 x 128.
    We perform dimension shrinking using pixel area relation.
    """
    x = math.log2(img_size)
    assert x == int(x), f"Given image size {img_size} must be a power of 2"
    source_dir = os.path.join("img_align_celeba", "img_align_celeba")
    target_dir = f"img_alig_celeba_{img_size}"

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    print(f"Resizing CelebA images to {img_size}:")

    for label in tqdm.tqdm(os.listdir(source_dir)):
        img = cv2.imread(os.path.join(source_dir, label))

        if max(img.shape) <= img_size:
            img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_CUBIC)
        
        else:
            img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_AREA)

        cv2.imwrite(os.path.join(target_dir, label), img)


if __name__ == "__main__":
    preprocess(256)