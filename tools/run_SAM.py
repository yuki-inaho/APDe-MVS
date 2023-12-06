import numpy as np
import torch
import cv2
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import tqdm
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default=None)
parser.add_argument('--max_size', type=int, default=2560)
parser.add_argument('--scans', type=str, nargs='+', default=None)
args = parser.parse_args()


def write_bin_mat(filename, mat):
    with open(filename, 'wb') as f:
        version = 1
        rows = mat.shape[0]
        cols = mat.shape[1]
        if len(mat.shape) == 2:
            if mat.dtype == np.uint8:
                type = 0
            elif mat.dtype == np.float32:
                type = 0x05
        else:
            if mat.dtype == np.float32:
                type = 0x15
        f.write(version.to_bytes(4, byteorder='little'))
        f.write(rows.to_bytes(4, byteorder='little'))
        f.write(cols.to_bytes(4, byteorder='little'))
        f.write(type.to_bytes(4, byteorder='little'))

        # CV_8UC1  = 0x00
        # CV_32FC3 = 0x15
        # CV_32FC1 = 0x05
        if type == 0:
            mat = mat.reshape((rows * cols))
            f.write(mat.tobytes())
        elif type == 0x05:
            mat = mat.reshape((rows * cols))
            f.write(mat.tobytes())
        elif type == 0x15:
            mat = mat.reshape((rows * cols * 3))
            f.write(mat.tobytes())


def save_anns(origin_img, anns, save_path):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # expend the origin image channel to 4
    mask_img = np.ones(origin_img.shape, dtype=np.float32)
    mask = np.zeros(origin_img.shape[:2], dtype=np.uint8)
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.random.random((1, 3)).tolist()[0]
        mask[m] = i + 1
        mask_img[m] = color_mask

    if save_path is None:
        plt.imshow(mask_img)
        plt.show()
    else:
        cv2.imwrite(save_path, mask_img * 255)
        write_bin_mat(save_path.replace('.png', '.bin'), mask)


if __name__ == "__main__":

    if args.scans is not None:
        scans = args.scans
    else:
        print("work dir: ", args.work_dir)
        scans = os.listdir(args.work_dir)
    print("total scans: ", len(scans))

    sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("model loaded")

    for scan in scans:
        scan_path = os.path.join(args.work_dir, scan)
        image_folder = os.path.join(scan_path, 'images')
        if not os.path.exists(image_folder):
            raise Exception("image path not exists")
        mask_folder = os.path.join(scan_path, 'sa_masks')
        if not os.path.exists(mask_folder):
            os.mkdir(mask_folder)
        print("processing scan: ", scan)
        for img_name in tqdm.tqdm(os.listdir(image_folder)):
            img_path = os.path.join(image_folder, img_name)
            save_path = os.path.join(mask_folder, img_name.split('.')[0] + '.png')  # save as png
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if max(image.shape) > args.max_size:
                scale = args.max_size / max(image.shape)
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            masks = mask_generator.generate(image)
            save_anns(image, masks, save_path)