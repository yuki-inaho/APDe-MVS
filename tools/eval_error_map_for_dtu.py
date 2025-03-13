import os
import argparse
import numpy as np
import re
import cv2
import tqdm
import matplotlib.pyplot as plt
import torch
import plyfile


def read_bin_mat(path):
    """
    :param path: the path of bin file
    :return: the mat of bin file
    """
    with open(path, 'rb') as f:
        version = int.from_bytes(f.read(4), byteorder='little')
        rows = int.from_bytes(f.read(4), byteorder='little')
        cols = int.from_bytes(f.read(4), byteorder='little')
        type = int.from_bytes(f.read(4), byteorder='little')
        if version != 1:
            raise Exception("Version error: ", path)

        # CV_8UC1 = 0x00
        # CV_32SC1 = 0x04
        # CV_32FC1 = 0x05
        if type == 0x04:
            mat = np.fromfile(f, dtype=np.int32, count=rows * cols)
            mat = mat.reshape((rows, cols))
        elif type == 0x00:
            mat = np.fromfile(f, dtype=np.uint8, count=rows * cols)
            mat = mat.reshape((rows, cols))
        elif type == 0x05:
            mat = np.fromfile(f, dtype=np.float32, count=rows * cols)
            mat = mat.reshape((rows, cols))

        return mat


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def depth_to_pointcloud(depth, path):
    # depth [H, W]: torch tensor
    # path: save path

    # depth_np = depth.cpu().numpy()  # [H, W]
    # rgb = image.cpu().numpy()  # [3, H, W]
    # rgb = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)  # [H, W, 3]
    # points = depth_image_to_point_cloud(rgb, depth_np, 1.0, intrinsics.cpu().numpy(), extrinsics.cpu().numpy())
    # write_point_cloud(path, points)

    depth = torch.from_numpy(depth).float().cuda()
    intrinsics = np.array(
        [[2892.33, 0, 823.205],
         [0, 2883.18, 619.071],
         [0, 0, 1]]
    )

    extrinsics = np.eye(4)
    intrinsics = torch.from_numpy(intrinsics).float().cuda()
    extrinsics = torch.from_numpy(extrinsics).float().cuda()

    H, W = depth.shape
    x, y = torch.meshgrid(torch.arange(W, device=depth.device), torch.arange(H, device=depth.device), indexing='xy')
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = depth.reshape(-1)
    mask = z > 0
    x = x[mask]
    y = y[mask]
    z = z[mask]
    x = (x - intrinsics[0, 2]) / intrinsics[0, 0]
    y = (y - intrinsics[1, 2]) / intrinsics[1, 1]
    xyz = torch.stack([x, y, torch.ones_like(x, device=depth.device)], dim=0)
    xyz = xyz * z
    xyz = torch.mm(torch.inverse(extrinsics), torch.cat([xyz, torch.ones_like(xyz[0, :]).unsqueeze(0)], dim=0))[0:3, :]
    xyz = xyz.permute(1, 0).cpu().numpy()
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4')
    ]
    elements = np.zeros(xyz.shape[0], dtype=dtype)
    attributes = xyz
    elements[:] = list(map(tuple, attributes))
    el = plyfile.PlyElement.describe(elements, 'vertex')
    plyfile.PlyData([el]).write(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--APD_dir', type=str, default='/home/zzj/Work/Data/DTU/test/scan10/APD_with_impetus')
    parser.add_argument('--APD_dir', type=str, default='/home/zzj/Work/Data/DTU/test/scan10/APD_wo_impetus')
    parser.add_argument('--gt_dir', type=str, default='/home/zzj/Work/Data/DTU/Depths/scan10')

    args = parser.parse_args()
    APD_dir = args.APD_dir
    gt_depth_dir = args.gt_dir

    if not os.path.exists(APD_dir):
        print('{} does not exist'.format(APD_dir))
        exit(1)
    if not os.path.exists(gt_depth_dir):
        print('{} does not exist'.format(gt_depth_dir))
        exit(1)

    images_folder = os.path.join(APD_dir)
    # 使用regex 进行过滤 所有的image_folder 都是 诸如 00000000 00000001 这样的 {:08d}
    images_folder_name = [f for f in os.listdir(images_folder) if re.match(r'\d{8}', f)]
    images_folder_name.sort()
    images_idx = [int(f) for f in images_folder_name]

    print('total images: ', len(images_idx))
    APD_depths = []
    gt_depths = []
    weak_masks = []
    for idx in images_idx:
        APD_path = os.path.join(APD_dir, '{:08d}'.format(idx), 'depths.bin')
        gt_depth_path = os.path.join(gt_depth_dir, 'depth_map_{:04d}.pfm'.format(idx))
        weak_mask_path = os.path.join(APD_dir, '{:08d}'.format(idx), 'weak.bin')
        if not os.path.exists(APD_path):
            print('{} does not exist'.format(APD_path))
            continue
        if not os.path.exists(gt_depth_path):
            print('{} does not exist'.format(gt_depth_path))
            continue
        if not os.path.exists(weak_mask_path):
            print('{} does not exist'.format(weak_mask_path))
            continue

        APD_depth = read_bin_mat(APD_path)
        gt_depth, _ = read_pfm(gt_depth_path)
        weak_mask = read_bin_mat(weak_mask_path)
        weak_mask = (weak_mask != 2) # 只计算强纹理的地方

        APD_depths.append(APD_depth)
        gt_depths.append(gt_depth)
        weak_masks.append(weak_mask)

    # 评估
    eval_folder = os.path.join(APD_dir, 'eval_error_map')
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)

    diff_sum = 0
    # x_min, y_min = 462, 242
    # x_max, y_max = 932, 526
    x_min, y_min = 0, 0
    x_max, y_max = 1600, 1200
    for i in tqdm.tqdm(range(len(images_idx))):
        if i != 14:
            continue
        # crop
        APD_depth = APD_depths[i]
        gt_depth = gt_depths[i]
        weak_mask = weak_masks[i]

        depth_to_pointcloud(APD_depth.copy(), os.path.join(eval_folder, '{:08d}_apd.ply'.format(images_idx[i])))
        depth_to_pointcloud(gt_depth.copy(), os.path.join(eval_folder, '{:08d}_gt.ply'.format(images_idx[i])))

        APD_depth = APD_depth[y_min:y_max, x_min:x_max]
        gt_depth = gt_depth[y_min:y_max, x_min:x_max]
        weak_mask = weak_mask[y_min:y_max, x_min:x_max]
        weak_ratio = np.sum(weak_mask) / (x_max - x_min) / (y_max - y_min)
        print('weak ratio: ', weak_ratio)

        # diff
        mask = gt_depth > 0
        diff = APD_depth - gt_depth
        diff = diff * mask * weak_mask
        # diff mean
        diff_mean = np.mean(np.abs(diff[mask]))
        diff_sum += diff_mean
        print("mean diff: ", diff_mean)
        # vis
        # diff_max = np.max(diff)
        diff_max = 2.5
        diff = diff / diff_max
        diff = np.clip(diff, -0.5, 0.5)
        diff = np.uint8((diff + 0.5) * 255)

        # # 二值化 diff 小于 128 的为 0 大于 128 的为 255
        # diff = cv2.threshold(diff, 128, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite(os.path.join(eval_folder, '{:08d}.png'.format(images_idx[i])), diff)
        plt.imshow(diff)
        plt.show()
        exit(0)

    print('mean diff: ', diff_sum / len(images_idx))
    print('eval error map done')
