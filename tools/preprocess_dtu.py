import os

##########################################################
data_dir = '/home/ubuntu/Data/DTU'
cams_dir = os.path.join(data_dir, 'cameras')
images_dir = os.path.join(data_dir, 'images')
depths_dir = os.path.join(data_dir, 'depths_masks')
type = "test"
##########################################################

if type == "test":
    target_dir = os.path.join(data_dir, 'test')
    scans = ["scan1", "scan10", "scan11", "scan110", "scan114", "scan118", "scan12", "scan13", "scan15", "scan23",
             "scan24", "scan29", "scan32", "scan33", "scan34", "scan4", "scan48", "scan49", "scan62", "scan75",
             "scan77", "scan9"]
else:
    target_dir = os.path.join(data_dir, 'train')
    scans = os.listdir(depths_dir)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print('create target dir: {}'.format(target_dir))

for scan in scans:
    print(scan)
    src_img_dir = os.path.join(images_dir, scan)
    src_depth_dir = os.path.join(depths_dir, scan)
    src_cam_dir = cams_dir
    src_pair_path = os.path.join(src_cam_dir, 'pair.txt')
    scan_train_dir = os.path.join(target_dir, scan)
    if not os.path.exists(scan_train_dir):
        os.makedirs(scan_train_dir)
        print('create scan target dir: {}'.format(scan_train_dir))
    dst_img_dir = os.path.join(scan_train_dir, 'images')
    if not os.path.exists(dst_img_dir):
        os.makedirs(dst_img_dir)
        print('create dst img dir: {}'.format(dst_img_dir))
    dst_cam_dir = os.path.join(scan_train_dir, 'cams')
    if not os.path.exists(dst_cam_dir):
        os.makedirs(dst_cam_dir)
        print('create dst cam dir: {}'.format(dst_cam_dir))
    for i in range(49):
        src_img_path = os.path.join(src_img_dir, 'rect_{:0>3d}_3_r5000.png'.format(i + 1))
        dst_img_path = os.path.join(dst_img_dir, '{:0>8d}.png'.format(i))
        os.system('cp {} {}'.format(src_img_path, dst_img_path))
        src_cam_path = os.path.join(src_cam_dir, '{:0>8d}_cam.txt'.format(i))
        dst_cam_path = os.path.join(dst_cam_dir, '{:0>8d}_cam.txt'.format(i))
        os.system('cp {} {}'.format(src_cam_path, dst_cam_path))
    dst_pair_path = os.path.join(scan_train_dir, 'pair.txt')
    os.system('cp {} {}'.format(src_pair_path, dst_pair_path))
    if type != "test":
        os.system('ln -s {} {}'.format(src_depth_dir, os.path.join(scan_train_dir, 'depths_masks')))
