import os

data_dir = '/home/ud202381476/data/BlendedMVS/temp/data'
names = os.listdir(data_dir)

for name in names:
    dst_dir = os.path.join(data_dir, name)
    if len(os.listdir(dst_dir)) == 1:
        src_dir = os.path.join(dst_dir, name, name)
        cmd = 'mv {}/* {}/'.format(src_dir, os.path.join(dst_dir))
        print(cmd)
        os.system(cmd)
        cmd = 'rm -rf {}'.format(os.path.join(dst_dir, name))
        print(cmd)
        os.system(cmd)
    if not os.path.exists(os.path.join(dst_dir, 'pair.txt')):
        cams_dir = os.path.join(dst_dir, 'cams')
        pair_path = os.path.join(cams_dir, 'pair.txt')
        cmd = 'mv {} {}/pair.txt'.format(pair_path, os.path.join(dst_dir))
        print(cmd)
        os.system(cmd)
    if os.path.exists(os.path.join(dst_dir, 'blended_images')):
        cmd = 'mv {} {}'.format(os.path.join(dst_dir, 'blended_images'), os.path.join(dst_dir, 'images'))
        print(cmd)
        os.system(cmd)
    if os.path.exists(os.path.join(dst_dir, 'occlusion_maps')):
        cmd = 'rmdir {}'.format(os.path.join(dst_dir, 'occlusion_maps'))
        print(cmd)
        os.system(cmd)
    if os.path.exists(os.path.join(dst_dir, 'rendered_depth_maps')):
        cmd = 'mv {} {}'.format(os.path.join(dst_dir, 'rendered_depth_maps'), os.path.join(dst_dir, 'depths_masks'))
        print(cmd)
        os.system(cmd)
    if os.path.exists(os.path.join(dst_dir, 'depths_masks')):
        depth_dir = os.path.join(dst_dir, 'depths_masks')
        depth_names = os.listdir(depth_dir)
        for depth_name in depth_names:
            index = int(depth_name.split('.')[0])
            cmd = 'mv {} {}'.format(os.path.join(depth_dir, depth_name), os.path.join(depth_dir, 'depth_map_{:0>4d}.pfm'.format(index)))
            # print(cmd)
            os.system(cmd)
        print('rename depth maps in {}'.format(depth_dir))