import os

data_sir = '/home/ud202381476/data/ETH3D/data'
collect_dir = '/home/ud202381476/data/ETH3D/result'
if not os.path.exists(collect_dir):
    os.mkdir(collect_dir)

scans = os.listdir(data_sir)
for scan in scans:
    scan_dir = os.path.join(data_sir, scan)
    APD_dir = os.path.join(scan_dir, 'APD')
    if not os.path.exists(APD_dir):
        print("APD result not exist: ", scan)
        continue
    src_ply = os.path.join(APD_dir, 'APD.ply')
    if not os.path.exists(src_ply):
        print("ply not exist: ", scan)
        continue
    dst_ply = os.path.join(collect_dir, '{}.ply'.format(scan))
    dst_log = os.path.join(collect_dir, '{}.txt'.format(scan))
    os.system('cp {} {}'.format(src_ply, dst_ply))
    print('copy {} -> {}'.format(src_ply, dst_ply))
    with open(dst_log, 'w') as f:
        f.write('runtime 9999999')
    print('write log -> {}'.format(dst_log))