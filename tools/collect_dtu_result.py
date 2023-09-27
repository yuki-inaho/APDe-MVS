import os

data_sir = '/home/ud202381476/data/DTU/test'
collect_dir = '/home/ud202381476/data/DTU/eval/MVSData/Points'
if not os.path.exists(collect_dir):
    os.mkdir(collect_dir)

scans = os.listdir(data_sir)
for scan in scans:
    APD_dir = os.path.join(data_sir, scan, 'APD')
    if not os.path.exists(APD_dir):
        print("APD result not exist: ", scan)
        continue
    src_ply = os.path.join(APD_dir, 'APD.ply')
    if not os.path.exists(src_ply):
        print("ply not exist: ", scan)
        continue
    scan_index = scan.split('scan')[1]
    if not os.path.exists(os.path.join(collect_dir, 'apd')):
        os.mkdir(os.path.join(collect_dir, 'apd'))
    dst_ply = os.path.join(collect_dir, 'apd', 'apd{:0>3}_l3.ply'.format(int(scan_index)))
    os.system('cp {} {}'.format(src_ply, dst_ply))
    print('copy {} to {}'.format(src_ply, dst_ply))