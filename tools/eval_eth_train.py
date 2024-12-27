import os
import multiprocessing as mp
import argparse
from texttable import Texttable
import numpy as np

#####################################################################################################
# args:
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/zzj/Work/Data/ETH3D/data')
parser.add_argument('--eval_program', type=str,
                    default='/home/zzj/Work/Data/ETH3D/multi-view-evaluation/build/ETH3DMultiViewEvaluation')
parser.add_argument('--gt_dir', type=str, default='/home/zzj/Work/Data/ETH3D/gt')
parser.add_argument('--scans', type=str, nargs='+', default=[])
parser.add_argument('--reservation', type=str, default=None, help='reservation for the server, e.g. 3h30m10s')
parser.add_argument('--no_save_ply', action='store_false', default=True)
parser.add_argument('--work_num', type=int, default=6)
parser.add_argument('--resume', action='store_true', default=False)
args = parser.parse_args()


def worker(scan):
    scan_dir = os.path.join(args.data_dir, scan)
    if not os.path.exists(scan_dir):
        print('{} does not exist'.format(scan_dir))
        return
    APD_dir = os.path.join(scan_dir, 'APD')
    if not os.path.exists(APD_dir):
        print('{} does not exist'.format(APD_dir))
        return
    APD_path = os.path.join(APD_dir, 'APD.ply')
    if not os.path.exists(APD_path):
        print('{} does not exist'.format(APD_path))
        return
    gt_path = os.path.join(args.gt_dir, scan, 'dslr_scan_eval', 'scan_alignment.mlp')
    if not os.path.exists(gt_path):
        print('{} does not exist'.format(gt_path))
        return
    eval_cmd = '{} --reconstruction_ply_path {} --ground_truth_mlp_path {} --tolerances 0.01,0.02,0.05,0.1,0.2,0.5'.format(
        args.eval_program, APD_path, gt_path)
    if args.no_save_ply:
        cmp_dir = os.path.join(APD_dir, 'compare', 'cmp')
        acc_dir = os.path.join(APD_dir, 'compare', 'acc')
        eval_cmd += ' --completeness_cloud_output_path {} --accuracy_cloud_output_path {}'.format(cmp_dir, acc_dir)
    result_path = os.path.join(APD_dir, 'result.txt')
    if os.path.exists(result_path) and args.resume:
        print('{} already exists'.format(result_path))
        return
    eval_cmd += ' > {}'.format(result_path)
    print('eval_cmd: {}'.format(eval_cmd))
    os.system(eval_cmd)


def parse_result(scan):
    scan_dir = os.path.join(args.data_dir, scan)
    APD_dir = os.path.join(scan_dir, 'APD')
    result_path = os.path.join(APD_dir, 'result.txt')
    if not os.path.exists(result_path):
        print('{} does not exist'.format(result_path))
        return
    result = {}
    with open(result_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("Completenesses"):
                line = line.split(':')[-1].strip()
                line = line.split(' ')
                line = [float(x) for x in line]
                result['completeness'] = {}
                result['completeness']['0.01'] = line[0]
                result['completeness']['0.02'] = line[1]
                result['completeness']['0.05'] = line[2]
                result['completeness']['0.1'] = line[3]
                result['completeness']['0.2'] = line[4]
                result['completeness']['0.5'] = line[5]
            elif line.startswith('Accuracies'):
                line = line.split(':')[-1].strip()
                line = line.split(' ')
                line = [float(x) for x in line]
                result['accuracy'] = {}
                result['accuracy']['0.01'] = line[0]
                result['accuracy']['0.02'] = line[1]
                result['accuracy']['0.05'] = line[2]
                result['accuracy']['0.1'] = line[3]
                result['accuracy']['0.2'] = line[4]
                result['accuracy']['0.5'] = line[5]
            elif line.startswith('F1-scores'):
                line = line.split(':')[-1].strip()
                line = line.split(' ')
                line = [float(x) for x in line]
                result['f1'] = {}
                result['f1']['0.01'] = line[0]
                result['f1']['0.02'] = line[1]
                result['f1']['0.05'] = line[2]
                result['f1']['0.1'] = line[3]
                result['f1']['0.2'] = line[4]
                result['f1']['0.5'] = line[5]
    return result


def show(results):
    # tolerances = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    tolerances = [0.02, 0.1]
    scan = [result['scan'] for result in results]
    # short scan name
    for i in range(len(scan)):
        if len(scan[i]) > 6:
            scan[i] = scan[i][:6] + '.'
    header = ['Data'] + scan + ['Average']
    for tolerance in tolerances:
        print('tolerance: {}'.format(tolerance))
        table = Texttable(max_width=0)
        table.header(header)
        table.set_cols_dtype(['t'] + ['f' for result in results] + ['f'])
        table.set_cols_align(['c'] + ['c' for result in results] + ['c'])
        table.set_precision(2)
        for key in ['completeness', 'accuracy', 'f1']:
            row = [key]
            data = [result['result'][key][str(tolerance)] * 100 for result in results]
            row.extend(data)
            row.append(np.mean(data))
            table.add_row(row)
        print(table.draw() + '\n')


if __name__ == "__main__":
    print(args)
    if args.reservation is not None:
        # sleep for reservation
        print('sleep for reservation: {}'.format(args.reservation))
        os.system('sleep {}'.format(args.reservation))
        print('sleep done')

    if args.scans:
        scans = args.scans
    else:
        scans = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'meadow', 'office', 'pipes', 'playground',
                 'relief', 'relief_2', 'terrace', 'terrains']

    print('scans size: {}'.format(len(scans)))
    pool = mp.Pool(processes=min(len(scans), args.work_num))
    pool.map(worker, scans)
    pool.close()
    pool.join()
    # parse result
    results = []
    for scan in scans:
        result = {}
        result['scan'] = scan
        result['result'] = parse_result(scan)
        results.append(result)
    # show result
    show(results)
