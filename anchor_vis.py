# Function: visualize the anchor points
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import argparse
import os

parser = argparse.ArgumentParser("Visualize the anchor points")
sub_parser = parser.add_subparsers(dest="mode")
sub_parser.required = True
parser_a = sub_parser.add_parser("scan", help="visualize the scan aa_show_point_data.\n" +
                                              "For example, if scan_dir is 'DTU/test/scan10, and view_index is 0,\n" +
                                              "the following files should be exist:\n" +
                                              "DTU/test/scan10/images/00000000.png\n" +
                                              "DTU/test/scan10/ADP/00000000/anchors_map.bin\n" +
                                              "DTU/test/scan10/ADP/00000000/anchors.bin\n" +
                                              "and optional:\n" +
                                              "DTU/test/scan10/sa_masks/sa_mask.bin")
parser_a.add_argument("--scan_dir", type=str, default="aa_show_point_data/anchor/scan",
                      help="the dir of scan aa_show_point_data")
parser_a.add_argument("--view_index", type=int, default=0, help="the index of view")
parser_a.add_argument("--close_pick", action="store_true", default=False, help="close double click to pick a point"
                                                                               " to show the anchors")
parser_a.add_argument("--init_xy", type=int, nargs=2, default=None, help="the init x and y")
parser_a.add_argument("--show_sa_mask", action="store_true", default=False, help="show the sa_mask")

parser_b = sub_parser.add_parser("specify", help="specify the path of image, anchors_map, anchors and optional sa_mask")
parser_b.add_argument("--img_path", type=str, default=None, required=True, help="the path of image")
parser_b.add_argument("--anchors_map_path", type=str, default=None, required=True, help="the path of anchors_map")
parser_b.add_argument("--anchors_path", type=str, default=None, required=True, help="the path of anchors")
parser_b.add_argument("--sa_mask_path", type=str, default=None, required=False, help="the path of sa_mask (bin file)")
parser_b.add_argument("--close_pick", action="store_true", default=False, help="close double click to pick a point"
                                                                               " to show the anchors")
parser_b.add_argument("--init_xy", type=int, nargs=2, default=None, help="the init x and y")
parser_b.add_argument("--show_sa_mask", action="store_true", default=False, help="show the sa_mask")

args = parser.parse_args()


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
        if type == 0x04:
            mat = np.fromfile(f, dtype=np.int32, count=rows * cols)
            mat = mat.reshape((rows, cols))
        elif type == 0x00:
            mat = np.fromfile(f, dtype=np.uint8, count=rows * cols)
            mat = mat.reshape((rows, cols))

        return mat


def read_anchor_map(path):
    """
    :param path: the path of anchors_map.bin
    :return: anchors_map
    """
    return read_bin_mat(path)


def read_sa_mask(path):
    """
    :param path: the path of sa_mask.png
    :return: sa_mask
    """
    return read_bin_mat(path)


def read_anchor(path):
    """
    :param path: the path of anchors.bin
    :return: anchors
    """
    # the anchors.bin is a bin file with format:
    # weak_count(int32), anchor_sampe_num(int32), anchors(short2 * weak_count * anchor_sampe_num)
    with open(path, 'rb') as f:
        weak_count = int.from_bytes(f.read(4), byteorder='little')
        anchor_sampe_num = int.from_bytes(f.read(4), byteorder='little')
        anchors = np.fromfile(f, dtype=np.int16, count=weak_count * anchor_sampe_num * 2)
        anchors = anchors.reshape((weak_count, anchor_sampe_num, 2))
        return anchors


def get_valid_anchors(x, y, anchors, anchors_map, sa_mask=None):
    invalid_flag = -1
    if anchors_map[y, x] == invalid_flag:
        print("Reliable point, no anchors")
        return None
    index = anchors_map[y, x]
    assert 0 <= index < anchors.shape[0]
    cur_anchors = anchors[index, :, :]
    valid_anchors = []
    for i in range(cur_anchors.shape[0]):
        if cur_anchors[i, 0] == invalid_flag and cur_anchors[i, 1] == invalid_flag:
            continue
        valid_anchors.append(cur_anchors[i, :])
    cur_anchors = np.array(valid_anchors)

    assert cur_anchors[0, 0] == x and cur_anchors[0, 1] == y
    # print the anchors
    if sa_mask is not None:
        print('############## anchors ##############')
    else:
        print('######## anchors ########')
    for i in range(cur_anchors.shape[0]):
        if sa_mask is None:
            print("# (x, y): [{:4}, {:4}]  #".format(cur_anchors[i, 0], cur_anchors[i, 1]))
        else:
            print("# (x, y): [{:4}, {:4}], sa_mask: {:2} #".format(
                cur_anchors[i, 0], cur_anchors[i, 1], sa_mask[cur_anchors[i, 1], cur_anchors[i, 0]]))
    if sa_mask is not None:
        print('######################################')
    else:
        print('#########################')
    return cur_anchors


def show_point(x, y, H, W, cur_anchors, sa_mask):
    color_map = {
        'weak_center': 'darkgreen',
        'weak_neighbor': 'lightgreen',
        'strong_center': 'firebrick',
        'strong_neighbor': 'khaki',
        'invalid_neighbor': 'royalblue',
    }
    alpha = 0.8
    scatter_size = int(min(W, H) * 0.05)
    line_width = 2

    # show its ncc window with radius=5 and increment=2
    # if the point's sa_mask is the same as center point, set it to purple, else set it to gray
    for j in range(-5, 6, 2):
        for k in range(-5, 6, 2):
            if j == 0 and k == 0:
                continue
            if cur_anchors[0, 0] + j < 0 or cur_anchors[0, 0] + j >= W or \
                    cur_anchors[0, 1] + k < 0 or cur_anchors[0, 1] + k >= H:
                continue
            if sa_mask is not None:
                if sa_mask[cur_anchors[0, 1] + k, cur_anchors[0, 0] + j] == sa_mask[y, x]:
                    plt.scatter(cur_anchors[0, 0] + j, cur_anchors[0, 1] + k, c=color_map['weak_neighbor'], alpha=alpha,
                                s=scatter_size, linewidths=line_width, edgecolors='black')
                else:
                    plt.scatter(cur_anchors[0, 0] + j, cur_anchors[0, 1] + k, c=color_map['invalid_neighbor'],
                                alpha=alpha, s=scatter_size, linewidths=line_width, edgecolors='black')
            else:
                plt.scatter(cur_anchors[0, 0] + j, cur_anchors[0, 1] + k, c=color_map['weak_neighbor'], alpha=alpha,
                            s=scatter_size, linewidths=line_width, edgecolors='black')

    # set the center point to green
    plt.scatter(cur_anchors[0, 0], cur_anchors[0, 1], c=color_map['weak_center'], s=scatter_size, linewidths=line_width, edgecolors='black')

    # cur_anchors[1:, :], the other points,
    # if its sa_mask is the same as center point, set it to red, else set it to gray
    for i in range(1, cur_anchors.shape[0]):
        if sa_mask is not None and sa_mask[cur_anchors[i, 1], cur_anchors[i, 0]] != sa_mask[y, x]:
            continue
        # then for each point, show its ncc window with radius=5 and increment=5
        # if the point's sa_mask is the same as center point, set it to yellow, else set it to gray
        for j in range(-5, 6, 5):
            for k in range(-5, 6, 5):
                if j == 0 and k == 0:
                    continue
                if cur_anchors[i, 0] + j < 0 or cur_anchors[i, 0] + j >= W \
                        or cur_anchors[i, 1] + k < 0 or cur_anchors[i, 1] + k >= H:
                    continue
                if sa_mask is not None:
                    if sa_mask[cur_anchors[i, 1] + k, cur_anchors[i, 0] + j] == sa_mask[y, x]:
                        plt.scatter(cur_anchors[i, 0] + j, cur_anchors[i, 1] + k, c=color_map['strong_neighbor'],
                                    alpha=alpha, s=scatter_size, linewidths=line_width, edgecolors='black')
                    else:
                        plt.scatter(cur_anchors[i, 0] + j, cur_anchors[i, 1] + k, c=color_map['invalid_neighbor'],
                                    alpha=alpha, s=scatter_size, linewidths=line_width, edgecolors='black')
                else:
                    plt.scatter(cur_anchors[i, 0] + j, cur_anchors[i, 1] + k, c=color_map['strong_neighbor'],
                                alpha=alpha, s=scatter_size, linewidths=line_width, edgecolors='black')

    # cur_anchors[1:, :], the other points,
    # if its sa_mask is the same as center point, set it to red, else set it to gray
    for i in range(1, cur_anchors.shape[0]):
        if sa_mask is not None and sa_mask[cur_anchors[i, 1], cur_anchors[i, 0]] != sa_mask[y, x]:
            continue
        plt.scatter(cur_anchors[i, 0], cur_anchors[i, 1], c=color_map['strong_center'], s=scatter_size, linewidths=line_width, edgecolors='black')


def show_static(x, y, img, anchors_map, anchors, sa_mask=None, sa_mask_color=None):
    cur_anchors = get_valid_anchors(x, y, anchors, anchors_map, sa_mask)
    if cur_anchors is None:
        return
    plt.imshow(img)
    if sa_mask_color is not None and args.show_sa_mask:
        plt.imshow(sa_mask_color, alpha=0.35)

    # show points in image
    show_point(x, y, img.shape[0], img.shape[1], cur_anchors, sa_mask)
    plt.show()


def show_dynamic(img, anchors_map, anchors, sa_mask=None, sa_mask_color=None, init_point=None):
    """
    :param img: the mat of image
    :param anchors_map: the mat of the anchors_map
    :param anchors: the bin array of anchors
    :param sa_mask: the mat of sa_mask
    :param sa_mask_color: the mat of sa_mask_color
    :param init_point: the init point
    :return: void
    """
    fig, ax = plt.subplots()
    plt.imshow(img)
    if sa_mask_color is not None and args.show_sa_mask:
        plt.imshow(sa_mask_color, alpha=0.35)

    if init_point is not None:
        x = init_point[0]
        y = init_point[1]
        cur_anchors = get_valid_anchors(x, y, anchors, anchors_map, sa_mask)
        if cur_anchors is not None:
            show_point(x, y, img.shape[0], img.shape[1], cur_anchors, sa_mask)

    # double click to pick a point
    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        if not event.dblclick:
            return
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))
        x = int(event.xdata)
        y = int(event.ydata)
        print("pick point -> x: {}, y: {}".format(x, y))
        cur_anchors = get_valid_anchors(x, y, anchors, anchors_map, sa_mask)
        if cur_anchors is None:
            return
        plt.clf()
        plt.imshow(img)
        if sa_mask_color is not None and args.show_sa_mask:
            plt.imshow(sa_mask_color, alpha=0.35)
        # show points in image
        show_point(x, y, img.shape[0], img.shape[1], cur_anchors, sa_mask)
        plt.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


# main
if __name__ == '__main__':
    img = None
    sa_mask = None
    anchors_map = None
    anchors = None
    if args.mode == "scan":
        scan_dir = args.scan_dir
        view_index = args.view_index
        # read image with png or jpg
        img_path = os.path.join(scan_dir, "images", "{:08d}.png".format(view_index))
        if not os.path.exists(img_path):
            img_path = os.path.join(scan_dir, "images", "{:08d}.jpg".format(view_index))
        if not os.path.exists(img_path):
            raise FileNotFoundError("image_path not exist: {}".format(img_path))
        anchors_map_path = os.path.join(scan_dir, "ADP", "{:08d}".format(view_index), "anchors_map.bin")
        anchors_path = os.path.join(scan_dir, "ADP", "{:08d}".format(view_index), "anchors.bin")
        sa_mask_path = os.path.join(scan_dir, "sa_masks", "{:08d}.bin".format(view_index))
        # check
        if not os.path.exists(img_path):
            raise FileNotFoundError("img_path not exist: {}".format(img_path))
        if not os.path.exists(anchors_map_path):
            raise FileNotFoundError("anchors_map_path not exist: {}".format(anchors_map_path))
        if not os.path.exists(anchors_path):
            raise FileNotFoundError("anchors_path not exist: {}".format(anchors_path))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        anchors_map = read_anchor_map(anchors_map_path)
        anchors = read_anchor(anchors_path)
        if os.path.exists(sa_mask_path):
            sa_mask = read_sa_mask(sa_mask_path)
    else:
        # check
        if not os.path.exists(args.img_path):
            raise FileNotFoundError("img_path not exist: {}".format(args.img_path))
        if not os.path.exists(args.anchors_map_path):
            raise FileNotFoundError("anchors_map_path not exist: {}".format(args.anchors_map_path))
        if not os.path.exists(args.anchors_path):
            raise FileNotFoundError("anchors_path not exist: {}".format(args.anchors_path))
        img = cv2.imread(args.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        anchors_map = read_anchor_map(args.anchors_map_path)
        anchors = read_anchor(args.anchors_path)
        if args.sa_mask_path is not None:
            if not os.path.exists(args.sa_mask_path):
                raise FileNotFoundError("sa_mask_path not exist: {}".format(args.sa_mask_path))
            sa_mask = read_sa_mask(args.sa_mask_path)
    # show args and files
    print("=================== args info ===================")
    print("mode             : ", args.mode)
    if args.mode == "scan":
        print("scan_dir         : ", args.scan_dir)
        print("view_index       : ", args.view_index)
    else:
        print("img_path         : ", args.img_path)
        print("anchors_map_path : ", args.anchors_map_path)
        print("anchors_path     : ", args.anchors_path)
        if args.sa_mask_path is not None:
            print("sa_mask_path     : ", args.sa_mask_path)
    print("close_pick       : ", args.close_pick)
    print("show_sa_mask     : ", args.show_sa_mask)
    print("init_xy          : ", args.init_xy)
    print("=================== shape info ===================")
    print("img              : ", img.shape)
    print("anchors_map      : ", anchors_map.shape)
    print("anchors          : ", anchors.shape)
    if sa_mask is not None:
        print("sa_mask          : ", sa_mask.shape)
    print("=================================================")
    if sa_mask is not None:
        if sa_mask.shape[0:2] != img.shape[0:2]:
            # resize
            sa_mask = cv2.resize(sa_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            print("resize sa_mask: ", sa_mask.shape)

    # get a sa_mask_color
    sa_mask_color = None
    if sa_mask is not None:
        sa_mask_color = np.random.randint(0, 255, (np.max(sa_mask) + 1, 3), dtype=np.uint8)
        sa_mask_color[0, :] = 0
        sa_mask_color = sa_mask_color[sa_mask]

    # show
    if not args.close_pick:
        show_dynamic(img, anchors_map, anchors, sa_mask, sa_mask_color, init_point=args.init_xy)
    else:
        if args.init_xy is None:
            raise ValueError("init_xy is None")
        x = args.init_xy[0]
        y = args.init_xy[1]
        show_static(x, y, img, anchors_map, anchors, sa_mask, sa_mask_color)
