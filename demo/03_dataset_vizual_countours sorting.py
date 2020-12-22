"""Cell recognition algorithm."""
import numpy as np
import cv2
from imutils import resize

# demo document
dem = dict(
    img_path='demo\\img\\cTDaR_t10335.jpg',
    gt_boxes=np.array([[451,  67, 749, 749]]),
    in_boxes=np.array([[455,  84, 760, 785]]))

dem_wdth = dem['in_boxes'][0][3] - dem['in_boxes'][0][1]
dem_hght = dem['in_boxes'][0][2] - dem['in_boxes'][0][0]
dem_xmin = dem['in_boxes'][0][1]
dem_ymin = dem['in_boxes'][0][0]

dem_image = cv2.imread(dem['img_path'])

# detected table from document
tbl_image = dem_image[dem_ymin: dem_ymin + dem_hght,
                      dem_xmin: dem_xmin + dem_wdth]

# threshold and resize table image
tbl_gray = cv2.cvtColor(tbl_image, cv2.COLOR_BGR2GRAY)
tbl_thresh_bin = cv2.threshold(tbl_gray, 127, 255, cv2.THRESH_BINARY)[1]

R = 2.5
tbl_resized = resize(tbl_thresh_bin, width=int(tbl_image.shape[1] // R))


def get_dividers(img, axis):
    """Return array indicies of white horizontal or vertical lines."""
    blank_lines = np.where(np.all(img == 255, axis=axis))[0]
    filtered_idx = np.where(np.diff(blank_lines) != 1)[0]
    return blank_lines[filtered_idx]


dims = tbl_image.shape[0], tbl_image.shape[1]

# table mask to search for gridlines
tbl_str = np.zeros(dims, np.uint8)
tbl_str = cv2.rectangle(tbl_str, (0, 0), (dims[1] - 1, dims[0] - 1), 255, 1)

for a in [0, 1]:
    dividers = get_dividers(tbl_resized, a)
    start_point = [0, 0]
    end_point = [dims[1], dims[1]]
    for i in dividers:
        i *= R
        start_point[a] = int(i)
        end_point[a] = int(i)
        cv2.line(tbl_str,
                 tuple(start_point),
                 tuple(end_point),
                 255,
                 1)


contours, hierarchy = cv2.findContours(tbl_str,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)


def sort_contours(cnts, method="left-to-right"):
    """Return sorted countours."""
    reverse = False
    k = 0
    if method in ['right-to-left', 'bottom-to-top']:
        reverse = True
    if method in ['top-to-bottom', 'bottom-to-top']:
        k = 1
    b_boxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, b_boxes) = zip(*sorted(zip(cnts, b_boxes),
                                  key=lambda b: b[1][k],
                                  reverse=reverse))
    return (cnts, b_boxes)


contours, boundingBoxes = sort_contours(contours, method='top-to-bottom')

# remove countours of the whole table
bb_filtered = [list(t) for t in boundingBoxes
               if t[2] < dims[1] and t[3] < dims[0]]

# allocate countours in table-like structure
rows = []
columns = []

for i, bb in enumerate(bb_filtered):
    if i == 0:
        columns.append(bb)
        previous = bb
    else:
        if bb[1] < previous[1] + previous[3]/2:
            columns.append(bb)
            previous = bb
            if i == len(bb_filtered) - 1:
                rows.append(columns)
        else:
            rows.append(columns)
            columns = []
            previous = bb
            columns.append(bb)
