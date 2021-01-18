# libraries
import cv2
import numpy as np
from PIL import Image, ImageFilter

# name of jpg image file
# jpg_image_name = "old_square.jpg"

def mura_map_gen(input):
    sub_pattern_size = 8 # pixel [64x64] : 4, [128x128] : 8
    height, width, ch = input.shape
    dim_pat_v = np.array([height, sub_pattern_size, 3])
    cols_v, rows_v, ch_v = dim_pat_v
    pat_v = []
    param_v = []

    dim_pat_h = np.array([sub_pattern_size, width, 3])
    cols_h, rows_h, ch_h = dim_pat_h
    pat_h = []
    param_h = []

    for i in range(cols_v//rows_v):
        pattern = np.zeros(dim_pat_v)
        parameter = (np.random.randint(0, 255), 0.1)
        pat_v.append(pattern)
        param_v.append(parameter)

    for pattern, parameters in zip(pat_v, param_v):
        for h in range(cols_v):
           for w in range(rows_v):
                pattern[h, w, :] = np.random.normal(parameters[0], parameters[1])

    for i in range(rows_h//cols_h):
        pattern = np.zeros(dim_pat_h)
        parameter = (np.random.randint(0, 255), 0.1)
        pat_h.append(pattern)
        param_h.append(parameter)

    for pattern, parameters in zip(pat_h, param_h):
        for h in range(cols_h):
           for w in range(rows_h):
                pattern[h, w, :] = np.random.normal(parameters[0], parameters[1])


    img_v = np.concatenate(pat_v, axis=1)
    h, w, _ = img_v.shape
    img_v = Image.fromarray(np.uint8(img_v))
    img_v = img_v.filter(ImageFilter.GaussianBlur(radius=3))
    # img_v.save("./mura_map/ver_pat_th.png")

    img_h = np.concatenate(pat_h, axis=0)
    img_h = Image.fromarray(np.uint8(img_h))
    img_h = img_h.filter(ImageFilter.GaussianBlur(radius=3))
    # img_h.save("./mura_map/hor_pat_th.png")

    img = np.array(img_h)/2 + np.array(img_v)/2
    img = Image.fromarray(np.uint8(img))
    # img.save("pattern.png")

    ''' Gaussian grid generation'''
    x, y = np.meshgrid(np.linspace(np.random.rand(1)*2-1,np.random.rand(1)*2-1, h),
                       np.linspace(np.random.rand(1)*2-1,np.random.rand(1)*2-1, w))
    # x, y = np.meshgrid(np.linspace(0,1,256), np.linspace(-1,1,256))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 0.7, 0.0
    g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
    g = g/np.max(g)
    gau_pat = np.uint8(g*255)
    gau_pat = Image.fromarray(gau_pat)
    # gau_pat.save("./mura_map/glb_pat_th.png")

    g = np.transpose([g, g, g], (1,2,0))
    out_np = img * g
    out_img = Image.fromarray(np.uint8(out_np))
    # out_img.save("./mura_map/mura_pat_th.png")

    return out_np



input = np.full([128, 128, 3], 64, dtype=np.int)  # for test
mura_map_gen(input)

