import numpy as np
from PIL import Image

# Parameters
TARGET_GAMMA = 2.2
NORAML_MU0 = '2.2' # target MU0
MU0_START_IDX = int(float(NORAML_MU0)/2 * 10 -2)
NORMAL_VTH = '0.8' # target Vth
NORMAL_VTH_INT = int(float(NORMAL_VTH)*10 +13)
NORMAL_VTH_IDX = str((MU0_START_IDX-1)*31 + NORMAL_VTH_INT)

def gray2vdata(input):
    # read image and normalization to one
    # input = Image.open('64g_patch.png')
    # input = Image.open(img_path)
    # input = np.array(input, dtype=np.uint8)
    # input = np.full_like(input, 255) # for test

    # read out vgs-ids LUT and normalization to one
    lut_name = 'data/MU0='+NORAML_MU0+'/MU'+str(MU0_START_IDX)+'_dc'+NORMAL_VTH_IDX+'.csv'
    print('target (normal) TFT file is'+lut_name)
    data = np.genfromtxt(lut_name, delimiter=',')
    vgs_array = data[:, 0][5200:10000]
    ids_array = data[:, 1][5200:10000]
    # print(ids_array)
    print("Target maximum current(ids) is :", np.max(ids_array))
    ids_array = ids_array/np.max(ids_array)      # normalize to 0
    # print(ids_array)

    # gray to vgs LUT generator : gray to vgs LUT
    bit_array = np.array(range(256))/255
    tag_lum_array = np.power(bit_array, TARGET_GAMMA)
    tag_vgs_array = np.array([])
    for i, lum in enumerate(tag_lum_array):
        idx = np.argmin(np.abs(ids_array - lum))
        tag_vgs_array = np.append(tag_vgs_array, vgs_array[idx])

    # map gray to vgs
    output_vgs = tag_vgs_array[input]
    # print(output_vgs)
    return output_vgs




