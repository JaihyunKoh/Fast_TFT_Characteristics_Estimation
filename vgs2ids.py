import numpy as np
from scipy import interpolate
from mura_map_gen import mura_map_gen
from gray2vdata import gray2vdata
from PIL import Image

img_path = 'test_img/64g_patch.png'

MAX_IDS = 1.105319569495631e-05

input = Image.open(img_path)
input = np.array(input, dtype=np.uint8)
input = np.full([16, 16, 3], 64, dtype=np.int) # for test

ids_map = np.zeros_like(input)

# prepare the vgs and vth maps
vgs_map = gray2vdata(input)
vth_mura_ptn = mura_map_gen(input) # np uint8 format
vth_map = (vth_mura_ptn/255 * 3) - 1.2 # vth range is from -1.2v ~ 1.8v
mu0_mura_ptn = mura_map_gen(input) # np uint8 format
mu0_map = (mu0_mura_ptn/255 * 3.2) + 0.6 # mu0 range is from 0.6 ~ 3.8v

ids_map = np.zeros_like(vgs_map)

vth_array = np.arange(-1.2, 1.9, 0.1)
mu0_array = np.arange(0.6, 4.0, 0.2)
vth_array_text = ['-1.2', '-1.1', '-1.0', '-0.9', '-0.8', '-0.7', '-0.6', '-0.5', '-0.4', '-0.3', '-0.2',
                '-0.1', '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9',
                '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8']
mu0_array_text = ['0.6', '0.8', '1.0', '1.2', '1.4', '1.6', '1.8', '2.0',
                '2.2', '2.4', '2.6', '2.8', '3.0', '3.2', '3.4', '3.6', '3.8']


for (x,y,z), vgs in np.ndenumerate(vgs_map):
    vth = vth_map[x, y, z]
    mu0 = mu0_map[x, y, z]
    vth_idx = np.argmin(np.abs(vth_array - vth))
    mu0_idx = np.argmin(np.abs(mu0_array - mu0))

    if vth_array[vth_idx] < vth:
        vth1_idx = vth_idx
        vth2_idx = vth_idx + 1
    else:
        vth1_idx = vth_idx - 1
        vth2_idx = vth_idx

    if mu0_array[mu0_idx] < mu0:
        mu01_idx = mu0_idx
        mu02_idx = mu0_idx + 1
    else:
        mu01_idx = mu0_idx - 1
        mu02_idx = mu0_idx

    mu01_start = int(float(mu0_array_text[mu01_idx]) / 2 * 10 - 2)
    mu02_start = int(float(mu0_array_text[mu02_idx]) / 2 * 10 - 2)

    vth1_idx_in_mu01 = int(float(vth_array_text[vth1_idx]) * 10 + 13) + (mu01_start-1) * 31
    vth1_idx_in_mu02 = int(float(vth_array_text[vth1_idx]) * 10 + 13) + (mu02_start-1) * 31
    vth2_idx_in_mu01 = int(float(vth_array_text[vth2_idx]) * 10 + 13) + (mu01_start-1) * 31
    vth2_idx_in_mu02 = int(float(vth_array_text[vth2_idx]) * 10 + 13) + (mu02_start-1) * 31

    mu01_vth1_file = 'data/MU0=' + mu0_array_text[mu01_idx] + '/MU' + str(mu01_start) + '_dc' + str(vth1_idx_in_mu01) + '.csv'
    mu01_vth2_file = 'data/MU0=' + mu0_array_text[mu01_idx] + '/MU' + str(mu01_start) + '_dc' + str(vth2_idx_in_mu01) + '.csv'
    mu02_vth1_file = 'data/MU0=' + mu0_array_text[mu02_idx] + '/MU' + str(mu02_start) + '_dc' + str(vth1_idx_in_mu02) + '.csv'
    mu02_vth2_file = 'data/MU0=' + mu0_array_text[mu02_idx] + '/MU' + str(mu02_start) + '_dc' + str(vth2_idx_in_mu02) + '.csv'

    files = [mu01_vth1_file, mu01_vth2_file, mu02_vth1_file, mu02_vth2_file]
    idses = []


    for file in files:
        data = np.genfromtxt(file, delimiter=',')
        f_vgs_idx = np.argmin(np.abs(data[:,0][1:] - vgs))
        idses.append(data[:,1][1:][f_vgs_idx])

    lut_intep = interpolate.interp2d([mu0_array[mu01_idx], mu0_array[mu02_idx]],
                                     [vth_array[vth1_idx], vth_array[vth2_idx]],
                                     [[idses[0], idses[2]],
                                      [idses[1], idses[3]]])
    ids_map[x, y, z] = lut_intep(mu0, vth)
    print('coordinate: ', x, y, z, '  ids_value: ', ids_map[x, y, z])


'''
NORAML_MU0 = '2.2'
MU0_START_IDX = int(float(NORAML_MU0)/2 * 10 -2)
NORMAL_VTH = '0.8'
NORMAL_VTH_INT = int(float(NORMAL_VTH)*10 +13)
NORMAL_VTH_IDX = str((MU0_START_IDX-1)*31 + NORMAL_VTH_INT)

lut_name = 'data/MU0='+NORAML_MU0+'/MU'+str(MU0_START_IDX)+'_dc'+NORMAL_VTH_IDX+'.csv'
'''
ids_norm = np.clip(ids_map/MAX_IDS, 0, 1)
ids_inv_gamma = np.power(ids_norm, 1/2.2)
ids_gray = np.uint8(ids_inv_gamma*255)
demo_img = np.concatenate([input, ids_gray], axis=1)
demo_img = Image.fromarray(demo_img.astype(np.uint8))
demo_img.save("output/estimated_mura.png")




# in_vth = [-1.2] # range -1.2 ~ 1.8
# in_vgs = [-5] # range -5 ~ 30
#
# out_data = lut_intep(in_vth, in_vgs)
# print(out_data)
#
# print(np.max(lut2d))