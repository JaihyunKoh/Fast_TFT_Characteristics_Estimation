import numpy as np
from mura_map_gen import mura_map_gen
from gray2vdata import gray2vdata
from PIL import Image
from model.models import create_model
from options.test_options import TestOptions
import torch
from scipy import interpolate
import time
import csv

opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
model = create_model(opt)

test_gray = '128'

path_png_k_mura ="./surf/output_128p_{gray}g/mura_k_128p_{gray}g_rand1.png".format(gray=str(test_gray))
path_png_th_mura ="./surf/output_128p_{gray}g/mura_th_128p_{gray}g_rand1.png".format(gray=str(test_gray))
path_png_est_mura = "./surf/output_128p_{gray}g/estimated_mura_128p_{gray}g_rand1.png".format(gray=str(test_gray))
path_png_comp_mura = "./surf/output_128p_{gray}g/estimated_comp_mura_128p_{gray}g_rand1.png".format(gray=str(test_gray))
dump_main = "./surf/output_128p_{gray}g/dump_main_128p_{gray}g_rand1.csv".format(gray=str(test_gray))
dump_ids_non_comp = "./surf/output_128p_{gray}g/dump_ids_non_comp_128p_{gray}g_rand1.csv".format(gray=str(test_gray))
dump_ids_comp = "./surf/output_128p_{gray}g/dump_ids_comp_128p_{gray}g_rand1.csv".format(gray=str(test_gray))
input_vgs_save = "./surf/output_128p_{gray}g/input_vgs_{gray}g_rand1.txt".format(gray=str(test_gray))
comp_vgs_save = "./surf/output_128p_{gray}g/comp_vgs_{gray}g_rand1.txt".format(gray=str(test_gray))
org_ids_save = "./surf/output_128p_{gray}g/org_ids_{gray}g_rand1.txt".format(gray=str(test_gray))
comp_ids_save = "./surf/output_128p_{gray}g/comp_ids_{gray}g_rand1.txt".format(gray=str(test_gray))

img_path = 'test_img/mid_mountain.png'

MAX_IDS = 7.047463454901772e-07 # refer to grat2vdata.py
k_array =[1.20566e-08, 1.53580e-08, 1.86586e-08, 2.19585e-08, 2.52576e-08,
          2.85560e-08, 3.18537e-08, 3.51506e-08, 3.84467e-08, 4.17420e-08,
          4.50365e-08, 4.83303e-08, 5.16232e-08, 5.49152e-08, 5.82065e-08,
          6.14968e-08, 6.47864e-08] #refer to script_mu0_to_k.py

# normal(target) k is 3.84467e-08 : MU0=2.2, k_idx_list[8]

# input = Image.open(img_path)
# input = input.resize((128, 128))
# input = np.array(input, dtype=np.uint8)

# for solid gray simulation
input = np.full([128, 128, 3], int(test_gray), dtype=np.int)  # for gray test

start_time = time.time()

# prepare the vgs and vth maps
vgs_map = gray2vdata(input)

vth_mura_ptn = mura_map_gen(input)  # np uint8 format
# Load predefined mura
# vth_mura_ptn = Image.open('./mura_map/mura_pat_th.png')
# vth_mura_ptn = np.array(vth_mura_ptn, dtype=np.uint8)
# Load predefined mura
vth_map = (vth_mura_ptn / 255 * 3) - 1.2  # vth range is from -1.2v ~ 1.8v

k_mura_ptn = mura_map_gen(input)  # np uint8 format
# Load predefined mura
# k_mura_ptn = Image.open('./mura_map/mura_pat_k.png')
# k_mura_ptn = np.array(k_mura_ptn, dtype=np.uint8)
# Load predefined mura
k_map = (k_mura_ptn / 255 * (k_array[-1] - k_array[0])) + k_array[0]  # mu0 range is from 0.6 ~ 3.8v

mura_vth = Image.fromarray(np.uint8(vth_mura_ptn))
mura_k = Image.fromarray(np.uint8(k_mura_ptn))
mura_vth.save(path_png_th_mura)
mura_k.save(path_png_k_mura)

ids_map_o = np.zeros_like(vgs_map)
ids_map_c = np.zeros_like(vgs_map)

TARGET_GAMMA = 2.2
NORAML_MU0 = '2.2'
NORMAL_K = '3.84467e-08'
MU0_START_IDX = int(float(NORAML_MU0) / 2 * 10 - 2)
NORMAL_VTH = '0.8'
NORMAL_VTH_INT = int(float(NORMAL_VTH) * 10 + 13)
NORMAL_VTH_IDX = str((MU0_START_IDX - 1) * 31 + NORMAL_VTH_INT)

# we use vgs range [index  > 5200 and index < 25000] refer to the code in gray2vdata.py
tgt_vgs_ids = 'data/MU0=' + NORAML_MU0 + '/MU' + str(MU0_START_IDX) + '_dc' + NORMAL_VTH_IDX + '.csv'
tgt_data = np.genfromtxt(tgt_vgs_ids, delimiter=',')
vgs_tgt_array = tgt_data[:, 0][5200:10000] # 0.2V ~ 5.0V
ids_tgt_array = tgt_data[:, 1][5200:10000]

vth_array = np.arange(-1.2, 1.9, 0.1)
mu0_array = np.arange(0.6, 4.0, 0.2)
vth_array_text = ['-1.2', '-1.1', '-1.0', '-0.9', '-0.8', '-0.7', '-0.6', '-0.5', '-0.4', '-0.3', '-0.2',
                  '-0.1', '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9',
                  '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8']
mu0_array_text = ['0.6', '0.8', '1.0', '1.2', '1.4', '1.6', '1.8', '2.0',
                  '2.2', '2.4', '2.6', '2.8', '3.0', '3.2', '3.4', '3.6', '3.8']


''' 1. Vs sequence generator for input of NN based on vth_map and k_map 
    2. Estimate Vth and k using NN model
    3. Calculate compensated vgs map from input vgs map
'''
sample_t = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
comp_vgs = np.zeros_like(vgs_map)
abs_error_vs = 0
abs_error_k = 0

f = open(dump_main, 'w')
filewriter = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
filewriter.writerow(['vgs', 'k_map', 'vth_map', 'vth_from_vs', 'estimated_k', 'estimated_vth', 'comp_vgs'])

for (x, y, z), vth in np.ndenumerate(vth_map):
    if z == 0:
    # calulate nearest index and choose before/after value index for interpolation
        k = k_map[x, y, z]
        vth_idx = np.argmin(np.abs(vth_array - vth))
        k_idx = np.argmin(np.abs(k_array - k))
        if vth_array[vth_idx] < vth:
            vth1_idx = vth_idx
            vth2_idx = vth_idx + 1
        else:
            vth1_idx = vth_idx - 1
            vth2_idx = vth_idx
        if k_array[k_idx] < k:
            k1_idx = k_idx
            k2_idx = k_idx + 1
        else:
            k1_idx = k_idx - 1
            k2_idx = k_idx

        # idexing mu01, mu02, vth1, vth2 & decision of 4 file names for interpolation
        mu01_start = int(float(mu0_array_text[k1_idx]) / 2 * 10 - 2)
        mu02_start = int(float(mu0_array_text[k2_idx]) / 2 * 10 - 2)
        vth1_idx_in_mu01 = int(float(vth_array_text[vth1_idx]) * 10 + 13) + (mu01_start - 1) * 31
        vth1_idx_in_mu02 = int(float(vth_array_text[vth1_idx]) * 10 + 13) + (mu02_start - 1) * 31
        vth2_idx_in_mu01 = int(float(vth_array_text[vth2_idx]) * 10 + 13) + (mu01_start - 1) * 31
        vth2_idx_in_mu02 = int(float(vth_array_text[vth2_idx]) * 10 + 13) + (mu02_start - 1) * 31
        mu01_vth1_file = 'data/MU0=' + mu0_array_text[k1_idx] + '/MU' + str(mu01_start) + '_tran' + str(
            vth1_idx_in_mu01) + '.csv'
        mu01_vth2_file = 'data/MU0=' + mu0_array_text[k1_idx] + '/MU' + str(mu01_start) + '_tran' + str(
            vth2_idx_in_mu01) + '.csv'
        mu02_vth1_file = 'data/MU0=' + mu0_array_text[k2_idx] + '/MU' + str(mu02_start) + '_tran' + str(
            vth1_idx_in_mu02) + '.csv'
        mu02_vth2_file = 'data/MU0=' + mu0_array_text[k2_idx] + '/MU' + str(mu02_start) + '_tran' + str(
            vth2_idx_in_mu02) + '.csv'

        files = [mu01_vth1_file, mu01_vth2_file, mu02_vth1_file, mu02_vth2_file]

        # sampling vs at sample times(sample_t)
        vs_list = []
        for t in sample_t:
            # For single sample time, load 4 files and read vs data points.
            vses = []
            for file in files:
                data = np.genfromtxt(file, delimiter=',')
                vses.append(data[:, 1][1:][t])
            # generate interpolation lut
            lut_intep = interpolate.interp2d([k_array[k1_idx], k_array[k2_idx]],
                                             [vth_array[vth1_idx], vth_array[vth2_idx]],
                                             [[vses[0], vses[2]],
                                              [vses[1], vses[3]]])
            # interpolation and appending
            vs_list.append(lut_intep(k, vth))
        ''' Done in gathering Vs and
            start to estimate Vth and k
        '''
        # sample vs and convert to tensor : vs_list and mu
        vs_list = np.array(vs_list, dtype=np.float32)
        vs_list = torch.from_numpy(vs_list).permute(1, 0) # input
        gt_k = np.array(k, dtype=np.float32)
        gt_k = torch.from_numpy(gt_k).unsqueeze(0) # dummy, not used

        model_input = {'input': vs_list, 'k': gt_k, 'gt': gt_k, 'data_path': mu01_vth1_file}
        model.set_input(model_input)
        model.test()

        # model output : estimated vs
        est_vs_30ms = model.output_th_single
        est_k = model.output_k_single * 8.615843e-09 + 3.814008e-08 # denormalization

        vs_gt = []
        for file in files:
            data = np.genfromtxt(file, delimiter=',')
            vs_gt.append(data[:, 1][1:][3007])
        lut_intep_gt = interpolate.interp2d([k_array[k1_idx], k_array[k2_idx]],
                                            [vth_array[vth1_idx], vth_array[vth2_idx]],
                                            [[vs_gt[0], vs_gt[2]],
                                             [vs_gt[1], vs_gt[3]]])
        gt_vs_30ms = lut_intep_gt(k, vth)
        est_vs_30ms = est_vs_30ms.squeeze(0).cpu().numpy()
        abs_error_vs += abs(est_vs_30ms - gt_vs_30ms)
        est_k = est_k.squeeze(0).cpu().numpy()
        abs_error_k += abs(est_k - gt_k.numpy())
        est_vth = 5 - np.array(est_vs_30ms) # Vdata is 5V

    input_vgs = vgs_map[x, y, z]

    mod_vgs = np.sqrt((float(NORMAL_K)/est_k))*(input_vgs-float(NORMAL_VTH)) + est_vth
    comp_vgs[x, y, z] = mod_vgs

    if (x%10 == 0 and y == 0):
        print('processing... coordinate', x, y, z)
        print('input vgs is %f, modulated vgs is %f' % (input_vgs, mod_vgs))

    line = [input_vgs, k, vth, np.asscalar(5-gt_vs_30ms), np.asscalar(est_k), np.asscalar(est_vth), mod_vgs]
    filewriter.writerow(line)
f.close()

np.save(input_vgs_save, vgs_map)
np.save(comp_vgs_save, comp_vgs)




print('Visual simulation starts!!')
fo = open(dump_ids_non_comp, 'w')
filewriter_o = csv.writer(fo, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
filewriter_o.writerow(['vgs' ,'ids'])

fc = open(dump_ids_comp, 'w')
filewriter_c = csv.writer(fc, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
filewriter_c.writerow(['vgs' ,'ids'])

for (x, y, z), vgs in np.ndenumerate(vgs_map):
    vgs_c = comp_vgs[x, y, z]
    if z == 0:
        vth = vth_map[x, y, z]
        k = k_map[x, y, z]
        vth_idx = np.argmin(np.abs(vth_array - vth))
        k_idx = np.argmin(np.abs(k_array - k))
        if vth_array[vth_idx] < vth:
            vth1_idx = vth_idx
            vth2_idx = vth_idx + 1
        else:
            vth1_idx = vth_idx - 1
            vth2_idx = vth_idx
        if k_array[k_idx] < k:
            k1_idx = k_idx
            k2_idx = k_idx + 1
        else:
            k1_idx = k_idx - 1
            k2_idx = k_idx
        mu01_start = int(float(mu0_array_text[k1_idx]) / 2 * 10 - 2)
        mu02_start = int(float(mu0_array_text[k2_idx]) / 2 * 10 - 2)
        vth1_idx_in_mu01 = int(float(vth_array_text[vth1_idx]) * 10 + 13) + (mu01_start - 1) * 31
        vth1_idx_in_mu02 = int(float(vth_array_text[vth1_idx]) * 10 + 13) + (mu02_start - 1) * 31
        vth2_idx_in_mu01 = int(float(vth_array_text[vth2_idx]) * 10 + 13) + (mu01_start - 1) * 31
        vth2_idx_in_mu02 = int(float(vth_array_text[vth2_idx]) * 10 + 13) + (mu02_start - 1) * 31
        mu01_vth1_file = 'data/MU0=' + mu0_array_text[k1_idx] + '/MU' + str(mu01_start) + '_dc' + str(
            vth1_idx_in_mu01) + '.csv'
        mu01_vth2_file = 'data/MU0=' + mu0_array_text[k1_idx] + '/MU' + str(mu01_start) + '_dc' + str(
            vth2_idx_in_mu01) + '.csv'
        mu02_vth1_file = 'data/MU0=' + mu0_array_text[k2_idx] + '/MU' + str(mu02_start) + '_dc' + str(
            vth1_idx_in_mu02) + '.csv'
        mu02_vth2_file = 'data/MU0=' + mu0_array_text[k2_idx] + '/MU' + str(mu02_start) + '_dc' + str(
            vth2_idx_in_mu02) + '.csv'


        files = [mu01_vth1_file, mu01_vth2_file, mu02_vth1_file, mu02_vth2_file]

        datas = []
        for file in files:
            datas.append(np.genfromtxt(file, delimiter=','))

    idses_o = []
    idses_c = []
    for data in datas:
        f_vgs_o_idx = np.argmin(np.abs(data[:, 0][1:] - vgs))
        f_vgs_c_idx = np.argmin(np.abs(data[:, 0][1:] - vgs_c))
        idses_o.append(data[:, 1][1:][f_vgs_o_idx])
        idses_c.append(data[:, 1][1:][f_vgs_c_idx])

    lut_intep_o = interpolate.interp2d([k_array[k1_idx], k_array[k2_idx]],
                                     [vth_array[vth1_idx], vth_array[vth2_idx]],
                                     [[idses_o[0], idses_o[2]],
                                      [idses_o[1], idses_o[3]]])
    ids_map_o[x, y, z] = lut_intep_o(k, vth)

    lut_intep_c = interpolate.interp2d([k_array[k1_idx], k_array[k2_idx]],
                                       [vth_array[vth1_idx], vth_array[vth2_idx]],
                                       [[idses_c[0], idses_c[2]],
                                        [idses_c[1], idses_c[3]]])

    ids_map_c[x, y, z] = lut_intep_c(k, vth)

    if (x % 10 == 0 and y == 0):
        print('coordinate: ', x, y, z, '  ids_value: ', ids_map_o[x, y, z], ids_map_c[x, y, z])

    filewriter_o.writerow([vgs ,ids_map_o[x, y, z]])
    filewriter_c.writerow([vgs, ids_map_c[x, y, z]])


fo.close()
fc.close()

np.save(org_ids_save, ids_map_o)
np.save(comp_ids_save, ids_map_c)

'''
NORAML_MU0 = '2.2'
MU0_START_IDX = int(float(NORAML_MU0)/2 * 10 -2)
NORMAL_VTH = '0.8'
NORMAL_VTH_INT = int(float(NORMAL_VTH)*10 +13)
NORMAL_VTH_IDX = str((MU0_START_IDX-1)*31 + NORMAL_VTH_INT)

lut_name = 'data/MU0='+NORAML_MU0+'/MU'+str(MU0_START_IDX)+'_dc'+NORMAL_VTH_IDX+'.csv'
'''
ids_o_norm = np.clip(ids_map_o / MAX_IDS, 0, 1)
ids_o_inv_gamma = np.power(ids_o_norm, 1 / 2.2)
ids_o_gray = np.uint8(ids_o_inv_gamma * 255)
demo_img_o = np.concatenate([input, ids_o_gray], axis=1)
demo_img_o = Image.fromarray(demo_img_o.astype(np.uint8))
demo_img_o.save(path_png_est_mura)


ids_c_norm = np.clip(ids_map_c / MAX_IDS, 0, 1)
ids_c_inv_gamma = np.power(ids_c_norm, 1 / 2.2)
ids_c_gray = np.uint8(ids_c_inv_gamma * 255)
demo_img_c = np.concatenate([input, ids_c_gray], axis=1)
demo_img_c = Image.fromarray(demo_img_c.astype(np.uint8))
demo_img_c.save(path_png_comp_mura)

end_time = time.time()

processing_time = end_time - start_time
print("processing time is %fs" % processing_time)
