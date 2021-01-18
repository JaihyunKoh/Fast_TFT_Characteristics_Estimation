import numpy as np
from PIL import Image, ImageFilter

# generate vgs array
vth_grid = np.arange(-1.2, 1.9, 0.1)
vth_grid_text = ['-1.2', '-1.1', '-1.0', '-0.9', '-0.8', '-0.7', '-0.6', '-0.5', '-0.4', '-0.3', '-0.2',
                '-0.1', '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9',
                '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8']
vth_vgs_ids_lut = np.empty((35001, 0))
for vth in vth_grid_text:
    file_name = 'data/ids_vgs/Vth='+vth+'V_Id-Vgs data.csv'
    data = np.genfromtxt(file_name, delimiter=',')
    vgs = np.empty([], dtype=np.float16)
    ids = np.empty([], dtype=np.float16)
    for index, data in enumerate(data):
        if index != 0:
            vgs = np.hstack([vgs, data[0]])
            ids = np.vstack([ids, data[1]])
    vgs = vgs[1:]
    ids = ids[1:]
    vth_vgs_ids_lut = np.append(vth_vgs_ids_lut, ids, axis=1)

np.save('data/ids_vgs/ids_lut2d', vth_vgs_ids_lut)
np.save('data/ids_vgs/vgs', vgs)
np.save('data/ids_vgs/vth', vth_grid)

print(vth_vgs_ids_lut)
print(vth_vgs_ids_lut.shape)
print(vgs)
print(vgs.shape)
print(vth_grid)
print(vth_grid.shape)

vth_t_vs_lut = np.empty((3008, 0))
for vth in vth_grid_text:
    file_name = 'data/vsense_time/Vth='+vth+'V.csv'
    data = np.genfromtxt(file_name, delimiter=',')
    time = np.empty([], dtype=np.float16)
    vs = np.empty([], dtype=np.float16)
    for index, data in enumerate(data):
        if index != 0:
            time = np.hstack([time, data[0]])
            vs = np.vstack([vs, data[1]])
    time = time[1:]
    vs = vs[1:]
    vth_t_vs_lut = np.append(vth_t_vs_lut, vs, axis=1)

np.save('data/vsense_time/vs_lut2d', vth_t_vs_lut)
np.save('data/vsense_time/time', time)
np.save('data/vsense_time/vth', vth_grid)

print(vth_t_vs_lut)
print(vth_t_vs_lut.shape)
print(time)
print(time.shape)
print(vth_grid)
print(vth_grid.shape)


