from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

k_array =[1.20566e-08, 1.53580e-08, 1.86586e-08, 2.19585e-08, 2.52576e-08,
          2.85560e-08, 3.18537e-08, 3.51506e-08, 3.84467e-08, 4.17420e-08,
          4.50365e-08, 4.83303e-08, 5.16232e-08, 5.49152e-08, 5.82065e-08,
          6.14968e-08, 6.47864e-08] #refer to script_mu0_to_k.py

# mura_k = Image.open('./mura_map/mura_pat_k.png')
# mura_k = np.array(mura_k, dtype=np.uint8)
# k_map = (mura_k / 255 * (k_array[-1] - k_array[0])) + k_array[0]
# k_map = k_map[:,:,0]
# k_hm = sns.heatmap(k_map)
# plt.show()
#
# hor_pat_k = Image.open('./mura_map/hor_pat_k.png')
# hor_pat_k = np.array(hor_pat_k, dtype=np.uint8)
# hor_pat_k = hor_pat_k[:,:,0]
# hor_k_hm = sns.heatmap(hor_pat_k)
# plt.show()
#
# ver_pat_k = Image.open('./mura_map/ver_pat_k.png')
# ver_pat_k = np.array(ver_pat_k, dtype=np.uint8)
# ver_pat_k = ver_pat_k[:,:,0]
# ver_k_hm = sns.heatmap(ver_pat_k)
# plt.show()
#
# glb_pat_k = Image.open('./mura_map/glb_pat_k.png')
# glb_pat_k = np.array(glb_pat_k, dtype=np.uint8)
# # glb_pat_k = glb_pat_k[:,:,0]
# glb_k_hm = sns.heatmap(glb_pat_k)
# plt.show()


# mura_th = Image.open('./mura_map/mura_pat_th.png')
# mura_th= np.array(mura_th, dtype=np.uint8)
# th_map = (mura_th / 255 * 3) - 1.2
# th_map = th_map[:,:,0]
# th_hm = sns.heatmap(th_map)
# plt.show()
#
# hor_pat_th = Image.open('./mura_map/hor_pat_th.png')
# hor_pat_th = np.array(hor_pat_th, dtype=np.uint8)
# hor_pat_th = hor_pat_th[:,:,0]
# hor_th_hm = sns.heatmap(hor_pat_th)
# plt.show()
#
# ver_pat_th = Image.open('./mura_map/ver_pat_th.png')
# ver_pat_th = np.array(ver_pat_th, dtype=np.uint8)
# ver_pat_th = ver_pat_th[:,:,0]
# ver_th_hm = sns.heatmap(ver_pat_th)
# plt.show()
#
# glb_pat_th = Image.open('./mura_map/glb_pat_th.png')
# glb_pat_th = np.array(glb_pat_th, dtype=np.uint8)
# # glb_pat_k = glb_pat_k[:,:,0]
# glb_th_hm = sns.heatmap(glb_pat_th)
# plt.show()

input_vgs_map = np.load('./simple/output_128p_192g/input_vgs_192g_pre_define.txt.npy')
comp_vgs_map = np.load('./simple/output_128p_192g/comp_vgs_192g_pre_define.txt.npy')
print(np.min(comp_vgs_map), np.max(comp_vgs_map))
plt.figure(1)
vgs = sns.heatmap(input_vgs_map[:,:,0], vmin=np.min(comp_vgs_map), vmax=np.max(comp_vgs_map), cmap='coolwarm', )
vgs_cbar = vgs.collections[0].colorbar
vgs_cbar.set_ticks([ 12.1, 14.1, 16.1, 18.1, 20.1])
plt.figure(2)
vgs_comp = sns.heatmap(comp_vgs_map[:,:,0], vmin=np.min(comp_vgs_map), vmax=np.max(comp_vgs_map), cmap='coolwarm')
vgs_cbar = vgs_comp.collections[0].colorbar
vgs_cbar.set_ticks([ 12.1, 14.1, 16.1, 18.1, 20.1])

plt.show()
