import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

org_ids_map = np.load('./surf/output_128p_128g/org_ids_128g_rand1.txt.npy')
comp_ids_map = np.load('./surf/output_128p_128g/comp_ids_128g_rand1.txt.npy')

# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:org_ids_map.shape[0], 0:org_ids_map.shape[1]]
print('original data')
print(np.max(org_ids_map[:,:,0]), np.min(org_ids_map[:,:,0]))
print(np.mean(org_ids_map[:,:,0]))
print(np.std(org_ids_map[:,:,0]))

print('compensated data')
print(np.max(comp_ids_map[:,:,0]), np.min(comp_ids_map[:,:,0]))
print(np.mean(comp_ids_map[:,:,0]))
print(np.std(comp_ids_map[:,:,0]))

fig = plt.figure()
ax = fig.gca(projection='3d')
# norm = plt.Normalize(org_ids_map[:,:,0].min(), org_ids_map[:,:,0].max())
# colors = cm.jet(norm(org_ids_map[:,:,0]))
surf = ax.plot_surface(xx, yy, comp_ids_map[:,:,0] ,rstride=1, cstride=1, cmap='RdBu',
                       linewidth=0, antialiased=False)
# surf = ax.plot_surface(xx, yy, comp_vgs_map[:,:,0] , facecolor=colors, shade=False)
# surf.set_facecolor((0,0,0,0))
# surf = ax.plot_wireframe(xx, yy, comp_vgs_map[:,:,0], color='r')
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02e'))
# cbar = fig.colorbar(surf, aspect=20, shrink=0.8)
# cbar.ax.tick_params(labelsize=16)
ax.set_zlim(np.min(org_ids_map[:,:,0]), np.max(org_ids_map[:,:,0]))
ax.view_init(30, 230)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='z', labelsize=8, pad=8)

plt.show()

