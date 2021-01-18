import numpy as np
import math
from skimage.measure import compare_ssim as ssim
from PIL import Image


def SSIM(img1, img2):
    ssim_val = ssim(img1, img2, multichannel=True)
    return ssim_val


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

gray = "mid_mountain"

img_path = './simple/output_128p_{gray}g/estimated_comp_mura_128p_{gray}g_rand1.png'.format(gray=str(gray))

input = Image.open(img_path)
input = np.array(input, dtype=np.uint8)

tgt = input[:, :128, :]
ref = input[:, 128:, :]

psnr = PSNR(tgt, ref)
ssim = SSIM(tgt, ref)

print(psnr)
print(ssim)