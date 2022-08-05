import numpy as np
from PIL import Image
from ISR.models import RDN


img = Image.open('./data/input/sample/baboon.png')
lr_img = np.array(img)
rdn = RDN(arch_params={'C': 3, 'D':10, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('./weights/sample_weights/rdn-C3-D10-G64-G064-x2/PSNR-driven/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5')
sr_img = rdn.predict(lr_img)
#Image.fromarray(sr_img)
#sr_img = model.predict(image, by_patch_of_size=50)

im = Image.fromarray(sr_img)
im.save("./filename.jpeg")
