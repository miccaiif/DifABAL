import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from templates import *

path = 'H:/ABAL2/diffae-master_revision/diffae-master/'
device = 'cuda:0'
conf = ffhq256_autoenc()
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cuda:0')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

data = ImageDataset(path+'imgs_align', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
batch = data[1]['img'][None]

import matplotlib.pyplot as plt
plt.imshow(batch[0].permute([1, 2, 0]) / 2 + 0.5)
plt.show()

cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=250)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ori = (batch + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(xT[0].permute(1, 2, 0).cpu())
plt.show()


pred = model.render(xT, cond, T=20)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ori = (batch + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(pred[0].permute(1, 2, 0).cpu())

plt.show()

# 导入必要的库
import torch
import matplotlib.pyplot as plt
from PIL import Image

# 确保result文件夹存在
if not os.path.exists('result'):
    os.makedirs('result')

# 保存ori图像
ori_img = (ori[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
ori_pil = Image.fromarray(ori_img)
ori_pil.save('result/ori.png')

# 保存xT图像
xT_img = (xT[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
xT_pil = Image.fromarray(xT_img)
xT_pil.save('result/xT.png')

# 保存pred图像
pred_img = (pred[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
pred_pil = Image.fromarray(pred_img)
pred_pil.save('result/pred.png')