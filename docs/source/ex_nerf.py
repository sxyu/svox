import svox
import torch
import matplotlib.pyplot as plt

device = 'cuda:0'

t = svox.N3Tree.load("lego.npz", device=device)
r = svox.VolumeRenderer(t)

# Matrix copied from lego test set image 0
c2w = torch.tensor([[ -0.9999999403953552, 0.0, 0.0, 0.0 ],
                    [ 0.0, -0.7341099977493286, 0.6790305972099304, 2.737260103225708 ],
                    [ 0.0, 0.6790306568145752, 0.7341098785400391, 2.959291696548462 ],
                    [ 0.0, 0.0, 0.0, 1.0 ],
             ], device=device)

with torch.no_grad():
    im = r.render_persp(c2w, height=800, width=800, fx=1111.111).clamp_(0.0, 1.0)
plt.imshow(im.cpu())
plt.show()
