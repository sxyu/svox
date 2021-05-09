import svox
import torch

device = 'cuda:0'
t = svox.N3Tree(map_location=device, data_format="SH1")

t[0, 0, 0, :-1] = 0.0
t[0, 0, 0, -1:] = 0.5
r = svox.VolumeRenderer(t)

target =  torch.tensor([[0.0, 1.0, 0.5]], device=device)

ray_ori = torch.tensor([[0.1, 0.1, -0.1]], device=device)
ray_dir = torch.tensor([[0.0, 0.0, 1.0]], device=device)
ray = svox.Rays(origins=ray_ori, dirs=ray_dir, viewdirs=ray_dir)

lr = 2.5

print('GRADIENT DESC')

for i in range(20):
    rend = r(ray, cuda=True)
    if i % 2 == 0:
        print(rend.detach()[0].cpu().numpy())
    ((rend - target) ** 2).sum().backward()
    t.data.data -= lr * t.data.grad
    t.zero_grad()

print('Expanding..')
t.expand("SH4")
print(r.data_format)
for i in range(20):
    rend = r(ray, cuda=True)
    if i % 2 == 0:
        print(rend.detach()[0].cpu().numpy())
    ((rend - target) ** 2).sum().backward()
    t.data.data -= lr * t.data.grad
    t.zero_grad()

print('TARGET')
print(target[0].cpu().numpy())
