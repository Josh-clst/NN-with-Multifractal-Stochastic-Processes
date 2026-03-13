from data import *
from params import *

import os
import torch.distributions as D

bin_centers = torch.linspace(-6.0, 6.0, N_BINS)
temperature = 2
delta_sigma_opt = 1
M_training = 1
M_val = 16
N_BINS = 300

logits_dir = f"params/global/MT{M_training}MV{M_val}BIN{N_BINS}/"
os.makedirs(logits_dir, exist_ok=True)

# Load optimal params from logits_dir
logits_opt_n1 = torch.load(os.path.join(logits_dir, "optimized_logits_n1.pt"))
logits_opt_n2 = torch.load(os.path.join(logits_dir, "optimized_logits_n2.pt"))
c1_opt = torch.load(os.path.join(logits_dir, "optimized_c1.pt"))
c2_opt = torch.load(os.path.join(logits_dir, "optimized_c2.pt"))

M_validation = 32
print("Génération d'un échantillon de validation...")
with torch.no_grad():
    current_logits = logits_opt_n1.view(1, 1, -1).expand(M_validation, int(N), N_BINS)
    random_uniform = torch.rand(M_validation, int(N), N_BINS)
    random_gumbel = -torch.log(-torch.log(random_uniform + 1e-9) + 1e-9)
    y_soft = F.softmax((current_logits + random_gumbel) / temperature, dim=-1)
    noise1_raw = torch.sum(y_soft * bin_centers, dim=-1)
    noise1_final = (noise1_raw - noise1_raw.mean(dim=1, keepdim=True)) / (noise1_raw.std(dim=1, keepdim=True) + 1e-8)

    current_logits = logits_opt_n2.view(1, 1, -1).expand(M_validation, int(N), N_BINS)
    random_uniform = torch.rand(M_validation, int(N), N_BINS)
    random_gumbel = -torch.log(-torch.log(random_uniform + 1e-9) + 1e-9)
    y_soft = F.softmax((current_logits + random_gumbel) / temperature, dim=-1)
    noise2_raw = torch.sum(y_soft * bin_centers, dim=-1)
    noise2_final = (noise2_raw - noise2_raw.mean(dim=1, keepdim=True)) / (noise2_raw.std(dim=1, keepdim=True) + 1e-8)

    MRW_final = delta_sigma_opt * functions.synthMRWregul_Torch(
        noise1_final, noise2_final, int(N), c1_opt, c2_opt, np.exp(8), epsilon=1.0, win=1
    )

    MRW_reshaped = torch.reshape(MRW_final, (-1, N))
    moments_final = functions.analyseIncrsTorchcuda(MRW_reshaped, scales, device='cpu')
    S2_optim = torch.exp(moments_final[:, 0, :]).mean(dim=0).cpu().numpy()
    flat_optim = moments_final[:, 2, :].mean(dim=0).cpu().numpy()

scales_np = scales.cpu().numpy()
s2_target_np = S2.cpu().numpy()
flat_target_np = flatness.cpu().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.loglog(scales_np, s2_target_np, 'o-', label='Cible', color='blue')
ax1.loglog(scales_np, S2_optim, 'x--', label='Optimisé', color='orange', linewidth=2)
ax1.set_xlabel("Échelle (log)"); ax1.set_ylabel("S2 (log)"); ax1.set_title("Spectre de variance (S2)")
ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

ax2.loglog(scales_np, flat_target_np, 'o-', label='Cible', color='blue')
ax2.loglog(scales_np, flat_optim, 'x--', label='Optimisé', color='orange', linewidth=2)
ax2.set_xlabel("Échelle (log)"); ax2.set_ylabel("Flatness"); ax2.set_title("Facteur de flatness")
ax2.legend(); ax2.grid(True, which="both", alpha=0.3)
plt.show()
