from data import *
from params import *

import os
import torch.distributions as D

# Créer un dossier pour les figures
output_dir = "docs/figures"
os.makedirs(output_dir, exist_ok=True)

#######################################
##      NOISE 1 OPTIMIZATION      ###
#######################################

c1_opt = torch.tensor(c1, requires_grad=True)
c2_opt = torch.tensor(c2, requires_grad=True)
delta_sigma_opt = torch.tensor(delta_sigma)
L_opt = torch.tensor(L)
h_mu_opt = torch.tensor(h_mu)
h_sigma_opt = torch.tensor(h_sigma, requires_grad=True)

learning_rate = 0.01
num_epochs = 50

bin_centers = torch.linspace(-6.0, 6.0, N_BINS)

def load_or_init_logits():
    return torch.tensor(D.Normal(0, 1).log_prob(bin_centers), requires_grad=True)

logits_opt = load_or_init_logits()

temperature = 2.0
smoothness_weight = 0.5

optimizer = torch.optim.Adam([{'params': [logits_opt], 'lr': learning_rate}])
print("Modèle prêt.")

num_epochs = 50
loss_history, c2_history = [], []

print("Démarrage de l'optimisation...")


for epoch in range(num_epochs):
    optimizer.zero_grad()

    uniform_noise = torch.rand(M, int(N), N_BINS)
    gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-9) + 1e-9)

    # 1. Gumbel-Softmax -> noise1
    current_logits = logits_opt.view(1, 1, -1).expand(M, int(N), N_BINS)
    y_soft = F.softmax((current_logits + gumbel_noise) / temperature, dim=-1)
    noise1_raw = torch.sum(y_soft * bin_centers, dim=-1)
    noise1 = (noise1_raw - noise1_raw.mean(dim=1, keepdim=True)) / (noise1_raw.std(dim=1, keepdim=True) + 1e-8)

    # 2. Synthèse MRW
    noise2 = torch.randn(M, int(N)) * h_sigma_opt + h_mu_opt
    MRW = delta_sigma_opt * functions.synthMRWregul_Torch(
        noise1, noise2, int(N), c1_opt, c2_opt, np.exp(8), epsilon=1.0, win=1
        )

    # 3. Loss physique
    MRW_reshaped = torch.reshape(MRW, (-1, 2**16))
    moments = functions.analyseIncrsTorchcuda(MRW_reshaped, scales, device='cpu')
    S2_sim = torch.exp(moments[:, 0, :]).mean(dim=0)
    flat_sim = moments[:, 2, :].mean(dim=0)

    mse_lin_s2 = F.mse_loss(S2_sim, S2)
    mse_log_s2 = F.mse_loss(torch.log(S2_sim + 1e-10), torch.log(S2 + 1e-10))
    loss_S2 = alpha * mse_lin_s2 + (1 - alpha) * mse_log_s2

    mse_lin_flat = F.mse_loss(flat_sim, flatness)
    mse_log_flat = F.mse_loss(torch.log(flat_sim + 1e-10), torch.log(flatness + 1e-10))
    loss_Flat = beta * mse_lin_flat + (1 - beta) * mse_log_flat
    
    loss_phy = gamma * loss_S2 + (1 - gamma) * loss_Flat * 1000

    # Régularisation des logits (lissage)
    diff_logits = logits_opt[1:] - logits_opt[:-1]
    loss_smooth = torch.sum(torch.abs(diff_logits))

    loss = loss_phy + smoothness_weight * loss_smooth
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    c2_history.append(c2_opt.item())

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | "
            f"loss_phy: {loss_phy.item():.4f} | loss_smooth: {loss_smooth.item():.4f} | "
            f"c1: {c1_opt.item():.4f} | c2: {c2_opt.item():.4f} | delta_sigma: {delta_sigma_opt.item():.4f}"
        )

plt.style.use('default')

# Plotting final results

# 1) Loi apprise vs gaussienne
with torch.no_grad():
    x_axis = bin_centers.cpu().numpy()
    dx = x_axis[1] - x_axis[0]
    learned_probs = F.softmax(logits_opt, dim=0).cpu().numpy()
    learned_density = learned_probs / dx
    ref_density = torch.exp(D.Normal(0, 1).log_prob(bin_centers)).cpu().numpy()

    mean_dist = np.sum(x_axis * learned_probs)
    var_dist = np.sum(((x_axis - mean_dist)**2) * learned_probs)
    kurt_dist = np.sum(((x_axis - mean_dist)**4) * learned_probs) / (var_dist**2)
    excess_kurtosis = kurt_dist - 3

plt.figure(figsize=(10, 6))
plt.plot(x_axis, learned_density, label='Distribution apprise (noise2)', color='#d62728', linewidth=2.5)
plt.plot(x_axis, ref_density, '--', label='Gaussienne standard', color='black', alpha=0.5)
plt.fill_between(x_axis, learned_density, alpha=0.1, color='#d62728')
plt.title(f"Loi du bruit (excess kurtosis ≈ {excess_kurtosis:.2f})")
plt.xlabel("Valeur du bruit"); plt.ylabel("Densité de probabilité")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# 2) Vérification physique
print("Génération d'un échantillon de validation...")
with torch.no_grad():
    current_logits = logits_opt.view(1, 1, -1).expand(M, int(N), N_BINS)
    random_uniform = torch.rand(M, int(N), N_BINS)
    random_gumbel = -torch.log(-torch.log(random_uniform + 1e-9) + 1e-9)
    y_soft = F.softmax((current_logits + random_gumbel) / temperature, dim=-1)
    noise1_raw = torch.sum(y_soft * bin_centers, dim=-1)
    noise1_final = (noise1_raw - noise1_raw.mean(dim=1, keepdim=True)) / (noise1_raw.std(dim=1, keepdim=True) + 1e-8)

    noise2 = torch.randn(M, int(N)) * h_sigma_opt + h_mu_opt
    MRW_final = delta_sigma_opt * functions.synthMRWregul_Torch(
        noise1_final, noise2, int(N), c1_opt, c2_opt, np.exp(8), epsilon=1.0, win=1
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

ax2.semilogx(scales_np, flat_target_np, 'o-', label='Cible', color='blue')
ax2.semilogx(scales_np, flat_optim, 'x--', label='Optimisé', color='orange', linewidth=2)
ax2.set_xlabel("Échelle (log)"); ax2.set_ylabel("Flatness"); ax2.set_title("Facteur de flatness")
ax2.legend(); ax2.grid(True, which="both", alpha=0.3)
plt.show()

# 3) Historique
y = len(loss_history)
if y > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(loss_history); ax1.set_title("Convergence de la loss"); ax1.set_yscale("log"); ax1.grid(True)
    ax2.plot(c2_history); ax2.set_title("Évolution de c2"); ax2.grid(True)
    plt.show()
else:
    print("Pas d'historique (modèle chargé).")