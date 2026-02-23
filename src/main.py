from data import *
from params import *

import os

# Créer un dossier pour les figures
output_dir = "docs/figures"
os.makedirs(output_dir, exist_ok=True)



#######################################
##      PARAMETER OPTIMIZATION      ###
#######################################

c1_opt = torch.tensor(c1, requires_grad=True)
c2_opt = torch.tensor(c2, requires_grad=True)
delta_sigma_opt = torch.tensor(delta_sigma)
L_opt = torch.tensor(L)
h_mu_opt = torch.tensor(h_mu)
h_sigma_opt = torch.tensor(h_sigma, requires_grad=True)

learning_rate = 0.001
num_epochs = 50
optimizer = torch.optim.Adam([c1_opt, c2_opt], lr=learning_rate)

# Cellule 3 : optimisation des paramètres
c1_opt = torch.tensor(c1, requires_grad=True)
c2_opt = torch.tensor(c2, requires_grad=True)
delta_sigma_opt = torch.tensor(delta_sigma)
L_opt = torch.tensor(L)
h_mu_opt = torch.tensor(h_mu)
h_sigma_opt = torch.tensor(h_sigma, requires_grad=True)

learning_rate = 0.001
num_epochs = 50
optimizer = torch.optim.Adam([c1_opt, c2_opt], lr=learning_rate)

# Bruits fixés pour la stabilité
base_noise1 = torch.randn(M, int(N))
base_noise2 = torch.randn(M, int(N))
loss_history = []
print("Début de l'optimisation...")

for epoch in range(num_epochs):
    optimizer.zero_grad()

    base_noise1 = torch.randn(M, int(N))
    base_noise2 = torch.randn(M, int(N))

    noise1 = base_noise1 * h_sigma_opt + h_mu_opt
    noise2 = base_noise2

    MRW = delta_sigma_opt * functions.synthMRWregul_Torch(
        noise1, noise2, int(N), c1_opt, c2_opt, L_opt, epsilon=0.2, win=1
    )
    MRW_reshaped = torch.reshape(MRW, (-1, 2**16))

    moments = functions.analyseIncrsTorchcuda(MRW_reshaped, scales, device='cpu')
    S2_sim = torch.exp(moments[:, 0, :]).mean(dim=0)
    flat_sim = moments[:, 2, :].mean(dim=0)

    mse_lin_S2 = F.mse_loss(S2_sim, S2)
    mse_log_S2 = F.mse_loss(torch.log(S2_sim + 1e-10), torch.log(S2 + 1e-10))
    loss_S2 = alpha * mse_lin_S2 + (1 - alpha) * mse_log_S2

    mse_lin_flat = F.mse_loss(flat_sim, flatness)
    mse_log_flat = F.mse_loss(torch.log(flat_sim + 1e-10), torch.log(flatness + 1e-10))
    loss_flat = beta * mse_lin_flat + (1 - beta) * mse_log_flat

    loss = gamma * loss_S2 + (1 - gamma) * loss_flat * 1000
    
    if IN_TRAINING:
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.6f} | "
            f"c1: {c1_opt.item():.4f} | c2: {c2_opt.item():.4f} | "
            f"delta_sigma: {delta_sigma_opt.item():.6f} | h_sigma: {h_sigma_opt.item():.4f}"
        )

print("\nOptimisation terminée!")
print(
    f"Paramètres finaux: c1={c1_opt.item():.4f}, c2={c2_opt.item():.4f}, "
    f"delta_sigma={delta_sigma_opt.item():.6f}, h_mu={h_mu_opt.item():.4f}, h_sigma={h_sigma_opt.item():.4f}"
)

with torch.no_grad():
    # Génération initiale
    MRW_init = delta_sigma_opt * functions.synthMRWregul_Torch(
        base_noise1, base_noise2, int(N), torch.tensor(c1), torch.tensor(c2), torch.tensor(L), epsilon=1, win=1
    )
    MRW_init_reshaped = torch.reshape(MRW_init, (-1, 2**16))
    moments_init = functions.analyseIncrsTorchcuda(MRW_init_reshaped, scales, device='cpu')
    S2_init = torch.exp(moments_init[:, 0, :]).mean(dim=0)
    flat_init = moments_init[:, 2, :].mean(dim=0)

    # Génération optimisée
    MRW_final = delta_sigma_opt * functions.synthMRWregul_Torch(
        base_noise1, base_noise2, int(N), c1_opt, c2_opt, torch.tensor(L), epsilon=1, win=1
    )
    MRW_final_reshaped = torch.reshape(MRW_final, (-1, 2**16))
    moments_final = functions.analyseIncrsTorchcuda(MRW_final_reshaped, scales, device='cpu')
    S2_final = torch.exp(moments_final[:, 0, :]).mean(dim=0)
    flat_final = moments_final[:, 2, :].mean(dim=0)


# --- S2 linéaire ---
plt.figure(figsize=(10, 6))
plt.plot(scales.numpy(), S2.numpy(), 'o-', label='S2 cible', linewidth=2, color='blue')
plt.plot(scales.numpy(), S2_init.numpy(), '^--', label='S2 simulé', linewidth=2, color='green', alpha=0.7)
plt.plot(scales.numpy(), S2_final.numpy(), 's-', label='S2 optimisé', linewidth=2, color='orange')
plt.xlabel('Échelle')
plt.ylabel('S2')
plt.title('S2 : cible / simulé')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "S2_linear.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- S2 log-log ---
plt.figure(figsize=(10, 6))
plt.loglog(scales.numpy(), S2.numpy(), 'o-', label='S2 cible', linewidth=2, color='blue')
plt.loglog(scales.numpy(), S2_init.numpy(), '^--', label='S2 initial', linewidth=2, color='green', alpha=0.7)
plt.loglog(scales.numpy(), S2_final.numpy(), 's-', label='S2 optimisé', linewidth=2, color='orange')
plt.xlabel('Échelle (log)')
plt.ylabel('S2 (log)')
plt.title('S2 (log-log)')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig(os.path.join(output_dir, "S2_loglog.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- Flatness semilogx ---
plt.figure(figsize=(10, 6))
plt.semilogx(scales.numpy(), flatness.numpy(), 'o-', label='Flatness cible', linewidth=2, color='blue')
plt.semilogx(scales.numpy(), flat_init.numpy(), '^--', label='Flatness initial', linewidth=2, color='green', alpha=0.7)
plt.semilogx(scales.numpy(), flat_final.numpy(), 's-', label='Flatness optimisé', linewidth=2, color='orange')
plt.xlabel('Échelle (log)')
plt.ylabel('Flatness')
plt.title('Flatness : cible / initial')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig(os.path.join(output_dir, "Flatness_semilogx.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- Flatness log-log ---
plt.figure(figsize=(10, 6))
plt.loglog(scales.numpy(), flatness.numpy(), 'o-', label='Flatness cible', linewidth=2, color='blue')
plt.loglog(scales.numpy(), flat_init.numpy(), '^--', label='Flatness initial', linewidth=2, color='green', alpha=0.7)
plt.loglog(scales.numpy(), flat_final.numpy(), 's-', label='Flatness optimisé', linewidth=2, color='orange')
plt.xlabel('Échelle (log)')
plt.ylabel('Flatness (log)')
plt.title('Flatness (log-log)')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig(os.path.join(output_dir, "Flatness_loglog.png"), dpi=300, bbox_inches='tight')
plt.close()