

# Simulation parameters 

K = 1000        # catégories pour éventuelle discrétisation
N_BINS = 600    # nombre de bins pour les logits
N = 2**16       # taille du signal
M = 64          # réalisations
alpha = 0.3     # mix MSE lin / log pour S2
beta = 1.0      # mix MSE lin / log pour flatness
gamma = 0.95    # poids S2 vs flatness
tau = 2         # température Gumbel-Softmax
L = 2350 # np.exp(8)
c1 = 1 / 3       # Hurst
c2 = 0.0025       # intermittence
delta_sigma = 1.0
h_mu = 0.0
h_sigma = 1.0


# Custom parameters
IN_TRAINING = True

