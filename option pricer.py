import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# from mpl_toolkits.mplot3d import Axes3D # uncomment if you want 3d plots

# params
S = 100    # spot
K = 100    # strike
T = 1.0    # time
r = 0.05   # rate
sigma = 0.2
opt_type = 'call'

def get_d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1, d2

def bs_price(S, K, T, r, sigma, type='call'):
    d1, d2 = get_d1_d2(S, K, T, r, sigma)
    
    if type == 'call':
        p = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        p = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return p

def get_greeks(S, K, T, r, sigma, type='call'):
    d1, d2 = get_d1_d2(S, K, T, r, sigma)
    
    if type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1)
    
    return delta, gamma, vega

def monte_carlo_check(S, K, T, r, sigma, type='call', n=100000):
    # simple geometric brownian motion
    z = np.random.standard_normal(n)
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
    
    if type == 'call':
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
        
    return np.exp(-r*T) * np.mean(payoff)

def get_implied_vol(market_price, S, K, T, r, type='call'):
    # newton raphson
    v = 0.5 # initial guess
    for i in range(100):
        price = bs_price(S, K, T, r, v, type)
        diff = market_price - price
        
        if abs(diff) < 1e-5:
            return v
            
        _, _, vega = get_greeks(S, K, T, r, v)
        
        if vega == 0: break
        
        v = v + diff/vega
        
    return None

# --- EXECUTION ---

print(f"Pricing {opt_type} option...")
price = bs_price(S, K, T, r, sigma, opt_type)
print("BS Price:", round(price, 4))

mc = monte_carlo_check(S, K, T, r, sigma, opt_type)
print("MC Price:", round(mc, 4))

delta, gamma, vega = get_greeks(S, K, T, r, sigma, opt_type)
print(f"Greeks -> Delta: {delta:.3f}, Gamma: {gamma:.3f}, Vega: {vega:.3f}")

# check implied vol logic
dummy_price = 12.0
iv = get_implied_vol(dummy_price, S, K, T, r)
print(f"IV for price ${dummy_price}: {iv:.2%}")

# PLOTTING SURFACE (Delta)
# S_range = np.linspace(50, 150, 50)
# T_range = np.linspace(0.1, 2, 50)
# X, Y = np.meshgrid(S_range, T_range)
# Z = np.zeros_like(X)

# for i in range(len(S_range)):
#     for j in range(len(T_range)):
#         d, _, _ = get_greeks(X[j,i], K, Y[j,i], r, sigma)
#         Z[j,i] = d

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis')
# ax.set_xlabel('Spot')
# ax.set_ylabel('Time')
# plt.show()
