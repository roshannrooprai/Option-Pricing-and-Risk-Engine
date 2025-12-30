Python 3.14.2 (v3.14.2:df793163d58, Dec  5 2025, 12:18:06) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
Enter "help" below or click "Help" above for more information.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

class EuropeanOption:
    """
    A Financial Instrument class for pricing European Options.
    
    Methods:
    - Black-Scholes-Merton (Analytical)
    - Monte Carlo Simulation (Numerical Validation)
    - Greeks Calculation (Sensitivity Analysis)
    - Implied Volatility (Newton-Raphson Root Finding)
    """

    def __init__(self, S, K, T, r, sigma, option_type='call'):
        """
        Parameters:
        S : float : Spot Price
        K : float : Strike Price
        T : float : Time to Maturity (years)
        r : float : Risk-free Interest Rate (decimal)
        sigma : float : Volatility (decimal)
        option_type : str : 'call' or 'put'
        """
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.option_type = option_type.lower()

    def _d1_d2(self):
        """Helper to calculate d1 and d2 for BSM."""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price(self):
        """
        Calculates the theoretical price using Black-Scholes-Merton closed-form solution.
        Reference: Hull, Ch. 15.
        """
        d1, d2 = self._d1_d2()
        if self.option_type == 'call':
            price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return price

    def greeks(self):
        """
        Calculates first-order risk metrics (Delta, Vega, Gamma).
        Reference: Hull, Ch. 19.
        """
        d1, d2 = self._d1_d2()
        
        # Delta
        if self.option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
            
        # Gamma (Same for Call & Put)
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        
        # Vega (Same for Call & Put)
        vega = self.S * np.sqrt(self.T) * norm.pdf(d1)
        
        return {'Delta': delta, 'Gamma': gamma, 'Vega': vega}

    def monte_carlo_price(self, simulations=100_000):
        """
        Validates the analytical price using Monte Carlo Simulation.
        Simulates Geometric Brownian Motion (GBM).
        """
        np.random.seed(42) # Reproducibility
        
        # S_T = S_0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        z = np.random.standard_normal(simulations)
        ST = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + 
                             self.sigma * np.sqrt(self.T) * z)
        
        if self.option_type == 'call':
            payoffs = np.maximum(ST - self.K, 0)
        else:
            payoffs = np.maximum(self.K - ST, 0)
            
        # Discount mean payoff back to present
        mc_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return mc_price

    @staticmethod
    def implied_volatility(market_price, S, K, T, r, option_type='call', tol=1e-5, max_iter=100):
        """
        Calculates Implied Volatility using Newton-Raphson Method.
        x_n+1 = x_n - f(x) / f'(x)  <-- where f(x) is Price Error, f'(x) is Vega
        """
        sigma = 0.5 # Initial guess
        for i in range(max_iter):
            opt = EuropeanOption(S, K, T, r, sigma, option_type)
            price = opt.price()
            vega = opt.greeks()['Vega']
            
            diff = market_price - price
            
            if abs(diff) < tol:
                return sigma
            
            if vega < 1e-8: # Avoid division by zero
                break
                
            sigma = sigma + diff / vega # Newton-Raphson step
            
        return np.nan # Failed to converge

    def plot_greeks_surface(self, greek_name='Delta'):
        """
        Visualizes the Greek sensitivity vs Spot Price and Time to Maturity.
        """
        S_range = np.linspace(self.S * 0.5, self.S * 1.5, 50)
        T_range = np.linspace(0.01, self.T, 50)
        S_mesh, T_mesh = np.meshgrid(S_range, T_range)
        
        # Vectorized Greek calculation for plotting
        d1 = (np.log(S_mesh / self.K) + (self.r + 0.5 * self.sigma ** 2) * T_mesh) / (self.sigma * np.sqrt(T_mesh))
        
        Z = np.zeros_like(S_mesh)
        
        if greek_name == 'Delta':
            if self.option_type == 'call':
                Z = norm.cdf(d1)
            else:
                Z = norm.cdf(d1) - 1
        elif greek_name == 'Gamma':
            Z = norm.pdf(d1) / (S_mesh * self.sigma * np.sqrt(T_mesh))
        elif greek_name == 'Vega':
            Z = S_mesh * np.sqrt(T_mesh) * norm.pdf(d1)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(S_mesh, T_mesh, Z, cmap='viridis', edgecolor='none')
        
...         ax.set_xlabel('Spot Price')
...         ax.set_ylabel('Time to Maturity')
...         ax.set_zlabel(greek_name)
...         plt.title(f'{greek_name} Surface for {self.option_type.title()} Option')
...         plt.colorbar(surf)
...         plt.show()
... 
... # --- DEMONSTRATION BLOCK ---
... if __name__ == "__main__":
...     # 1. Instantiate Option
...     opt = EuropeanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call')
...     
...     # 2. Compare Analytical vs Monte Carlo
...     bs_price = opt.price()
...     mc_price = opt.monte_carlo_price(simulations=100_000)
...     
...     print(f"--- PRICING VALIDATION ---")
...     print(f"Black-Scholes Price: ${bs_price:.4f}")
...     print(f"Monte Carlo Price:   ${mc_price:.4f}")
...     print(f"Convergence Error:   {abs(bs_price - mc_price):.4f}")
...     
...     # 3. Show Greeks
...     print(f"\n--- GREEKS ---")
...     for k, v in opt.greeks().items():
...         print(f"{k}: {v:.4f}")
... 
...     # 4. Implied Volatility Check
...     market_price = 12.00 # Suppose market is trading higher than model
...     imp_vol = EuropeanOption.implied_volatility(market_price, 100, 100, 1, 0.05)
...     print(f"\n--- IMPLIED VOLATILITY ---")
...     print(f"Market Price: ${market_price}")
...     print(f"Implied Vol:  {imp_vol:.2%}")
...     
...     # 5. Plot (Uncomment to see)
...     opt.plot_greeks_surface('Delta')
...     # opt.plot_greeks_surface('Gamma')
...     
SyntaxError: multiple statements found while compiling a single statement
