import argparse
import numpy as np
from scipy.stats import truncnorm
# import pandas as pd
import matplotlib.pyplot as plt
import os


def truncated_gaussian(mu=0.3, sigma=0.3, lower=0, upper=1):
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma)

class piecewise_uniform:
    def __init__(self, tau_l, tau_h, beta1=0.01, beta2=0.95):
        self.tau_l = tau_l
        self.tau_h = tau_h
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = 1 - (beta1 + beta2)
        self.norm1 = beta1 / tau_l
        self.norm2 = beta2 / (tau_h - tau_l)
        self.norm3 = self.beta3 / (1 - tau_h)

    def rvs(self):
        while True:
            sample = np.random.rand()
            if sample <= self.tau_l:
                prob = self.norm1
            elif sample <= self.tau_h:
                prob = self.norm2
            else:
                prob = self.norm3
            return sample

def utility_sample(gamma, tau_a, tau_b, y_sample):
    if tau_a <= y_sample <= tau_b:
        return (2 + gamma) * y_sample - 1
    else:
        return 0.0

def get_utility_matrix_sampled(gamma_l, gamma_h, tau_l, tau_h, y_sample):
    U1 = np.array([
        [utility_sample(gamma_l, tau_l, 1, y_sample)/2, 0, utility_sample(gamma_l, tau_h, 1, y_sample)/2, 0],
        [utility_sample(gamma_l, tau_l, 1, y_sample), utility_sample(gamma_h, tau_l, 1, y_sample)/2,
         utility_sample(gamma_l, tau_h, 1, y_sample), utility_sample(gamma_h, tau_h, 1, y_sample)/2],
        [utility_sample(gamma_l, tau_l, tau_h, y_sample) + utility_sample(gamma_l, tau_h, 1, y_sample)/2,
         utility_sample(gamma_h, tau_l, tau_h, y_sample), utility_sample(gamma_l, tau_h, 1, y_sample)/2, 0],
        [utility_sample(gamma_l, tau_l, 1, y_sample),
         utility_sample(gamma_h, tau_l, tau_h, y_sample) + utility_sample(gamma_h, tau_h, 1, y_sample)/2,
         utility_sample(gamma_l, tau_h, 1, y_sample), utility_sample(gamma_h, tau_h, 1, y_sample)/2]
    ])
    return U1

def exponential_weights(T, eta, num_actions, gamma_l, gamma_h, tau_l, tau_h, dist, init_probs='equal'):
    if init_probs == 'equal':
        p1 = np.tile((0.25, 0.25, 0.25, 0.25), (T+1, 1))
        p2 = np.tile((0.25, 0.25, 0.25, 0.25), (T+1, 1))
    elif init_probs == 'normal':
        p1 = np.tile((0.1, 0.5, 0.3, 0.1), (T+1, 1))
        p2 = np.tile((0.1, 0.3, 0.5, 0.1), (T+1, 1))
    else:
        p1 = np.tile((0, 0.9, 0.1, 0), (T+1, 1))
        p2 = np.tile((0, 0.8, 0.2, 0), (T+1, 1))

    for t in range(T):
        y_sample = dist.rvs() if hasattr(dist, 'rvs') else dist.rvs(size=1)[0]
        U1 = get_utility_matrix_sampled(gamma_l, gamma_h, tau_l, tau_h, y_sample)
        U2 = U1  # Zero-sum

        for k in range(num_actions):
            util1 = np.dot(p2[t], U1[:, k])
            util2 = np.dot(p1[t], U2[:, k])

            p1[t+1, k] = p1[t, k] * np.exp(eta * util1)
            p2[t+1, k] = p2[t, k] * np.exp(eta * util2)

        p1[t+1] /= np.sum(p1[t+1])
        p2[t+1] /= np.sum(p2[t+1])

    return p1, p2

def plot_probabilities_per_player(p1, p2, filename):
    # Ensure output directory exists
    output_dir = "incomplete_info"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i in range(4):
        plt.plot(p1[:, i], label=f'Strategy {i+1}')
    plt.title(r'Bank 1: $p(\theta_i)$')
    plt.xlabel('Time Step')
    plt.ylabel('Probability')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(4):
        plt.plot(p2[:, i], label=f'Strategy {i+1}')
    plt.title(r'Bank 2: $p(\theta_i)$')
    plt.xlabel('Time Step')
    plt.ylabel('Probability')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot to {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution', '-d', choices=['gauss', 'unif'], required=True,
                        help='Choose between truncated Gaussian and piecewise Uniform distribution')
    parser.add_argument('--mu', '-m', type=float, default=None)
    parser.add_argument('--sigma', '-s', type=float, default=None)
    parser.add_argument('--gamma_low', '-gl', type=float, default=0.4)
    parser.add_argument('--gamma_high', '-gh', type=float, default=0.8)
    parser.add_argument('--T', '-T', type=int, default=10000)
    parser.add_argument('--init_probs', '-i', choices=['equal', 'normal', 'poisson'], default='equal',
                        help='Normal sets higher probability to strategies 2, 3')
    args = parser.parse_args()

    tau_l = 1 / (2 + args.gamma_high)
    tau_h = 1 / (2 + args.gamma_low)
    eta = 10 / np.sqrt(args.T)

    if args.distribution == 'gauss':
        dist = truncated_gaussian(mu=args.mu, sigma=args.sigma)
    elif args.distribution == 'unif':
        dist = piecewise_uniform(tau_l, tau_h)

    p1, p2 = exponential_weights(args.T, eta, 4, args.gamma_low, args.gamma_high, tau_l, tau_h, dist, init_probs=args.init_probs)

    if args.distribution == 'gauss':
        filename = f"strategy_probs_{args.distribution}_{args.mu}_{args.sigma}_{args.init_probs}.png"
    else:
        filename = f"strategy_probs_{args.distribution}_{args.init_probs}.png"

    plot_probabilities_per_player(p1, p2, filename)
