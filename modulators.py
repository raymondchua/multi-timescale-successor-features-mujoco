import numpy as np

class NoisySineModulator:
    def __init__(self, period=25000, noise_std=0.05, min_val=0.75, max_val=1.25, phase=0.0, seed=97):
        """
        A noisy sine wave modulator for smooth, periodic scaling values.

        Args:
            period (int): Number of steps per full sine cycle.
            noise_std (float): Standard deviation of Gaussian noise added to sine.
            min_val (float): Minimum value of the scaled output.
            max_val (float): Maximum value of the scaled output.
            phase (float): Phase offset for sine wave (in radians), useful for offsetting mass vs friction.
        """
        self.period = period
        self.noise_std = noise_std
        self.min_val = min_val
        self.max_val = max_val
        self.phase = phase
        self.rng = np.random.default_rng(seed)

    def sample(self, step: int):
        """Return modulated value at a given training step."""
        sine_val = 0.5 * (np.sin(2 * np.pi * step / self.period + self.phase) + 1)
        noisy_val = sine_val + self.rng.normal(0.0, self.noise_std)
        noisy_val = np.clip(noisy_val, 0.0, 1.0)
        scaled_val = self.min_val + (self.max_val - self.min_val) * noisy_val
        return scaled_val

class OUDrift:
    def __init__(self, mu=1.0, theta=0.01, sigma=0.02, low=0.9, high=1.1, x0=None, seed=97):
        """
        Ornstein–Uhlenbeck drift generator.

        Parameters:
            mu : float      - long-term mean
            theta : float   - mean reversion rate (smaller = slower drift)
            sigma : float   - noise magnitude
            x0 : float      - initial value (optional)
        """
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.low = low
        self.high = high
        self.x_prev = np.clip(x0 if x0 is not None else mu, self.low, self.high)
        self.rng = np.random.default_rng(seed)


    def sample(self):
        """
        Returns the next value in the OU drift process.
        """
        epsilon = self.rng.standard_normal()
        x = self.x_prev + self.theta * (self.mu - self.x_prev) + self.sigma * epsilon
        x = np.clip(x, self.low, self.high)
        self.x_prev = x
        return x


class NoisyAPeriodicSineModulator:
    def __init__(
        self,
        period=25000,
        noise_std=0.05,
        min_val=0.75,
        max_val=1.25,
        phase=0.0,
        half_period_jitter=0.3,
        seed=97,
    ):
        """
        A noisy, aperiodic sine modulator where frequency changes
        every half-cycle.

        Args:
            period (int): Average number of steps per full cycle.
            noise_std (float): Std of additive Gaussian noise.
            min_val, max_val (float): Output range.
            phase (float): Initial phase offset (radians).
            half_period_jitter (float): Relative jitter for half-cycle duration.
        """
        self.noise_std = noise_std
        self.min_val = min_val
        self.max_val = max_val
        self.phase = phase

        self.avg_half_period = period / 2
        self.half_period_jitter = half_period_jitter

        self.rng = np.random.default_rng(seed)

        # phase bookkeeping
        self.phi = phase
        self._resample_half_cycle()


    def _resample_half_cycle(self):
        """Sample duration and frequency for the next half-cycle."""
        jitter = 1.0 + self.half_period_jitter * self.rng.standard_normal()
        self.half_period = max(1.0, self.avg_half_period * jitter)

        # angular frequency so that π phase advance == half-cycle
        self.omega = np.pi / self.half_period
        self.phase_remaining = np.pi

    def sample(self):
        """Return modulated value at the current step."""
        dphi = self.omega

        if dphi >= self.phase_remaining:
            # hit half-cycle boundary
            self.phi += self.phase_remaining
            leftover = dphi - self.phase_remaining

            # resample frequency for next half-cycle
            self._resample_half_cycle()

            # consume leftover phase
            self.phi += leftover
            self.phase_remaining -= leftover
        else:
            self.phi += dphi
            self.phase_remaining -= dphi

        # sine in [0,1]
        sine_val = 0.5 * (np.sin(self.phi) + 1.0)

        # add noise
        noisy_val = sine_val + self.rng.normal(0.0, self.noise_std)
        noisy_val = np.clip(noisy_val, 0.0, 1.0)

        # scale
        return self.min_val + (self.max_val - self.min_val) * noisy_val