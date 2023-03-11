"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np
import torch.nn.functional as F
import time


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize
    sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
    sqrt_1m_alphas_cumprod = self.sqrt_1m_alphas_cumprod

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - diffusion[:, None, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

      def discretize_ddim(self, x, t, tprev=None, encode=False):
        """DDPM discretization."""
        timestep = (t * (N - 1) / T).long()
        timestep_prev = (tprev * (N - 1) / T).long()

        score = score_fn(x.float(), t.float())

        # alphas1prev_div_alphas1 = torch.exp(log_diff)
        alphas1 = sqrt_alphas_cumprod[timestep].cuda()[:, None, None, None, None]
        alphas2 = sqrt_1m_alphas_cumprod[timestep].cuda()[:, None, None, None, None]
        alphas1_prev = sqrt_alphas_cumprod[timestep_prev].cuda()[:, None, None, None, None]
        alphas2_prev = sqrt_1m_alphas_cumprod[timestep_prev].cuda()[:, None, None, None, None]
        alphas1prev_div_alphas1 = alphas1_prev.double() / alphas1.double()
        alphas2prev_div_alphas2 = alphas2_prev.double() / alphas2.double()


        x0_pred_scaled = (x.double() - alphas2.double() * score.double())
        use_clip = False
        if use_clip:
          x0_pred_scaled = x0_pred_scaled.clamp(-alphas1[0].squeeze(), alphas1[0].squeeze())
        score_scaled_t = x - x0_pred_scaled
        x0_pred = x0_pred_scaled / alphas1

        x_new = (
          alphas1prev_div_alphas1.double() * x + 
          (-alphas1prev_div_alphas1 + alphas2prev_div_alphas2.double()) * score_scaled_t.double()
        )
        return x_new, x0_pred


      def discretize_conditional_ddpm(self, x, t, tprev=None, condition_func=None, condition=False):
        """DDPM discretization."""
        timestep = (t * (N - 1) / T).long()
        timestep_prev = (tprev * (N - 1) / T).long()

        score = score_fn(x.float(), t.float())

        # alphas1prev_div_alphas1 = torch.exp(log_diff)
        alphas1 = sqrt_alphas_cumprod[timestep].cuda()[:, None, None, None, None]
        alphas2 = sqrt_1m_alphas_cumprod[timestep].cuda()[:, None, None, None, None]
        alphas1_prev = sqrt_alphas_cumprod[timestep_prev].cuda()[:, None, None, None, None]
        alphas1prev_div_alphas1 = alphas1_prev.double() / alphas1.double()

        x0_pred_scaled = (x.double() - alphas2.double() * score.double())
        x0_pred_scaled = x0_pred_scaled.clamp(-alphas1[0].squeeze(), alphas1[0].squeeze())
        x0_pred = x0_pred_scaled / alphas1

        if condition is None:
          condition_update = 0
        else:
          if (t - 0.99).mean() < 1e-3:
            x = x0_pred
          condition_update = condition_func(x.float(), condition)

        x_new = (
          x - alphas1prev_div_alphas1.double() * condition_update
        )
        return x_new, x0_pred


    return RSDE()


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N).cuda()
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.alphas_cumprod_ext = torch.cat([torch.tensor([1.0 - 1e-4]).cuda(), torch.cumprod(self.alphas, dim=0)], dim=0)

    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    self.alphas_cumprod = self.alphas_cumprod
    self.alphas_cumprod_ext = self.alphas_cumprod_ext

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3, 4)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None, None] * x - x
    G = sqrt_beta
    return f, G