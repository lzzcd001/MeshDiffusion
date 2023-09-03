# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from .models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
from . import sde_lib
from .models import utils as mutils

import logging
import tqdm

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps, grid_mask=None, return_traj=False):
    """Create a sampling function.

    Args:
        config: A `ml_collections.ConfigDict` object that contains all configuration information.
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        shape: A sequence of integers representing the expected shape of a single sample.
        inverse_scaler: The inverse data normalizer function.
        eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
            trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    if sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(sde=sde,
                                    shape=shape,
                                    predictor=predictor,
                                    corrector=corrector,
                                    inverse_scaler=inverse_scaler,
                                    snr=config.sampling.snr,
                                    n_steps=config.sampling.n_steps_each,
                                    probability_flow=config.sampling.probability_flow,
                                    continuous=config.training.continuous,
                                    denoise=config.sampling.noise_removal,
                                    eps=eps,
                                    device=config.device,
                                    grid_mask=grid_mask,
                                    return_traj=return_traj)
    elif sampler_name.lower() == 'ddim':
        predictor = get_predictor('ddim')
        sampling_fn = get_ddim_sampler(sde=sde,
                                    shape=shape,
                                    predictor=predictor,
                                    inverse_scaler=inverse_scaler,
                                    n_steps=config.sampling.n_steps_each,
                                    denoise=config.sampling.noise_removal,
                                    eps=eps,
                                    device=config.device,
                                    grid_mask=grid_mask)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

        Args:
            x: A PyTorch tensor representing the current state
            t: A PyTorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None, None] * z
        return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)
        x_mean = (x + beta[:, None, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t):
        if isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t)
        else:
            raise NotImplementedError


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x

@register_predictor(name='ddim')
class DDIMPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)


    def update_fn(self, x, t, tprev=None):
        x, x0_pred = self.rsde.discretize_ddim(x, t, tprev=tprev)
        return x, x0_pred

@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None, None] * noise

        return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

    We include this corrector only for completeness. It was not directly used in our paper.
    """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None, None]

        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                                     n_steps=1, probability_flow=False, continuous=False,
                                     denoise=True, eps=1e-3, device='cuda', grid_mask=None, return_traj=False):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        sde: An `sde_lib.SDE` object representing the forward SDE.
        shape: A sequence of integers. The expected shape of a single sample.
        predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
        corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
        inverse_scaler: The inverse data normalizer.
        snr: A `float` number. The signal-to-noise ratio for configuring correctors.
        n_steps: An integer. The number of corrector steps per predictor update.
        probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
        continuous: `True` indicates that the score model was continuously trained.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def pc_sampler(model, 
            partial=None, partial_mask=None, partial_channel=0, 
            freeze_iters=None):
        """ The PC sampler funciton.

        Args:
            model: A score model.
        Returns:
            Samples, number of function evaluations.
        """
        with torch.no_grad():

            if freeze_iters is None:
                freeze_iters = sde.N + 10 # just some randomly large number greater than sde.N
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)



            def compute_xzero(sde, model, x, t, grid_mask_input):
                timestep_int = (t * (sde.N - 1) / sde.T).long()
                alphas1 = sde.sqrt_alphas_cumprod[timestep_int].cuda()
                alphas2 = sde.sqrt_1m_alphas_cumprod[timestep_int].cuda()
                alphas1_prev = sde.sqrt_alphas_cumprod[timestep_int - 1].cuda()
                alphas2_prev = sde.sqrt_1m_alphas_cumprod[timestep_int - 1].cuda()
                score_pred = model(x, t * torch.ones(shape[0], device=x.device))
                x0_pred_scaled = (x - alphas2 * score_pred)
                x0_pred = x0_pred_scaled / alphas1
                x0_pred = x0_pred.clamp(-1, 1)
                return x0_pred * grid_mask_input
        
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            assert len(x.size()) == 5
            x = x * grid_mask

            traj_buffer = []
        
            if partial is not None:
                assert len(partial.size()) == 5
                t = timesteps[0]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x[:, partial_channel] = partial[:, partial_channel] * grid_mask[:, partial_channel]

                partial_mean, partial_std = sde.marginal_prob(x, vec_t)
                sampled_update = partial_mean[:, partial_channel] + partial_std[:, None, None, None, None] * torch.randn_like(partial_mean[:, partial_channel], device=partial_std.device)
                x[:, partial_channel] = (
                    x[:, partial_channel] * (1 - partial_mask[:, partial_channel]) 
                    + sampled_update[:, partial_channel] * partial_mask[:, partial_channel]
                ) * grid_mask[:, partial_channel]


            if partial is not None:
                x_mean = x
                for i in tqdm.trange(sde.N):
                    t = timesteps[i]
                    vec_t = torch.ones(shape[0], device=t.device) * t

                    x, x_mean = corrector_update_fn(x, vec_t, model=model)
                    x, x_mean = x * grid_mask, x_mean * grid_mask
                    x, x_mean = predictor_update_fn(x, vec_t, model=model)
                    x, x_mean = x * grid_mask, x_mean * grid_mask


                    if i != sde.N - 1 and i < freeze_iters:

                        x[:, partial_channel] = (x[:, partial_channel] * (1 - partial_mask[:, partial_channel]) + partial[:, partial_channel] * partial_mask[:, partial_channel]) * grid_mask[:, partial_channel]
                        x_mean[:, partial_channel] = (x_mean[:, partial_channel] * (1 - partial_mask[:, partial_channel]) + partial[:, partial_channel] * partial_mask[:, partial_channel]) * grid_mask[:, partial_channel]

                        ### add noise to the condition x0_star
                        partial_mean, partial_std = sde.marginal_prob(x, timesteps[i] * torch.ones(shape[0], device=t.device))
                        sampled_update = partial_mean[:, partial_channel] + partial_std[:, None, None, None] * torch.randn_like(partial_mean[:, partial_channel], device=partial_std.device)
                        x[:, partial_channel] = (
                            x[:, partial_channel] * (1 - partial_mask[:, partial_channel]) 
                            + sampled_update * partial_mask[:, partial_channel]
                        ) * grid_mask[:, partial_channel]
                        x_mean[:, partial_channel] = x[:, partial_channel]

            else:

                for i in tqdm.trange(sde.N - 1):
                    t = timesteps[i]

                    vec_t = torch.ones(shape[0], device=t.device) * t
                    x, x_mean = corrector_update_fn(x, vec_t, model=model)
                    x, x_mean = x * grid_mask, x_mean * grid_mask
                    x, x_mean = predictor_update_fn(x, vec_t, model=model)
                    x, x_mean = x * grid_mask, x_mean * grid_mask

                    if return_traj and i >= 700 and i % 10 == 0:
                        traj_buffer.append(compute_xzero(sde, model, x, t, grid_mask))

            if return_traj:
                return traj_buffer, sde.N * (n_steps + 1)
            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

    return pc_sampler

def ddim_predictor_update_fn(x, t, tprev, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    assert not continuous
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=False, std_scale=False)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t, tprev)

def get_ddim_sampler(sde, shape, predictor, inverse_scaler, n_steps=1,
                    denoise=False, eps=1e-3, device='cuda', grid_mask=None):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        shape: A sequence of integers. The expected shape of a single sample.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """

    predictor_update_fn = functools.partial(ddim_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=False,
                                            continuous=False)

    def ddim_sampler(model, schedule='quad', num_steps=100, x0=None,
            partial=None, partial_mask=None, partial_channel=0):
        """ The PC sampler funciton.

        Args:
            model: A score model.
        Returns:
            Samples, number of function evaluations.
        """
        with torch.no_grad():
            if x0 is not None:
                x = x0 * grid_mask
            else:
                # Initial sample
                x = sde.prior_sampling(shape).to(device)
                x = x * grid_mask

            if partial is not None:
                x[:, partial_channel] = x[:, partial_channel] * (1 - partial_mask) + partial * partial_mask

            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            if schedule == 'uniform':
                skip = sde.N // num_steps
                seq = range(0, sde.N, skip)
            elif schedule == 'quad':
                seq = (
                    np.linspace(
                        0, np.sqrt(sde.N * 0.8), 100
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]

            timesteps = torch.tensor(seq) / sde.N

            for i in tqdm.tqdm(reversed(range(1, len(timesteps)))):
                t = timesteps[i]
                tprev = timesteps[i - 1]
                vec_t = torch.ones(shape[0], device=t.device) * t
                vec_tprev = torch.ones(shape[0], device=t.device) * tprev
                x, x0_pred = predictor_update_fn(x, vec_t, model=model, tprev=vec_tprev)
                x, x0_pred = x * grid_mask, x0_pred * grid_mask
                if partial is not None:
                    x[:, partial_channel] = x[:, partial_channel] * (1 - partial_mask) + partial * partial_mask
                    x0_pred[:, partial_channel] = x0_pred[:, partial_channel] * (1 - partial_mask) + partial * partial_mask

            return inverse_scaler(x0_pred * grid_mask if (denoise and not encode) else x * grid_mask), sde.N * (n_steps + 1)
    return ddim_sampler
