import base64
import io
import logging

import arviz as az
import pymc as pm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore

logging.getLogger("pymc").setLevel(logging.CRITICAL)

class SwimTimeModel:
    def __init__(self, times, target, params):
        self.delta = params.delta
        self.champ_p = params.champ_p
        self.dual_p = params.dual_p

        self.target = target

        self.times = np.array([float(t) for t in times], dtype=np.float64)
        self.log_times = np.log(self.times)

    def prediction_model(self):
        with pm.Model() as model:
            theta = pm.StudentT("theta", mu=np.mean(self.log_times), sigma=0.05, nu=(len(self.log_times) - 1))
            sigma = pm.HalfNormal("sigma", 0.1)
            delta = pm.LogNormal("delta", mu=np.log(self.delta), sigma=0.5)

            p = pm.Beta("p", self.champ_p, self.dual_p)
            z = pm.Bernoulli("z", p, shape=len(self.log_times))

            mu = theta * (1 - delta * z)
            obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=self.log_times)

            trace = pm.sample(draws=1000,
                              tune=300,
                              cores=1,
                              chains=2,
                              target_accept=0.95,
                              progressbar=False,
                              compute_convergence_checks=False)

            # extract posterior samples
            theta_samples = trace.posterior["theta"].values.flatten()
            delta_samples = trace.posterior["delta"].values.flatten()
            sigma_samples = trace.posterior["sigma"].values.flatten()

            meet_types = np.array([0, 1])  # dual, championship
            predicted_times = {
                mt: np.exp(np.random.normal(theta_samples * (1 - delta_samples * mt), sigma_samples))
                for mt in meet_types
            }

            response = {}

            response["championship"] = {
                "q2.5": np.percentile(predicted_times[1], 2.5),
                "q25": np.percentile(predicted_times[1], 25),
                "q50": np.percentile(predicted_times[1], 50),
                "q75": np.percentile(predicted_times[1], 75),
                "q97.5": np.percentile(predicted_times[1], 97.5),
                "target": percentileofscore(predicted_times[1], self.target) if self.target is not None else -1
            }

            response["dual"] = {
                "q2.5": np.percentile(predicted_times[0], 2.5),
                "q25": np.percentile(predicted_times[0], 25),
                "q50": np.percentile(predicted_times[0], 50),
                "q75": np.percentile(predicted_times[0], 75),
                "q97.5": np.percentile(predicted_times[0], 97.5),
                "target": percentileofscore(predicted_times[0], self.target) if self.target is not None else -1
            }

            ppc = pm.sample_posterior_predictive(trace)
            ppc_obs = ppc.posterior_predictive["obs"].values
            ppc_obs = ppc_obs.reshape(-1, ppc_obs.shape[-1])

            residuals = self.log_times - ppc_obs

            response["dist"] = self.plot_distributions(predicted_times)
            response["res"] = self.plot_residuals(ppc_obs.mean(axis=0), residuals)
            response["trace"] = self.plot_parameters(trace)

            return response

    def img_to_base64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)

        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        return img_base64

    def plot_residuals(self, pred_mean, residuals):
        fig, ax = plt.subplots()

        ax.errorbar(
            pred_mean,
            residuals.mean(axis=0),
            yerr=np.std(residuals, axis=0),
            fmt='o'
        )
        ax.axhline(0, color='r', linestyle='--')
        return self.img_to_base64(fig)

    def plot_parameters(self, trace):
        axes = az.plot_trace(trace, var_names=['theta', 'delta', 'sigma', 'p'])
        fig = axes[0, 0].figure
        fig.tight_layout()

        return self.img_to_base64(fig)

    def plot_distributions(self, predicted_times):
        dual_samples = predicted_times[0]
        champ_samples = predicted_times[1]

        fig, ax = plt.subplots()
        sns.kdeplot(dual_samples, ax=ax, label='Dual Meet', color='blue', fill=True, alpha=0.3)
        sns.kdeplot(champ_samples, ax=ax, label='Tapered Meet', color='red', fill=True, alpha=0.3)

        ax.axvline(dual_samples.mean(), color='blue', linestyle='--', label='Dual Mean')
        ax.axvline(champ_samples.mean(), color='red', linestyle='--', label='Tapered Mean')

        ax.set_xlabel('Predicted Time')
        ax.set_ylabel('Density')
        ax.set_title('Predicted Time Distributions by Meet Type')
        ax.legend()

        return self.img_to_base64(fig)
