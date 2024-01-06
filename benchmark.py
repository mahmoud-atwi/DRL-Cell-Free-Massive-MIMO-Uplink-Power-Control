import copy
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from compute_spectral_efficiency import compute_se


# create class for benchmarking
class Benchmark:
    """
    Benchmark class to evaluate different power allocation algorithms in a wireless communication environment.

    Attributes:
        model (Any): A predictive model used in the benchmark.
        model_env (Any): The environment for the model, containing parameters like number of users (K).
        max_power (float): Maximum power limit for the algorithms.
        prelog_factor (float): Pre-log factor used in spectral efficiency computation.
        lagging_se (bool): Flag to indicate if lagging spectral efficiency is considered.
        include_maxmin (bool): Flag to include max-min power allocation algorithm in the benchmark.
        include_maxprod (bool): Flag to include max-product power allocation algorithm in the benchmark.
        include_sumrate (bool): Flag to include sum-rate power allocation algorithm in the benchmark.
        num_of_iterations (int): Number of iterations for the benchmark.
        [various other attributes for storing results]
        mobility (bool): Flag indicating if user mobility is considered in the environment.
    """

    def __init__(self, **kwargs):
        self.model: Any = kwargs.get('model')
        self.model_env: Any = kwargs.get('env')
        self.max_power: float = kwargs.get('max_power', 100)
        self.prelog_factor: float = kwargs.get('prelog_factor', 1)
        self.lagging_se: bool = kwargs.get('lagging_se', False)
        self.include_maxmin: bool = kwargs.get('include_maxmin', True)
        self.include_maxprod: bool = kwargs.get('include_maxprod', True)
        self.include_sumrate: bool = kwargs.get('include_sumrate', True)
        self.num_of_iterations: int = kwargs.get('num_of_iterations', 100)

        self.model_p = np.zeros((self.model_env.K, self.num_of_iterations))
        self.maxmin_p = np.zeros((self.model_env.K, self.num_of_iterations))
        self.maxprod_p = np.zeros((self.model_env.K, self.num_of_iterations))
        self.sumrate_p = np.zeros((self.model_env.K, self.num_of_iterations))

        self.model_se = np.zeros((self.model_env.K, self.num_of_iterations))
        self.maxmin_se = np.zeros((self.model_env.K, self.num_of_iterations))
        self.maxprod_se = np.zeros((self.model_env.K, self.num_of_iterations))
        self.sumrate_se = np.zeros((self.model_env.K, self.num_of_iterations))

        self.model_signals = np.zeros((self.model_env.K, self.num_of_iterations))
        self.maxmin_signals = np.zeros((self.model_env.K, self.num_of_iterations))
        self.maxprod_signals = np.zeros((self.model_env.K, self.num_of_iterations))
        self.sumrate_signals = np.zeros((self.model_env.K, self.num_of_iterations))

        self.model_interferences = np.zeros((self.model_env.K, self.model_env.K, self.num_of_iterations))
        self.maxmin_interferences = np.zeros((self.model_env.K, self.model_env.K, self.num_of_iterations))
        self.maxprod_interferences = np.zeros((self.model_env.K, self.model_env.K, self.num_of_iterations))
        self.sumrate_interferences = np.zeros((self.model_env.K, self.model_env.K, self.num_of_iterations))

        self.obs, self.info = self.model_env.reset()
        self.signal = self.info['init_signal']
        self.interference = self.info['init_interference']

        self.maxmin_env = copy.deepcopy(self.model_env)
        self.maxprod_env = copy.deepcopy(self.model_env)
        self.sumrate_env = copy.deepcopy(self.model_env)

        self.maxmin_signal = copy.deepcopy(self.signal)
        self.maxprod_signal = copy.deepcopy(self.signal)
        self.sumrate_signal = copy.deepcopy(self.signal)

        self.maxmin_interference = copy.deepcopy(self.interference)
        self.maxprod_interference = copy.deepcopy(self.interference)
        self.sumrate_interference = copy.deepcopy(self.interference)

        self.mobility: bool = self.model_env.UEs_mobility

    def run(self, show_progress: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Executes the benchmark for the specified number of iterations and returns the results as Pandas DataFrames.

        Args:
            show_progress (bool): If True, shows a progress bar during execution.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing the results of the benchmark as Pandas DataFrames.
        """

        iterator = tqdm(range(self.num_of_iterations), desc="Running Benchmark") if show_progress else (
            range(self.num_of_iterations))

        for i in iterator:
            action, _ = self.model.predict(self.obs, deterministic=True)
            self.obs, _, _, _, self.info = self.model_env.step(action)
            self.model_se[:, i] = compute_se(self.info['signal'], self.info['interference'],
                                             self.info['predicted_power'], self.prelog_factor)
            self.model_p[:, i] = self.info['predicted_power']
            self.model_signals[:, i] = self.info['signal']
            self.model_interferences[:, :, i] = self.info['interference']

            if self.mobility:
                new_ues_positions = self.info['ues_positions']
            else:
                new_ues_positions = None

            if self.include_maxmin:
                maxmin_info = self.maxmin_env.maxmin_algo(self.maxmin_signal, self.maxmin_interference, self.max_power,
                                                          new_ues_positions, self.prelog_factor, self.lagging_se, None,
                                                          None)
                self.maxmin_signal = maxmin_info['signal']
                self.maxmin_interference = maxmin_info['interference']

                self.maxmin_se[:, i] = compute_se(self.maxmin_signal, self.maxmin_interference,
                                                  maxmin_info['optimized_power'], self.prelog_factor)

                self.maxmin_p[:, i] = maxmin_info['optimized_power']
                self.maxmin_signals[:, i] = maxmin_info['signal']
                self.maxmin_interferences[:, :, i] = maxmin_info['interference']

            if self.include_maxprod:
                maxprod_info = self.maxprod_env.maxprod_algo(self.maxprod_signal, self.maxprod_interference,
                                                             self.max_power, new_ues_positions, self.prelog_factor,
                                                             self.lagging_se, None, None)
                self.maxprod_signal = maxprod_info['signal']
                self.maxprod_interference = maxprod_info['interference']

                self.maxprod_se[:, i] = compute_se(self.maxprod_signal, self.maxprod_interference,
                                                   maxprod_info['optimized_power'], self.prelog_factor)

                self.maxprod_p[:, i] = maxprod_info['optimized_power']
                self.maxprod_signals[:, i] = maxprod_info['signal']
                self.maxprod_interferences[:, :, i] = maxprod_info['interference']

            if self.include_sumrate:
                sumrate_info = self.sumrate_env.maxsumrate_algo(self.sumrate_signal, self.sumrate_interference,
                                                                self.max_power, new_ues_positions, self.prelog_factor,
                                                                self.lagging_se, None, None)
                self.sumrate_signal = sumrate_info['signal']
                self.sumrate_interference = sumrate_info['interference']

                self.sumrate_se[:, i] = compute_se(self.sumrate_signal, self.sumrate_interference,
                                                   sumrate_info['optimized_power'], self.prelog_factor)

                self.sumrate_p[:, i] = sumrate_info['optimized_power']
                self.sumrate_signals[:, i] = sumrate_info['signal']
                self.sumrate_interferences[:, :, i] = sumrate_info['interference']

        # Convert results to Pandas DataFrames
        results_df = {
            'model_se': pd.DataFrame(self.model_se),
            'model_p': pd.DataFrame(self.model_p),
            'model_signals': pd.DataFrame(self.model_signals),
            'model_interferences': pd.DataFrame(self.model_interferences.reshape(self.model_env.K, -1)),
            'maxmin_se': pd.DataFrame(self.maxmin_se),
            'maxmin_p': pd.DataFrame(self.maxmin_p),
            'maxmin_signals': pd.DataFrame(self.maxmin_signals),
            'maxmin_interferences': pd.DataFrame(self.maxmin_interferences.reshape(self.model_env.K, -1)),
            'maxprod_se': pd.DataFrame(self.maxprod_se),
            'maxprod_p': pd.DataFrame(self.maxprod_p),
            'maxprod_signals': pd.DataFrame(self.maxprod_signals),
            'maxprod_interferences': pd.DataFrame(self.maxprod_interferences.reshape(self.model_env.K, -1)),
            'sumrate_se': pd.DataFrame(self.sumrate_se),
            'sumrate_p': pd.DataFrame(self.sumrate_p),
            'sumrate_signals': pd.DataFrame(self.sumrate_signals),
            'sumrate_interferences': pd.DataFrame(self.sumrate_interferences.reshape(self.model_env.K, -1))
        }

        return results_df


class MultiModelBenchmark:
    """
    Benchmark class to evaluate different models and classical power allocation algorithms .

    Attributes:
        models (Dict[str, Any]): A dictionary of predictive models used in the benchmark, keyed by model names.
        env (Any): The environment for the models, containing parameters like number of users (K).
        max_power (float): Maximum power limit for the algorithms.
        prelog_factor (float): Pre-log factor used in spectral efficiency computation.
        lagging_se (bool): Flag to indicate if lagging spectral efficiency is considered.
        include_maxmin (bool): Flag to include max-min power allocation algorithm in the benchmark.
        include_maxprod (bool): Flag to include max-product power allocation algorithm in the benchmark.
        include_sumrate (bool): Flag to include sum-rate power allocation algorithm in the benchmark.
        num_of_iterations (int): Number of iterations for the benchmark.
        [various other attributes for storing results]
        mobility (bool): Flag indicating if user mobility is considered in the environment.
    """

    def __init__(self, models: Dict[str, Any], env: Any, max_power: float = 100, prelog_factor: float = 1,
                 lagging_se: bool = False, include_maxmin: bool = True, include_maxprod: bool = True,
                 include_sumrate: bool = True, num_of_iterations: int = 100, mobility: bool = False):

        self.models: Dict[str, Any] = models
        self.env: Any = env
        self.max_power: float = max_power
        self.prelog_factor: float = prelog_factor
        self.lagging_se: bool = lagging_se
        self.include_maxmin: bool = include_maxmin
        self.include_maxprod: bool = include_maxprod
        self.include_sumrate: bool = include_sumrate
        self.num_of_iterations: int = num_of_iterations
        self.mobility: bool = mobility

        self.results = {model_name: {
            'SEs': np.zeros((self.env.K, self.num_of_iterations)),
            'POWERS': np.zeros((self.env.K, self.num_of_iterations)),
            'SIGNALS': np.zeros((self.env.K, self.num_of_iterations)),
            'CF_SEs': np.zeros((self.env.K, self.num_of_iterations)),
            'SINRs': np.zeros((self.env.K, self.num_of_iterations)),
            # 'INTERFERENCEs': np.zeros((self.env.K, self.env.K, self.num_of_iterations))
        } for model_name in self.models}

        self.models_env = {model_name: copy.deepcopy(self.env) for model_name in self.models}

        self.maxmin_env = copy.deepcopy(self.env)
        self.maxprod_env = copy.deepcopy(self.env)
        self.sumrate_env = copy.deepcopy(self.env)

        self.MAXMIN_POWERs = np.zeros((self.env.K, self.num_of_iterations))
        self.MAXPROD_POWERs = np.zeros((self.env.K, self.num_of_iterations))
        self.SUMRATE_POWERs = np.zeros((self.env.K, self.num_of_iterations))

        self.MAXMIN_SEs = np.zeros((self.env.K, self.num_of_iterations))
        self.MAXPROD_SEs = np.zeros((self.env.K, self.num_of_iterations))
        self.SUMRATE_SEs = np.zeros((self.env.K, self.num_of_iterations))

        self.MAXMIN_SIGNALs = np.zeros((self.env.K, self.num_of_iterations))
        self.MAXPROD_SIGNALs = np.zeros((self.env.K, self.num_of_iterations))
        self.SUMRATE_SIGNALs = np.zeros((self.env.K, self.num_of_iterations))

        self.MAXMIN_SINRs = np.zeros((self.env.K, self.num_of_iterations))
        self.MAXPROD_SINRs = np.zeros((self.env.K, self.num_of_iterations))
        self.SUMRATE_SINRs = np.zeros((self.env.K, self.num_of_iterations))

        # self.MAXMIN_INTERFERENCEs = np.zeros((self.env.K, self.env.K, self.num_of_iterations))
        # self.MAXPROD_INTERFERENCEs = np.zeros((self.env.K, self.env.K, self.num_of_iterations))
        # self.SUMRATE_INTERFERENCEs = np.zeros((self.env.K, self.env.K, self.num_of_iterations))

        self.MAXMIN_CF_SEs = np.zeros((self.env.K, self.num_of_iterations))
        self.MAXPROD_CF_SEs = np.zeros((self.env.K, self.num_of_iterations))
        self.SUMRATE_CF_SEs = np.zeros((self.env.K, self.num_of_iterations))

        self.init_obs, self.init_info = self.env.reset()
        self.ues_positions = self.env.UEs_positions

        self.obs = {model_name: copy.deepcopy(self.init_obs) for model_name in self.models}

        init_signal = self.init_info['init_signal']
        init_interference = self.init_info['init_interference']

        self.maxmin_signal = copy.deepcopy(init_signal)
        self.maxprod_signal = copy.deepcopy(init_signal)
        self.sumrate_signal = copy.deepcopy(init_signal)

        self.maxmin_interference = copy.deepcopy(init_interference)
        self.maxprod_interference = copy.deepcopy(init_interference)
        self.sumrate_interference = copy.deepcopy(init_interference)

        self.UEs_LOCATIONS = np.zeros((self.env.K, self.num_of_iterations), dtype=np.complex64)

    def run(self, show_progress: bool = True):
        """
        Executes the benchmark for the specified number of iterations and returns the results as Pandas DataFrames.

        Args:
            show_progress (bool): If True, shows a progress bar during execution.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing the results of the benchmark as Pandas DataFrames.
        """

        iterator = tqdm(range(self.num_of_iterations), desc="Running Benchmark") if show_progress else (
            range(self.num_of_iterations))

        for i in iterator:
            if self.mobility:
                self.ues_positions = self.env.update_ue_positions(self.ues_positions)
            else:
                self.ues_positions = None

            for model_name, model in self.models.items():
                action, _ = model.predict(self.obs[model_name], deterministic=True)
                self.obs[model_name], info = self.models_env[model_name].simulate(action=action,
                                                                                  ues_positions=self.ues_positions)
                self.results[model_name]['SEs'][:, i] = compute_se(info['signal'], info['interference'],
                                                                   info['predicted_power'], self.prelog_factor)
                self.results[model_name]['POWERS'][:, i] = info['predicted_power']
                self.results[model_name]['SIGNALS'][:, i] = info['signal']
                self.results[model_name]['CF_SEs'][:, i] = info['cf_spectral_efficiency']
                self.results[model_name]['SINRs'][:, i] = info['sinr']
                # self.results[model_name]['INTERFERENCEs'][:, :, i] = info['interference']

            self.UEs_LOCATIONS[:, i] = copy.deepcopy(self.ues_positions)

            if self.include_maxmin:
                maxmin_info = self.maxmin_env.maxmin_algo(self.maxmin_signal, self.maxmin_interference, self.max_power,
                                                          self.ues_positions, self.prelog_factor, self.lagging_se, None,
                                                          None)

                self.maxmin_signal = maxmin_info['signal']
                self.maxmin_interference = maxmin_info['interference']

                self.MAXMIN_CF_SEs[:, i] = maxmin_info['cf_spectral_efficiency']
                self.MAXMIN_SEs[:, i] = compute_se(self.maxmin_signal, self.maxmin_interference,
                                                   maxmin_info['optimized_power'], self.prelog_factor)

                self.MAXMIN_POWERs[:, i] = maxmin_info['optimized_power']
                self.MAXMIN_SIGNALs[:, i] = maxmin_info['signal']
                self.MAXMIN_SINRs[:, i] = maxmin_info['sinr']
                # self.MAXMIN_INTERFERENCEs[:, :, i] = maxmin_info['interference']

            if self.include_maxprod:
                maxprod_info = self.maxprod_env.maxprod_algo(self.maxprod_signal, self.maxprod_interference,
                                                             self.max_power, self.ues_positions, self.prelog_factor,
                                                             self.lagging_se, None, None)

                self.maxprod_signal = maxprod_info['signal']
                self.maxprod_interference = maxprod_info['interference']

                self.MAXPROD_CF_SEs[:, i] = maxprod_info['cf_spectral_efficiency']
                self.MAXPROD_SEs[:, i] = compute_se(self.maxprod_signal, self.maxprod_interference,
                                                    maxprod_info['optimized_power'], self.prelog_factor)

                self.MAXPROD_POWERs[:, i] = maxprod_info['optimized_power']
                self.MAXPROD_SIGNALs[:, i] = maxprod_info['signal']
                self.MAXPROD_SINRs[:, i] = maxprod_info['sinr']
                # self.MAXPROD_INTERFERENCEs[:, :, i] = maxprod_info['interference']

            if self.include_sumrate:
                sumrate_info = self.sumrate_env.maxsumrate_algo(self.sumrate_signal, self.sumrate_interference,
                                                                self.max_power, self.ues_positions, self.prelog_factor,
                                                                self.lagging_se, None, None)

                self.sumrate_signal = sumrate_info['signal']
                self.sumrate_interference = sumrate_info['interference']

                self.SUMRATE_CF_SEs[:, i] = sumrate_info['cf_spectral_efficiency']
                self.SUMRATE_SEs[:, i] = compute_se(self.sumrate_signal, self.sumrate_interference,
                                                    sumrate_info['optimized_power'], self.prelog_factor)

                self.SUMRATE_POWERs[:, i] = sumrate_info['optimized_power']
                self.SUMRATE_SIGNALs[:, i] = sumrate_info['signal']
                self.SUMRATE_SINRs[:, i] = sumrate_info['sinr']
                # self.SUMRATE_INTERFERENCEs[:, :, i] = sumrate_info['interference']

        # Convert results to Pandas DataFrames
        results_df = {
            'MAXMIN_SEs': pd.DataFrame(self.MAXMIN_SEs),
            'MAXMIN_POWERs': pd.DataFrame(self.MAXMIN_POWERs),
            'MAXMIN_SIGNALs': pd.DataFrame(self.MAXMIN_SIGNALs),
            'MAXMIN_CF_SEs': pd.DataFrame(self.MAXMIN_CF_SEs),
            'MAXMIN_SINRs': pd.DataFrame(self.MAXMIN_SINRs),
            # 'MAXMIN_INTERFERENCEs': pd.DataFrame(self.MAXMIN_INTERFERENCEs.reshape(self.env.K, -1)),
            'MAXPROD_SEs': pd.DataFrame(self.MAXPROD_SEs),
            'MAXPROD_POWERs': pd.DataFrame(self.MAXPROD_POWERs),
            'MAXPROD_SIGNALs': pd.DataFrame(self.MAXPROD_SIGNALs),
            'MAXPROD_CF_SEs': pd.DataFrame(self.MAXPROD_CF_SEs),
            'MAXPROD_SINRs': pd.DataFrame(self.MAXPROD_SINRs),
            # 'MAXPROD_INTERFERENCEs': pd.DataFrame(self.MAXPROD_INTERFERENCEs.reshape(self.env.K, -1)),
            'SUMRATE_SEs': pd.DataFrame(self.SUMRATE_SEs),
            'SUMRATE_POWERs': pd.DataFrame(self.SUMRATE_POWERs),
            'SUMRATE_SIGNALs': pd.DataFrame(self.SUMRATE_SIGNALs),
            'SUMRATE_CF_SEs': pd.DataFrame(self.SUMRATE_CF_SEs),
            'SUMRATE_SINRs': pd.DataFrame(self.SUMRATE_SINRs),
            # 'SUMRATE_INTERFERENCEs': pd.DataFrame(self.SUMRATE_INTERFERENCEs.reshape(self.env.K, -1))
            'UEs_LOCATIONS': pd.DataFrame(self.UEs_LOCATIONS)
        }

        # Add results from predictive models
        for model_name, metrics in self.results.items():
            for metric_name, values in metrics.items():
                df_key = f"{model_name}_{metric_name}"
                results_df[df_key] = pd.DataFrame(values.reshape(self.env.K, -1))

        return results_df
