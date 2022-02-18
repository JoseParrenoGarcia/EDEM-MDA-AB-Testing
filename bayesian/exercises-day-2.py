import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd
import seaborn as sns
import arviz as az

np.random.seed(123)

# ------------------------------------------------------------------------------------------------------------

x = np.linspace(0, 1, 5000)

beta_no_knowledge = beta.pdf(x, 1, 1)
beta_skeptical = beta.pdf(x, 4, 21)
beta_confident = beta.pdf(x, 32, 162)

plt.figure(0)
plt.plot(x, beta_no_knowledge, color='grey', lw=3, ls='-', label='Sin experiencia')
plt.xlabel('Click rate en la versión A calendario de Skyscanner')
plt.ylabel('Densidad')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(x, beta_no_knowledge, color='grey', lw=3, ls='-', label='Sin experiencia')
plt.plot(x, beta_skeptical, color='orange', lw=3, ls='-', label='Con algo más de información')
plt.xlabel('Click rate en la versión A calendario de Skyscanner')
plt.ylabel('Densidad')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(x, beta_no_knowledge, color='grey', lw=3, ls='-', label='Sin experiencia')
plt.plot(x, beta_skeptical, color='orange', lw=3, ls='-', label='Con algo más de información')
plt.plot(x, beta_confident, color='green', lw=3, ls='-', label='Con mucha información')
plt.xlabel('Click rate en la versión A calendario de Skyscanner')
plt.ylabel('Densidad')
plt.legend()
plt.show()

plt.figure(3)
plt.plot(x, beta_no_knowledge, color='grey', lw=3, ls='-', label='Sin experiencia (alpha=1, beta=1)')
plt.plot(x, beta_skeptical, color='orange', lw=3, ls='-', label='Con algo más de información (alpha=4, beta=21)')
plt.plot(x, beta_confident, color='red', lw=3, ls='-', label='Con mucha información (alpha=32, beta=162)')
plt.xlabel('Click rate en la versión A calendario de Skyscanner')
plt.ylabel('Densidad')
plt.legend(bbox_to_anchor=(1.1, 1.1))
plt.show()

# ------------------------------------------------------------------------------------------------------------
def simulation_results_metrics(data):
    """Method to calculate metrics for the differences between success rates of test and control"""
    return {
        "simulation_mean": np.mean(data),
        "simulation_median": np.median(data),
        "simulation_std": np.std(data),
    }


def simulation_results_intervals(data):
    """
    Method to calculate the lower and upper bounds of different intervals
    ROPE = Region of practical equivalence
    HDI = high density interval
    Confidence_interval = confidence interval
    """
    delta = data
    hdi = 0.95
    percentile = 0.95
    return {
        "lower_hdi": az.hdi(delta, credible_interval=hdi)[0],
        "upper_hdi": az.hdi(delta, credible_interval=hdi)[1],
        "lower_ci": np.quantile(delta, 0 + ((1 - percentile) / 2)),
        "upper_ci": np.quantile(delta, 1 - ((1 - percentile) / 2)),
    }


def simulation_results_probability(simulation_test_samples, simulation_ctrl_samples, num_trials: int):
    """
    Method to calculate the probability that test is better than control
    How many times did test win over control?
    """
    test_wins = sum(simulation_test_samples > simulation_ctrl_samples)
    return test_wins / num_trials

# --------------------------------------------------------------------------------------------

sample_size = [100, 1_000, 10_000, 100_000, 1_000_000]

figure_plot = 4
for n in sample_size:
    nA, nB = np.random.rand(2, n)

    # Simulamos resultados de nuestro experimento: tener unas tasas de exito de 4.99 y 5.17
    tasa_exito_A = 0.0499
    tasa_exito_B = 0.0517

    A_exitos = sum(nA < tasa_exito_A)
    A_fracasos = n - A_exitos
    B_exitos = sum(nB < tasa_exito_B)
    B_fracasos = n - B_exitos

    beta_distribution_A = beta(A_exitos, A_fracasos)
    beta_distribution_B = beta(B_exitos, B_fracasos)

    # Extraemos n_muestras de nuestras distribuciones
    n_muestras = 5_000
    A_muestras = pd.Series(beta_distribution_A.rvs() for _ in range(n_muestras))
    B_muestras = pd.Series(beta_distribution_B.rvs() for _ in range(n_muestras))

    # Para que trabajeis mejor y visualiceis, lo ponemos en un dataframe
    resultados_df = pd.DataFrame({'resultados_A': A_muestras,
                                  'resultados_B': B_muestras})
    resultados_df['B_mejor_que_A'] = resultados_df['resultados_B'] - resultados_df['resultados_A']
    resultados_df['B_mejor_que_A_rel'] = resultados_df['resultados_B']/resultados_df['resultados_A']
    # print(resultados_df.head())

    # Extraccion de resultados
    # Posterior distributions
    A_posterior_parameters = simulation_results_metrics(resultados_df['resultados_A'].to_numpy())
    B_posterior_parameters = simulation_results_metrics(resultados_df['resultados_B'].to_numpy())

    A_confidence_intervals = simulation_results_intervals(resultados_df['resultados_A'].to_numpy())
    B_confidence_intervals = simulation_results_intervals(resultados_df['resultados_B'].to_numpy())

    # Visualización de resultados
    plt.figure(figure_plot)
    fig = sns.kdeplot(resultados_df['resultados_A'], shade=True, label='Resultados A')
    fig = sns.kdeplot(resultados_df['resultados_B'], shade=True, label='Resultados B')
    plt.axvline(x=A_posterior_parameters['simulation_mean'], color='royalblue', linestyle='--', alpha=0.65)
    plt.axvline(x=B_posterior_parameters['simulation_mean'], color='orange', linestyle='--', alpha=0.65)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.1)
    msg1 = f'Sample size: {n}.'
    msg2 = f'True success rate A: {round(100*tasa_exito_A,2)}%. Estimated success rate A: {round(100*A_posterior_parameters["simulation_mean"],2)}+/-{round(100*A_posterior_parameters["simulation_std"],2)}%'
    msg3 = f'True success rate B: {round(100*tasa_exito_B,2)}%. Estimated success rate B: {round(100*B_posterior_parameters["simulation_mean"],2)}+/-{round(100*B_posterior_parameters["simulation_std"],2)}%'
    plt.title(f'{msg1}\n{msg2}\n{msg3}', loc='left')
    plt.xlabel('Click rate')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    figure_plot = figure_plot + 1
# --------------------------------------------------------------------------------------------

def resultados_bayes(A_exitos, A_fracasos, B_exitos, B_fracasos, figure_plot):
    beta_distribution_A = beta(A_exitos, A_fracasos)
    beta_distribution_B = beta(B_exitos, B_fracasos)

    # Extraemos n_muestras de nuestras distribuciones
    n_muestras = 5_000
    A_muestras = pd.Series(beta_distribution_A.rvs() for _ in range(n_muestras))
    B_muestras = pd.Series(beta_distribution_B.rvs() for _ in range(n_muestras))

    # Para que trabajeis mejor y visualiceis, lo ponemos en un dataframe
    resultados_df = pd.DataFrame({'resultados_A': A_muestras,
                                  'resultados_B': B_muestras})
    resultados_df['B_mejor_que_A'] = resultados_df['resultados_B'] - resultados_df['resultados_A']
    resultados_df['B_mejor_que_A_rel'] = resultados_df['resultados_B']/resultados_df['resultados_A']

    # Extraccion de resultados
    # Posterior distributions
    A_posterior_parameters = simulation_results_metrics(resultados_df['resultados_A'].to_numpy())
    B_posterior_parameters = simulation_results_metrics(resultados_df['resultados_B'].to_numpy())

    A_confidence_intervals = simulation_results_intervals(resultados_df['resultados_A'].to_numpy())
    B_confidence_intervals = simulation_results_intervals(resultados_df['resultados_B'].to_numpy())

    # Difference between B / A
    delta_rel_posterior_parameters = simulation_results_metrics(resultados_df['B_mejor_que_A_rel'].to_numpy())
    delta_rel_confidence_intervals = simulation_results_intervals(resultados_df['B_mejor_que_A_rel'].to_numpy())

    # Probabilidad de que B es mejor que A
    probabilidad_B_mejor_que_A = simulation_results_probability(simulation_test_samples=resultados_df['resultados_B'],
                                                                simulation_ctrl_samples=resultados_df['resultados_A'],
                                                                num_trials=n_muestras)

    # Visualización de resultados
    nA = A_exitos + A_fracasos
    nB = B_exitos + B_fracasos
    plt.figure(figure_plot)
    fig = sns.kdeplot(resultados_df['resultados_A'], shade=True, label='Resultados A')
    fig = sns.kdeplot(resultados_df['resultados_B'], shade=True, label='Resultados B')
    plt.axvline(x=A_posterior_parameters['simulation_median'], color='royalblue', linestyle='--', alpha=0.65)
    plt.axvline(x=A_confidence_intervals['lower_hdi'], color='royalblue', alpha=0.35)
    plt.axvline(x=A_confidence_intervals['upper_hdi'], color='royalblue', alpha=0.35)
    plt.axvline(x=B_posterior_parameters['simulation_median'], color='orange', linestyle='--', alpha=0.65)
    plt.axvline(x=B_confidence_intervals['lower_hdi'], color='orange', alpha=0.35)
    plt.axvline(x=B_confidence_intervals['upper_hdi'], color='orange', alpha=0.35)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.1)
    msg1 = f'Sample size A: {nA}, Sample size B: {nB}.'
    msg2 = f'Estimated success rate A: {round(100*A_posterior_parameters["simulation_mean"],2)}+/-{round(100*A_posterior_parameters["simulation_std"],2)}%'
    msg3 = f'Estimated success rate B: {round(100*B_posterior_parameters["simulation_mean"],2)}+/-{round(100*B_posterior_parameters["simulation_std"],2)}%'
    plt.title(f'{msg1}\n{msg2}\n{msg3}', loc='left')
    plt.xlabel('Click rate')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    figure_plot = figure_plot + 1
    plt.figure(figure_plot)
    fig = sns.kdeplot(resultados_df['B_mejor_que_A_rel'], shade=True)
    plt.axvline(x=1, color='lightcoral', linestyle='--', alpha=0.65)
    plt.axvline(x=delta_rel_posterior_parameters['simulation_median'], color='royalblue', linestyle='--', alpha=0.65)
    plt.axvline(x=delta_rel_confidence_intervals['lower_hdi'], color='royalblue', alpha=0.35)
    plt.axvline(x=delta_rel_confidence_intervals['upper_hdi'], color='royalblue', alpha=0.35)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.1)
    msg1 = f'Sample size A: {nA}, Sample size B: {nB}.'
    msg2 = f'Estimated difference of B/A: {round(delta_rel_posterior_parameters["simulation_mean"], 2)}+/-{round(delta_rel_posterior_parameters["simulation_std"], 2)}%'
    msg3 = f'Probability that B is better than A: {round(100 * probabilidad_B_mejor_que_A, 2)}%'
    plt.title(f'{msg1}\n{msg2}\n{msg3}', loc='left')
    plt.xlabel('Click rate de B / Click rate de A')
    plt.ylabel('Density')
    plt.show()

    figure_plot = figure_plot + 1

# Caso 1
tamano_muestra_a = 100_123
A_exitos = 5_000
A_fracasos = tamano_muestra_a - A_exitos

tamano_muestra_b = 100_133
B_exitos = 5_250
B_fracasos = tamano_muestra_b - B_exitos

resultados_bayes(A_exitos, A_fracasos, B_exitos, B_fracasos, figure_plot)

# Caso 2
tamano_muestra_a = 100_123
A_exitos = 5_000
A_fracasos = tamano_muestra_a - A_exitos

tamano_muestra_b = 100_133
B_exitos = 5_175
B_fracasos = tamano_muestra_b - B_exitos

resultados_bayes(A_exitos, A_fracasos, B_exitos, B_fracasos, figure_plot)

