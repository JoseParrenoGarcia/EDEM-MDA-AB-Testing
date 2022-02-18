import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

np.random.seed(123)

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
plt.plot(x, beta_confident, color='green', lw=3, ls='-', label='Con mucha información (alpha=32, beta=162)')
plt.xlabel('Click rate en la versión A calendario de Skyscanner')
plt.ylabel('Densidad')
plt.legend(bbox_to_anchor=(1.1, 1.1))
plt.show()

