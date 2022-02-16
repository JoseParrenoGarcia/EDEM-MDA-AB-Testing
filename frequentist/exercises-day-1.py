from frequentist.frequentist_proportions_test import calculate_proportions_ztest

# Caso 1
tamano_muestra_a = 100_123
tamano_muestra_b = 100_133
conversiones_a = 5_000
conversiones_b = 5_250
alpha = 0.05

print('-'*30)
print('Resultados caso 1: ')
print('-'*30)
calculate_proportions_ztest(sample_size_a=tamano_muestra_a, sample_size_b=tamano_muestra_b,
                            successes_a=conversiones_a, successes_b=conversiones_b,
                            alpha=alpha, test_type='two-sided')

# Caso 2
tamano_muestra_a = 100_123
tamano_muestra_b = 100_133
conversiones_a = 5_000
conversiones_b = 5_175
alpha = 0.05

print('')
print('-'*30)
print('Resultados caso 2: ')
print('-'*30)
calculate_proportions_ztest(sample_size_a=tamano_muestra_a, sample_size_b=tamano_muestra_b,
                            successes_a=conversiones_a, successes_b=conversiones_b,
                            alpha=alpha, test_type='two-sided')


