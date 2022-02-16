from statsmodels.stats.proportion import proportions_ztest
import numpy as np


def calculate_proportions_ztest(sample_size_a, sample_size_b, successes_a, successes_b,
                                alpha, test_type='two-sided'):

   # check our sample against Ho for Ha != Ho
   successes = np.array([successes_a, successes_b])
   samples = np.array([sample_size_a, sample_size_b])

   # note, no need for a Ho value here - it's derived from the other parameters
   stat, p_value = proportions_ztest(count=successes, nobs=samples, alternative=test_type)

   # report
   print('z_stat: %0.3f, p_value: %0.3f' % (stat, p_value))

   if p_value > alpha:
      print("Fail to reject the null hypothesis - we have nothing else to say")
   else:
      print("Reject the null hypothesis - suggest the alternative hypothesis is true")









