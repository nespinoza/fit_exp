import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
sns.set_style("ticks")

import juliet
import utils

########### Data handling and download ###########

# Define where we will be saving the Github 
# repo + downloading the data when updated
homefolder = '/Users/nespinoza/github/'
# Define current working directory:
cwd = os.getcwd()
# Go to homefolder and...
os.chdir(homefolder)
# If repo not already here, clone it:
if not os.path.exists(homefolder+'coronavirus'):
    os.system('git clone https://github.com/itoledor/coronavirus')
else:
    # If repo is present, update it in case more data was uploaded:
    os.system('git pull')
# Come back to CWD:
os.chdir(cwd)

# Read data:
days, infected = utils.read_data(dfolder = homefolder+'coronavirus/data')
# Add one so day zero is not zero (avoid log-infs):
days += 1
# Give days in ascending order (useful for some GP kernels):
idx = np.argsort(days)
days = days[idx]
infected = infected[idx]

# Destroy zero-counts, convert infected to log:
idx = np.where(infected>0)[0]
days = days[idx]
infected = np.log(infected[idx])

##################################################

# Save data, put dummy (small) errorbars on the log-infected:
t, data, errors = {}, {}, {}
t['chile'] = days
data['chile'] = infected
errors['chile'] = np.ones(len(infected))*1e-10

# Fit with juliet. This tool really fits exoplanet data, plus possible linear trends and 
# possible GPs. So we turn the amplitude of the planet data (K) to zero, and fix the parameters 
# of the planetary model (P_p1, t0_p1, ecc_p1, omega_p1) to other random values. The fit, then, 
# effectively fits for a line + a GP (in this case in log-space). mu_chile sets/fits 
# a mean for the whole dataset, sigma_w_chile sets/fits for the "errorbars" of the data. The GP_ parameters 
# are the parameters of the GP; rv_slope and rv_intercept are the parameters of a linear fit in log-space (i.e., 
# an exponential model in not-log-space). We set rv_intercept to zero because we are already fitting for it with 
# mu_chile:
params = ['K_p1', 'P_p1', 't0_p1', 'ecc_p1', 'omega_p1',
          'mu_chile', 'sigma_w_chile', 'GP_sigma_chile', 'GP_alpha0_chile', 'rv_slope','rv_intercept']

dists = ['fixed', 'fixed', 'fixed', 'fixed', 'fixed',
         'normal', 'loguniform', 'loguniform', 'loguniform', 'uniform','fixed']

hyperps = [0., 1., 0., 0., 90.,
          [-100.,100.], [0.001,100.], [0.001,100.], [0.001,100.], [-1e3,1e3],0.]

# Gather the priors in a dictionary:
priors = juliet.utils.generate_priors(params,dists,hyperps)

# Define the dataset:
dataset = juliet.load(priors = priors, t_rv = t, y_rv = data, yerr_rv = errors, 
                      GP_regressors_rv = t, out_folder = 'fit_'+str(int(np.max(days))))

# Fit (ta defines a zero-point for the times --- in our case is 0, but for exoplanet data typically is a random 
# julian date):
results = dataset.fit(n_live_points = 500, ta=0.)

# Now that fit has run, generate a set of model_times, and extrapolate a bit the model:
model_times = np.linspace(np.min(days),np.max(days)+4.,1000)
# Evaluate the model in those times, get mean of sampled models and 68% credibility bands:
model, m_up, m_down = results.rv.evaluate('chile',t = model_times, GPregressors = model_times, return_err = True, all_samples = True)
# Same, to get 95% credibility bands:
model, m_up95, m_down95 = results.rv.evaluate('chile',t = model_times, GPregressors = model_times, return_err = True, alpha = 0.95, all_samples = True)

# Plot data in original non-log space; convert model evaluations above as well:
plt.plot(days,np.exp(infected),'o',color='black',mfc='white')
plt.fill_between(model_times,np.exp(m_down),np.exp(m_up),color='cornflowerblue',alpha=0.5)
plt.fill_between(model_times,np.exp(m_down95),np.exp(m_up95),color='cornflowerblue',alpha=0.5)
plt.plot(model_times,np.exp(model),color='black')
plt.xlabel('Days since March 2nd')
plt.ylabel('Number of infected persons')

# Evaluate extrapolation in 1 and 2 days in the future:
model, m_up, m_down = results.rv.evaluate('chile',t = np.max(days) + np.array([1.,2.]), GPregressors = np.max(days) + np.array([1.,2.]), return_err = True)
model, m_up, m_down = np.exp(model), np.exp(m_up), np.exp(m_down)
plt.title('Prediction for tomorrow: {} +- {}, day after: {} +- {}'.format(int(model[0]),int((m_up[0]-model[0]+model[0]-m_down[0])/2.),
                                                                          int(model[1]),int((m_up[1]-model[0]+model[0]-m_down[1])/2.)))
plt.xlim(np.min(days),np.max(days)+3.)
plt.ylim(-10, np.max(m_up))
# Save plot:
plt.savefig('covid-ch.png')
