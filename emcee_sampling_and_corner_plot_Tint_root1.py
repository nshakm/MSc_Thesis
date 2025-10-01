# %%
import numpy as np
import matplotlib.pyplot as plt
import random
import corner

import pandas as pd
import h5py

from scipy.optimize import brentq
from scipy.interpolate import RegularGridInterpolator
import emcee
import os
from uncertainties import ufloat

# %%
#constants
Z0 = 0.014
Zenv = Z0
Mjup = 318 #M_E
Rjup = 11.2 #R_E
sigma_b = 5.67e-5 # erg/s/cm2/K4
conv_AU_to_RS = 215.032 #AU to RS

#grid file
filename = "my_forward_model_grid_finer_alpha_307.hdf5"
file = h5py.File(filename, "r")

#arrays
CMFs = file["CMF"][()]
# Zenvs = file["Zenv"][()]
masses = file["mass"][()]
Teqpls = file["Teqpl"][()]
k = 0
Tint_array = file["Tint"][()][k:]

#bounds
CMF_min, CMF_max = min(CMFs), max(CMFs)
mass_min, mass_max = min(masses)/Mjup, max(masses)/Mjup
Teq_min, Teq_max = (min(Teqpls), max(Teqpls))
# Tint_min, Tint_max = (min(Tint_array), max(Tint_array))

#datasets
data_set_Rtot = file["Rtot"][()][:, :, :, :]
data_set_Mtot = file["Mtot"][()][:, :, :, :]
data_set_age = file["age"][()][:, :, :, :]
data_set_Rbulk = file["Rbulk"][()][:, :, :, :]
data_set_Tsurf = file["Tsurf"][()][:, :, :, :]
data_set_Zplanet = file["Zplanet"][()][:, :, :, :]

#interpolators
method='pchip'
Rtot_interp = RegularGridInterpolator((CMFs, masses, Teqpls, Tint_array), data_set_Rtot, bounds_error=False, fill_value=None, method=method)
Mtot_interp = RegularGridInterpolator((CMFs, masses, Teqpls, Tint_array), data_set_Mtot, bounds_error=False, fill_value=None, method=method)
age_interp = RegularGridInterpolator((CMFs, masses, Teqpls, Tint_array), data_set_age, bounds_error=False, fill_value=None, method=method)
Rbulk_interp = RegularGridInterpolator((CMFs, masses, Teqpls, Tint_array), data_set_Rbulk, bounds_error=False, fill_value=None, method=method)
Tsurf_interp = RegularGridInterpolator((CMFs, masses, Teqpls, Tint_array), data_set_Tsurf, bounds_error=False, fill_value=None, method=method)
Zplanet_interp = RegularGridInterpolator((CMFs, masses, Teqpls, Tint_array), data_set_Zplanet, bounds_error=False, fill_value=None, method=method)

# %%
#get planet parameters from dataframe
def planet_parameters(dataframe, index):

    planet = dataframe.loc[index]

    #planet's name
    name = planet['pl_name']

    #planet's observed radius (normal distribution)
    rad = planet['pl_rad'] #Rjup
    rad_err1 = planet['pl_rad_err1'] #Rjup
    rad_err2 = np.abs(planet['pl_rad_err2']) #Rjup

    #planet's observed mass (normal distribution)
    mass = planet['pl_mass'] #Mjup
    mass_err1 = planet['pl_mass_err1'] #Mjup
    mass_err2 = np.abs(planet['pl_mass_err2']) #Mjup

    # #star's observed age (uniform distribution)
    # age = planet['st_age'] #Gyr
    # age_upper  = planet['st_age_upper'] #Gyr
    # age_lower = planet['st_age_lower'] #Gyr

    #Dealing with E1, E2 from Task IV -> task4_failed_planets.py
    #star's observed age (uniform distribution)
    age = planet['st_age'] #Gyr
    universe_age = 14 #Gyr, being generous and not using 13.8 Gyr

    age_upper  = planet['st_age_upper'] #Gyr
    #for E2
    if np.isnan(age_upper):
        # age_upper = 0
        # age_upper = age + 1
        age_upper = 1.5*age #age + 0.5*age
    #for E1
    elif age_upper >= universe_age:
        print(f"Warning: age_upper is {age_upper} Gyr for {index}) {name}. Setting to {universe_age} Gyr.")
        age_upper = universe_age

    age_lower = planet['st_age_lower'] #Gyr
    #for E2
    if np.isnan(age_lower):
        # age_lower = 0
        # age_lower = age - 1
        age_lower = 0.5*age #age - 0.5*age
    #for E1
    elif age_lower == 0:
        lower_tol = 0.5 #Gyr
        print(f"Warning: age_lower is {age_lower} Gyr for {index}) {name}. Setting to {lower_tol} Gyr.")
        age_lower = lower_tol


    #planet's observed equilibrium temperature (fixed value)
    Teq = planet['pl_teq'] #K

    #planet's observed metallicity (fixed value)
    Zp = planet['pl_zbulk']
    Zp_err = planet['pl_zbulk_err']
    if np.isnan(Zp_err):
        Zp_err = 0.0

    return name, rad, rad_err1, rad_err2, mass, mass_err1, mass_err2, age, age_upper, age_lower, Teq, Zp, Zp_err


#forward model
def forward_model(CMF, mass, Tint):
    """
    Returns the interpolated values of Rtot, Mtot, and age.
    """

    R_mod = Rtot_interp((CMF, mass*Mjup, Teq, Tint)) #R_J
    M_mod = Mtot_interp((CMF, mass*Mjup, Teq, Tint)) #M_J
    age_mod = age_interp((CMF, mass*Mjup, Teq, Tint)) #Gyr

    return R_mod, M_mod, age_mod


#log-likelihood
def log_likelihood(theta, y, y_err):

    #model parameters from forward model and interpolation
    # CMF_mod, mass_mod, Teq, Tint_mod = theta
    CMF_mod, mass_mod, Tint_mod = theta

    # R_mod, M_mod, age_mod = forward_model(CMF_mod, mass_mod, Teq, Tint_mod)
    R_mod, M_mod, age_mod = forward_model(CMF_mod, mass_mod, Tint_mod)

    #observed data
    R_data, M_data, age_data = y
    
    sigma_Rplus, sigma_Rminus = y_err[:2]
    sigma_Mplus, sigma_Mminus = y_err[2:4]
    sigma_age_upper, sigma_age_lower = y_err[4:]
    if np.isnan(sigma_age_upper):
        sigma_age_upper = 0
    if np.isnan(sigma_age_lower):
        sigma_age_lower = 0
    # sigma_age_plus, sigma_age_minus = y_err[4:]


    if R_mod > R_data:
        sigma_R = sigma_Rplus
    else:
        sigma_R = sigma_Rminus

    if M_mod > M_data:
        sigma_M = sigma_Mplus
    else:
        sigma_M = sigma_Mminus

    if age_mod > age_data:
        sigma_age = sigma_age_upper - age_data
        # sigma_age = sigma_age_plus
    else:
        sigma_age = age_data - sigma_age_lower
        # sigma_age = sigma_age_minus

    #likelihood
    try:
        log_L = -0.5 * ((R_mod - R_data)**2 / sigma_R**2 + 
                        (M_mod - M_data)**2 / sigma_M**2 + 
                        (age_mod - age_data)**2 / sigma_age**2)
    except Exception as e:
        print(f"Error in log_likelihood calculation -> {e}")
        print(f"CMF_mod: {CMF_mod}, mass_mod: {mass_mod}, Teq: {Teq}, Tint_mod: {Tint_mod}")
        print(f"R_mod: {R_mod}, M_mod: {M_mod}, age_mod: {age_mod}")
        print(f"R_data: {R_data}, M_data: {M_data}, age_data: {age_data}")
        print(f"R_data_err: {sigma_R}, M_data_err: {sigma_M}, age_data_err: {sigma_age}")
        return
    
    return log_L


def log_prior(theta):

    # CMF_mod, mass_mod, Teq_mod, Tint_mod = theta
    CMF_mod, mass_mod, Tint_mod = theta

    if CMF_min < CMF_mod < CMF_max and \
       mass_min < mass_mod < mass_max and \
       Tint_min < Tint_mod < Tint_max:
        return 0.0
    
    return -np.inf


#log-posterior function
def log_posterior(theta, y, y_err):

    if not np.isfinite(log_prior(theta)):
        return -np.inf
    
    return log_prior(theta) + log_likelihood(theta, y, y_err)

# %%
#original dataframe
df_original = pd.read_csv("giant_planets_dataset_v2.csv")

# %%
#lower age bound
age_min = 1 #Gyr

#filtered dataframe
df = df_original[
    (df_original['pl_mass'] >= mass_min) &
    (df_original['pl_mass'] <= mass_max) &
    (df_original['pl_teq'] >= Teq_min) &
    (df_original['pl_teq'] <= Teq_max) &
    (df_original['st_age'] >= age_min) &
    (df_original['st_age'].notna()) &
    (df_original['pl_zbulk'].notna())
]

# # %%
# k = 4
# N = 5
# for index in df[(k)*int(len(df)/N):(k+1)*int(len(df)/N)].index:
#     print(index)

# %%
#number of walkers
nwlk = 32

#number of steps
nsteps = 10000
# nsteps = 5000
# nsteps = 2000
# nsteps = 500

# save_flag = True

#initial samples independent of planet
CMF_init_samples = np.random.uniform(CMF_min, CMF_max, nwlk) #CMF: uniformly distributed from grid

# %%
for index in df.index:
    # print(index)

    # if os.path.isfile(f"Output5/corner/planet_{index}_corner_nsteps_{nsteps}.pdf"):
    if os.path.isfile(f"Output6/samples/planet_{index}_samples_nsteps_{nsteps}.dat"):
        continue
    
    #get planet parameters
    name, rad, rad_err1, rad_err2, mass, mass_err1, mass_err2, age, age_upper, age_lower, Teq, Zp, Zp_err = planet_parameters(df, index)

    #age errors
    age_err1 = age_upper - age
    age_err2 = age - age_lower


    #used later for corner plot
    mass_obs, rad_obs, age_obs = mass, rad, age
    Zenv = Z0


    #initial estimate for CMF
    # Zp = ufloat(Zp, Zp_err)
    CMF = ((ufloat(Zp, Zp_err)-Zenv)/(1-Zenv)).nominal_value
    CMF_obs = CMF
    CMF_err1 = ((ufloat(Zp, Zp_err)-Zenv)/(1-Zenv)).std_dev
    CMF_err2 = CMF_err1

    #observed data
    print(f"----------\nObserved data for {index}) {name}:\n----------")
    print(f"M_p: {mass:.2f} (- {mass_err2:.2f}, +{mass_err1:.2f}) Mjup")
    print(f"R_p: {rad:.2f} (- {rad_err2:.2f}, +{rad_err1:.2f}) Rjup")
    print(f"age: {age:.2f} = [{age_lower:.2f}, {age_upper:.2f}] Gyr = (- {age_err2:.2f}, +{age_err1:.2f}) Gyr")
    # print(f"Age: {age:.2f} [{age_lower:.2f}, {age_upper:.2f}] Gyr")
    print(f"T_eq: {Teq:.2f} K")

    # #CMF value read from cooling curve
    # CMF = 0.63
    # CMF_err1, CMF_err2 = 0.07, 0.07
    print(f"CMF: {CMF:.2f} (- {CMF_err2:.2f}, +{CMF_err1:.2f}) (planetsynth, Zenv = Z0)\n\n")

    # try:
    #     func_min = lambda x: age_upper - age_interp((CMF, mass*Mjup, Teq, x))
    #     Tint_min = brentq(func_min, min(Tint_array), max(Tint_array))

    #     func_max = lambda x: age_interp((CMF, mass*Mjup, Teq, x)) - age_lower
    #     Tint_max = brentq(func_max, min(Tint_array), max(Tint_array))

    #     # successful_indices.append(index)
    #     print(f"Successful for index {index}: Tint bounds = [{Tint_min:.2f}, {Tint_max:.2f}] K")
    # except ValueError as e:
    #     print(f"Failed for index {index}: {e}")
    #     # failed_indices.append(index)
    #     continue

    #functions for root-finding
    func_min = lambda x: age_upper - age_interp((CMF, mass*Mjup, Teq, x)) #age_upper - age_interp -> gives minimum of Tint, as we are working with maximum of age
    func_max = lambda x: age_interp((CMF, mass*Mjup, Teq, x)) - age_lower #age_interp - age_lower -> gives maximum of Tint, as we are working with minimum of age

    #root-finding for Tint bounds
    Tint_min = Tint_max = None  # Initialize to None in case of failure

    try:
        Tint_min = brentq(func_min, min(Tint_array), max(Tint_array))
        print(f"func_min succesful! -> Tint_min = {Tint_min:.2f} K")
    except ValueError as e:
        print(f"func_min failed for index {index}: {e}")

    try:        
        Tint_max = brentq(func_max, min(Tint_array), max(Tint_array))
        print(f"func_max succesful! -> Tint_max = {Tint_max:.2f} K")
    except ValueError as e:
        print(f"func_max failed for index {index}: {e}")

    if Tint_min is None or Tint_max is None:
        print(f"Warning: Could not determine both Tint bounds for index {index}.")
        continue


    #initial samples dependent on planet
    mass_init_samples = np.random.normal(mass, max(mass_err1, mass_err2), nwlk) #mass: normally distributed from df
    Tint_bound_error = 30 #K
    Tint_init_samples = np.random.uniform(Tint_min - Tint_bound_error, Tint_max + Tint_bound_error, nwlk) #Tint: uniformly distributed from grid and root-finding

    #observed data
    y = np.array([rad, mass, age])
    y_err = np.array([rad_err1, rad_err2, mass_err1, mass_err2, age_upper, age_lower])

    #define walker space
    pos = np.zeros((nwlk, 3))
    pos[:, 0] = CMF_init_samples
    pos[:, 1] = mass_init_samples
    pos[:, 2] = Tint_init_samples
    nwalkers, ndim = pos.shape
    print(f"1. Samples initliased!")

    #emcee main functions
    print(f"2. Starting emcee sampling with nsteps={nsteps}")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(y, y_err))
    sampler.run_mcmc(pos, nsteps, progress=True)
    print("3. Finished emcee sampling!")

    #plot and save convergence plot
    fig, axes = plt.subplots(ndim, figsize=(10, 11), sharex=True)
    plt.suptitle(f"'{index}) {name}'\nconvergence plot", fontsize=16)
    samples = sampler.get_chain()
    labels = ["CMF", "Mass [MJup]", "Tint [K]"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        # if labels[i] == "Tint [K]":
        #     ax.set_ylim(min(Tint_array), max(Tint_array))
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    fig.savefig(f"Output5/conv/planet_{index}_convergence_nsteps_{nsteps}.pdf",bbox_inches='tight',format='pdf', dpi=1000)
    fig.savefig(f"Output6/conv/planet_{index}_convergence_nsteps_{nsteps}.pdf",bbox_inches='tight',format='pdf', dpi=1000)

    # plt.close(fig)
    print("4. Convergence plot saved!")

    #get autocorrelation time and flat samples
    try:
        tau = sampler.get_autocorr_time()
        # print('tau = ', tau)
        ndiscard = int(2 * np.max(tau))
        nthin = int(0.5 * np.min(tau))
        flat_samples = sampler.get_chain(discard=ndiscard, thin=nthin, flat=True)
        print(f"5.1. Autocorrelation time: {tau}")
        print(f"5.2. Discarded {ndiscard} samples and thinned by {nthin}.")
    except (emcee.autocorr.AutocorrError, ValueError):
        flat_samples = sampler.get_chain(flat=True)
        print("5. Autocorrelation time could not be calculated. Using all samples.")

    #define flat sample space
    n = flat_samples.shape[0]
    CMF_flat_samples = flat_samples[:, 0]
    mass_flat_samples = flat_samples[:, 1]
    Tint_flat_samples = flat_samples[:, 2]
    print(f"6. Flat samples saved!")

    #get interpolated values using flat samples
    Rtot_samples = Rtot_interp((CMF_flat_samples, mass_flat_samples*Mjup, Teq, Tint_flat_samples)) #R_J
    Mtot_samples = Mtot_interp((CMF_flat_samples, mass_flat_samples*Mjup, Teq, Tint_flat_samples)) #M_J
    age_samples = age_interp((CMF_flat_samples, mass_flat_samples*Mjup, Teq, Tint_flat_samples)) #Gyr
    Zplanet_samples = Zplanet_interp((CMF_flat_samples, mass_flat_samples*Mjup, Teq, Tint_flat_samples))
    # Rbulk_samples = Rbulk_interp((CMF_flat_samples, mass_flat_samples, Teq, Tint_flat_samples))
    # Tsurf_samples = Tsurf_interp((CMF_flat_samples, mass_flat_samples, Teq, Tint_flat_samples))
    print(f"7. Interpolated values created with {n} samples!")

    #create empty data file
    data = np.zeros((n, 7))
    data[:, 0] = CMF_flat_samples
    data[:, 1] = mass_flat_samples
    data[:, 2] = Tint_flat_samples
    data[:, 3] = Rtot_samples
    data[:, 4] = Mtot_samples
    data[:, 5] = age_samples
    data[:, 6] = Zplanet_samples

    # np.savetxt(f"Output5/samples/planet_{index}_samples_nsteps_{nsteps}.dat",
    np.savetxt(f"Output6/samples/planet_{index}_samples_nsteps_{nsteps}.dat",
            data,
            header='CMF Mbulk[M_J] Tint[K] Rtot[R_J] Mtot[M_J] Age[Gyr] Zp',
            comments='', fmt='%1.4e',)
    print(f"8. Samples saved!\n")

    #plot corner plot
    print(f"CORNER PLOT\nProcessing planet '{index}) {name}'...")


    #get planet sample data
    # sample_data = pd.read_csv(f"Output4/samples/planet_{index}_samples.dat", sep='\s+')
    # sample_data = pd.read_csv(f"Output5/samples/planet_{index}_samples_nsteps_{nsteps}.dat", sep='\s+')
    sample_data = pd.read_csv(f"Output6/samples/planet_{index}_samples_nsteps_{nsteps}.dat", sep='\s+')


    CMF = sample_data['CMF']
    Mbulk = sample_data['Mbulk[M_J]']
    Tint = sample_data['Tint[K]']

    R = sample_data['Rtot[R_J]']
    M = sample_data['Mtot[M_J]']
    age = sample_data['Age[Gyr]']
    Zp = sample_data['Zp']

    #account for atmospheric mass
    Matm = M - Mbulk
    Menv_int = Mbulk * (1-CMF)
    EMF = (Menv_int + Matm) / Mbulk #EMF: includes atmosphere and interior and NOT core
    CMF_calc = 1 - EMF
    Zcore = 1
    # Zp_calc = Zcore*CMF_calc + Zenv*(1-CMF_calc) #= CMF_calc + Zenv*EMF
    Zp_calc = Zcore*CMF + Zenv*(1-CMF)

    #assign flat samples
    flat_samples = np.zeros((len(CMF), 6))
    flat_samples[:, 0] = CMF
    flat_samples[:, 1] = M
    flat_samples[:, 2] = R
    flat_samples[:, 3] = Tint
    flat_samples[:, 4] = age
    flat_samples[:, 5] = Zp_calc

    # #plot corner plot
    # figure = corner.corner(
    #     flat_samples,
    #     labels=[r"CMF", r"M [$M_{Jup}$]",\
    #             r"R [$R_{Jup}$]", r"$T_{int}$ [K] ",\
    #             r"Age [Gyr]", r"$Z_{planet}$"],
    #             quantiles=[0.16, 0.5, 0.84],\
    #             truths=[np.nan, mass_obs, rad_obs, np.nan, age_obs, np.nan],\
    #             show_titles=True,
    #             title_kwargs={"fontsize": 14},
    #             label_kwargs={"fontsize": 16},
    #             # hist_kwargs={"color": "C0", "alpha": 0.5, "density": False}
    # )
    # plt.tight_layout()
    # plt.suptitle(f"'{index}) {name}'\ncorner plot | nsteps = {nsteps}", fontsize=16)
    # # plt.subplots_adjust(top=0.9)
    # plt.savefig(f"Output5/corner/planet_{index}_corner_nsteps_{nsteps}.pdf")
    # plt.savefig(f"Output6/corner/planet_{index}_corner_nsteps_{nsteps}.pdf")
    #             # bbox_inches='tight', format='pdf', dpi=1000)
    # plt.show()
# %%
