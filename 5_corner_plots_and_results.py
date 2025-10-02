# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import corner
import h5py
from uncertainties import ufloat

print("Done")

# from emcee_sampling import planet_parameters
# %%
Z0 = 0.014 #solar value
Zenv = Z0
Mjup = 318 #M_E
Rjup = 11.2 #R_E
sigma_b = 5.67e-5 # erg/s/cm2/K4
conv_AU_to_RS = 215.032 #AU to RS

df_original = pd.read_csv("giant_planets_dataset_v2.csv")

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
Tint_min, Tint_max = (min(Tint_array), max(Tint_array))
age_min = 1 #Gyr

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

    #star's observed age (uniform distribution)
    age = planet['st_age'] #Gyr
    age_upper  = planet['st_age_upper'] #Gyr
    age_lower = planet['st_age_lower'] #Gyr

    #planet's observed equilibrium temperature (fixed value)
    Teq = planet['pl_teq'] #K

    #planet's bulk metallicity (based on planetsynth)
    Zp = planet['pl_zbulk'] #Zsun
    Zp_err = planet['pl_zbulk_err'] #Zsun

    return name, rad, rad_err1, rad_err2, mass, mass_err1, mass_err2, age, age_upper, age_lower, Teq, Zp, Zp_err

# %%
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

# %%
#new dataframe to be saved with new values
df_updated = df_original.copy()

#new parameters to add
#model radius
df_updated['R_mod'] = np.nan
df_updated['R_mod_err1'] = np.nan
df_updated['R_mod_err2'] = np.nan

#model mass
df_updated['M_mod'] = np.nan
df_updated['M_mod_err1'] = np.nan
df_updated['M_mod_err2'] = np.nan

#model age
df_updated['age_mod'] = np.nan
df_updated['age_mod_err1'] = np.nan
df_updated['age_mod_err2'] = np.nan

#model Tint
df_updated['Tint_mod'] = np.nan
df_updated['Tint_mod_err1'] = np.nan
df_updated['Tint_mod_err2'] = np.nan

#model CMF
df_updated['CMF_mod'] = np.nan
df_updated['CMF_mod_err1'] = np.nan
df_updated['CMF_mod_err2'] = np.nan

#model Zp
df_updated['Zp_mod'] = np.nan
#saving error on this as std dev because it is needed later for further calculations
#assymmetric errors would be hard to handle
df_updated['Zp_mod_err'] = np.nan 
# df_updated['Zp_mod_err1'] = np.nan
# df_updated['Zp_mod_err2'] = np.nan

#model Zp/Z*
df_updated['Zp_rel'] = np.nan
df_updated['Zp_rel_err'] = np.nan
# df_updated['Zp_rel_err1'] = np.nan
# df_updated['Zp_rel_err2'] = np.nan

#model Mz from obs mass
df_updated['Mz_mod_obs'] = np.nan
df_updated['Mz_mod_obs_err'] = np.nan
# df_updated['Mz_mod_obs_err1'] = np.nan
# df_updated['Mz_mod_obs_err2'] = np.nan

#model Mz from mod mass
df_updated['Mz_mod_mod'] = np.nan
df_updated['Mz_mod_mod_err'] = np.nan
# df_updated['Mz_mod_err1'] = np.nan
# df_updated['Mz_mod_err2'] = np.nan

# %%
median_level = 0.5
sigma_level = 0.68
nsteps = 10000
# nsteps = 100

failed_indices = []
for index in df.index:
    # print(index)
    # print(f"{index}) {df_original.loc[index]['pl_name']}")
    # print(f"planet_{index}_samples.dat")

    #get planet parameters
    name, rad_obs, _, _, mass_obs, mass_obs_err1, mass_obs_err2, age_obs, _, _, Teq_obs, Zp_obs, _ = planet_parameters(df, index)
    print(f"\n----------\nProcessing planet '{index}) {name}'...")

    #get planet sample data
    try:
        # sample_data = pd.read_csv(f"Output4/samples/planet_{index}_samples.dat", sep='\s+')
        sample_data = pd.read_csv(f"Output6/samples/planet_{index}_samples_nsteps_{nsteps}.dat", sep='\s+')
    except FileNotFoundError:
        print(f"Failed to process planet '{index}) {name}': sample file not found\n----------")
        failed_indices.append(index)
        continue

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
    
    # flat_samples[:, 0] = CMF
    # flat_samples[:, 1] = M
    # flat_samples[:, 2] = R
    # flat_samples[:, 3] = Tint
    # flat_samples[:, 4] = age
    # flat_samples[:, 5] = Zp_calc

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


    flat_samples[:, 0] = R
    flat_samples[:, 1] = M
    flat_samples[:, 2] = age
    flat_samples[:, 3] = Tint
    flat_samples[:, 4] = CMF
    flat_samples[:, 5] = Zp_calc

    if not os.path.isfile(f"Output6/corner/planet_{index}_corner_nsteps_{nsteps}.pdf"):
        #plot corner plot
        figure = corner.corner(
        flat_samples,
        # labels=[r"$R_{\text{mod}}$ [$R_J$]", r"$M_{\text{mod}}$ [$M_J$]",\
        #         r"$\tau_{\text{mod}}$ [Gyr]", r"$T_{\text{int,mod}}$ [K]",\
        #         r"CMF$_{\text{mod}}$", r"$Z__{\text{p,mod}}$"],
        labels=[r"$R_{p,\text{mod}}$ [$R_J$]", r"$M_{p,\text{mod}}$ [$M_J$]",\
                r"$\tau_{p,\text{mod}}$ [Gyr]", r"$T_{\text{int,mod}}$ [K]",\
                r"CMF$_{\text{mod}}$", r"$Z_{p,\text{mod}}$"],
                quantiles=[0.16, 0.5, 0.84],\
                truths=[rad_obs, mass_obs, age_obs, np.nan, np.nan, np.nan],\
                show_titles=True,
                title_kwargs={"fontsize": 16},
                label_kwargs={"fontsize": 16},
                # hist_kwargs={"color": "C0", "alpha": 0.5, "density": False}
        )
        plt.tight_layout()
        plt.suptitle(f"{index}) {name}\n"\
                    #  r"$n_{steps}$" f" = {nsteps}\n" \
                    fr"$R_p$ = {rad_obs:.2f} $R_J$ | $M_p$ = {mass_obs:.2f} $M_J$ | $\tau_p$ = {age_obs:.2f} Gyr |" r" $T_{eq}$" f" = {Teq_obs:.2f} K", fontsize=18)
        # plt.subplots_adjust(top=0.9)
        # plt.savefig(f"Output4/corner/planet_{index}_corner_nsteps_{nsteps}.pdf")
        # plt.savefig(f"Output6/corner/planet_{index}_corner_nsteps_{nsteps}.pdf")
        # plt.savefig(f"Output6/corner/2_planet_{index}_corner_nsteps_{nsteps}.pdf")
        #             # bbox_inches='tight', format='pdf', dpi=1000)
        plt.savefig(f"Output6/corner/2_planet_{index}_corner_nsteps_{nsteps}.png",
            bbox_inches='tight', format='pdf', dpi=300)
        plt.show()
        # continue    

    # %%


    #calculated median values
    R_mod, M_mod, age_mod, Tint_mod, CMF_mod, Zp_mod = np.quantile(flat_samples, median_level, axis=0)
    # CMF_mod, M_mod, R_mod, Tint_mod, age_mod, Zp_mod = np.quantile(flat_samples, median_level, axis=0)

    #calculated upper errors
    R_mod_err1, M_mod_err1, age_mod_err1, Tint_mod_err1, CMF_mod_err1, Zp_mod_err1 = \
        np.quantile(flat_samples, (median_level+(sigma_level)/2), axis=0) - np.quantile(flat_samples, median_level, axis=0)
        # np.quantile(flat_samples, median_level, axis=0) - np.quantile(flat_samples, (median_level-(sigma_level)/2), axis=0)

    #calculated lower errors
    R_mod_err2, M_mod_err2, age_mod_err2, Tint_mod_err2, CMF_mod_err2, Zp_mod_err2 = \
        np.quantile(flat_samples, median_level, axis=0) - np.quantile(flat_samples, (median_level-(sigma_level)/2), axis=0)
        # np.quantile(flat_samples, (median_level+(sigma_level)/2), axis=0) - np.quantile(flat_samples, median_level, axis=0)
        

    #append values to updated dataframe
    #model radius
    df_updated.loc[index, 'R_mod'] = R_mod
    df_updated.loc[index, 'R_mod_err1'] = R_mod_err1
    df_updated.loc[index, 'R_mod_err2'] = R_mod_err2

    #model mass
    df_updated.loc[index, 'M_mod'] = M_mod
    df_updated.loc[index, 'M_mod_err1'] = M_mod_err1
    df_updated.loc[index, 'M_mod_err2'] = M_mod_err2

    #model age
    df_updated.loc[index, 'age_mod'] = age_mod
    df_updated.loc[index, 'age_mod_err1'] = age_mod_err1
    df_updated.loc[index, 'age_mod_err2'] = age_mod_err2

    #model Tint
    df_updated.loc[index, 'Tint_mod'] = Tint_mod
    df_updated.loc[index, 'Tint_mod_err1'] = Tint_mod_err1
    df_updated.loc[index, 'Tint_mod_err2'] = Tint_mod_err2

    #model CMF
    df_updated.loc[index, 'CMF_mod'] = CMF_mod
    df_updated.loc[index, 'CMF_mod_err1'] = CMF_mod_err1
    df_updated.loc[index, 'CMF_mod_err2'] = CMF_mod_err2

    #turning assymmetric errors of Zp from MCMC into symmetric errors
    Zp_err = max(Zp_mod_err1, np.abs(Zp_mod_err2))

    #planet bulk metallicity (Zp)
    Zp_ufloat = ufloat(Zp_mod, Zp_err)
    df_updated.loc[index, 'Zp_mod'] = Zp_ufloat.nominal_value
    df_updated.loc[index, 'Zp_mod_err'] = Zp_ufloat.std_dev


    #stellar metallicity ([Fe/H])
    st_met_exp_nom_val = df.loc[index]['st_met']
    #turning assymmetric errors into symmetric
    st_met_exp_err = max(df.loc[index]['st_met_err1'], np.abs(df.loc[index]['st_met_err2']))
    if pd.isna(st_met_exp_err):
        print(f"Warning: No error for stellar metallicity for planet '{index}) {name}'")
        st_met_exp_err = 0.0
        print(f"Setting error to {st_met_exp_err} for planet '{index}) {name}'")
        df_updated.loc[index, 'st_met_exp_err'] = st_met_exp_err
        # print(f"st_met_exp_nom_val: {st_met_exp_nom_val}, st_met_exp_err: {st_met_exp_err}")


    st_met_exp_ufloat = ufloat(st_met_exp_nom_val, st_met_exp_err)
    # print(f"st_met_exp_ufloat: {st_met_exp_ufloat}")

    #stellar metallicity (Zstar = Z0 * 10^[Fe/H])
    Zstar_ufloat = Z0 * 10**(st_met_exp_ufloat)

    #get relative bulk metallicity (Zp/Zstar)
    Zp_rel_ufloat = (Zp_ufloat / Zstar_ufloat)
    df_updated.loc[index, 'Zp_rel'] = Zp_rel_ufloat.nominal_value
    df_updated.loc[index, 'Zp_rel_err'] = Zp_rel_ufloat.std_dev
    print(f"Zp_rel_ufloat: {Zp_rel_ufloat}")


    #turning assymmetric errors of M_obs from df into symmetric errors
    M_obs_err = max(mass_obs_err1, mass_obs_err2)
    
    #mass as ufloat object
    M_obs = mass_obs
    M_obs_ufloat = ufloat(M_obs, M_obs_err)

    #heavy element mass (Mz = Zp * Mp)
    Mz_mod_obs_ufloat = Zp_ufloat * M_obs_ufloat
    df_updated.loc[index, 'Mz_mod_obs'] = Mz_mod_obs_ufloat.nominal_value * Mjup #convert to Earth masses
    df_updated.loc[index, 'Mz_mod_obs_err'] = Mz_mod_obs_ufloat.std_dev * Mjup #convert to Earth masses


    #turning assymmetric errors of M_mod from MCMC into symmetric errors
    M_mod_err = max(M_mod_err1, M_mod_err2)
    
    #mass as ufloat object
    M_mod_ufloat = ufloat(M_mod, M_mod_err)

    #heavy element mass (Mz = Zp * Mp)
    Mz_mod_mod_ufloat = Zp_ufloat * M_mod_ufloat
    df_updated.loc[index, 'Mz_mod_mod'] = Mz_mod_mod_ufloat.nominal_value * Mjup #convert to Earth masses
    df_updated.loc[index, 'Mz_mod_mod_err'] = Mz_mod_mod_ufloat.std_dev * Mjup #convert to Earth masses

    print(f"Finsihed processing planet '{index}) {name}'\n----------")
    
# %%
# df_updated[['pl_rad', 'R_mod']]
# df_updated[['pl_mass', 'M_mod']]
# df_updated[['st_age', 'age_mod']]

# %%
#save updated dataframe
# df_updated.to_csv("giant_planets_dataset_v2_updated_nsteps_5000.csv", index=False)
# df_updated.to_csv("giant_planets_dataset_v2_updated_nsteps_10000.csv", index=False)
# df_updated.to_csv(f"giant_planets_dataset_v2_updated_nsteps_{nsteps}.csv", index=False)
df_updated.to_csv(f"giant_planets_dataset_v2_updated2_nsteps_{nsteps}.csv", index=False)

# %%
