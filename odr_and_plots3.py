# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from matplotlib.gridspec import GridSpec

import scipy.odr as odr
# %%
Z0 = 0.014 #solar value
Zenv = Z0
Mjup = 318 #M_E
Rjup = 11.2 #R_E
sigma_b = 5.67e-5 # erg/s/cm2/K4
conv_AU_to_RS = 215.032 #AU to RS

nsteps = 10000
# df_updated = pd.read_csv(f"giant_planets_dataset_v2_updated_nsteps_{nsteps}.csv")
df_updated = pd.read_csv(f"giant_planets_dataset_v2_updated2_nsteps_{nsteps}.csv")

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

# %%
# #converting df_updated to latex table
# latex_table_updated = df_updated.to_latex(float_format="%.2f")


# %%
#datasets

df_all = df_updated[
    (df_updated['pl_mass'] >= mass_min) &
    (df_updated['pl_mass'] <= mass_max) &
    (df_updated['pl_teq'] >= Teq_min) &
    (df_updated['pl_teq'] <= Teq_max) &
    (df_updated['st_age'] >= age_min) &
    (df_updated['st_age'].notna()) &
    (df_updated['pl_zbulk'].notna())
# }
# ].drop(124)
].drop([124, 130,                                               #inter/extrapolation error
        7, 25, 39, 51, 56, 65, 68, 87, 88, 108, 113, 122, 130]) #inflated planets
# ])
# ].drop([
#     # 0, 3, 8, 102, 108, #for planets not yet solved
#         124, 130])

# Set transition temperature
transition_temp = 3900

#separate the data into two groups based on the transition temperature
df_M = df_all[df_all['st_teff'] < transition_temp] # Group 1: Teff < transition_temp should be labelled df_M
df_FGK = df_all[df_all['st_teff'] >= transition_temp] # Group 2: Teff >= transition_temp should be labelled df_FGK


# %%
#linear function
def loglog_f(B, x):
    return (B[1] * x + B[0])

#model
loglog_model = odr.Model(loglog_f)
# %%
#mass obs
x_obs_label = 'pl_mass'
# lin_x_obs = df[x_obs_label]
# lin_x_obs_err = pd.Series(
#     [max(df.loc[index][x_obs_label+'_err1'], np.abs(df.loc[index][x_obs_label+'_err2'])) for index in df.index],
#     index=df.index, name=x_obs_label+'_err')

# #mass mod
# x_mod_label = 'M_mod'
# lin_x_mod = df[x_mod_label]
# lin_x_mod_err = pd.Series(
#     [max(df.loc[index][x_mod_label+'_err1'], np.abs(df.loc[index][x_mod_label+'_err2'])) for index in df.index],
#     index=df.index, name=x_mod_label+'_err')

# Define the variables to loop over
y_obs_labels = ['pl_zbulk_rel', 'pl_mbulk']
y_mod_labels = ['Zp_rel', 'Mz_mod_obs']
y_obs_err_labels = ['pl_zbulk_rel_err', 'pl_mbulk_err']
y_mod_err_labels = ['Zp_rel_err', 'Mz_mod_obs_err']
# y_names = ["Normalised Metallicity", "Heavy-element Mass"]
y_names = ["Normalised Metallicities", "Heavy-Element Masses"]
# y_symbols = ["$Z_p/Z_*$", "$M_{Z}$"]
y_symbols = [r"$Z_p/Z_*, Z_{p,\text{mod}}/Z_*$", r"$M_Z, M_{Z,\text{mod}}$"]
y_units = ['', "[$M_{\oplus}$]"]

planetsynth_color = '#1f77b4'
planetsynth_label = 'MH25'
GASTLI_color = '#ff7f0e'
GASTLI_label = 'A24'
alpha = 0.6

figsize = (6, 8)
ratio = [2.5, 0.75]  # Height ratios for the subplots

x_lim = (1e-1, 3e0)  # x-axis limits for mass

y_lim_Zp_rel_upper = (7e-2, 2e2)  # y-axis limits for Zp_rel upper
y_lim_Zp_rel = (5e-1, 2e1)  # y-axis limits for Zp_rel

y_lim_Mz_upper = (2e-1, 6e2)
y_lim_Mz = (1e0, 3e2)  # y-axis limits for Mz

# %%
#Thorngren et al. 2016
#normalised metallicity
def Zp_rel_Thorngren(x):
    """
    y = (b0_10 +/- b0_10_err) * (x)**(b1 +/- b1_err)
    """

    b0_10 = 9.7
    b0_10_err = 1.28

    b1 = -0.45
    b1_err = 0.09

    y_fit = (b0_10 * (x**b1))
    y_fit_lower = (b0_10 - b0_10_err) * (x**(b1-b1_err))
    y_fit_upper = (b0_10 + b0_10_err) * (x**(b1+b1_err))

    return y_fit, y_fit_lower, y_fit_upper

def Mz_Thorngren(x):
    """
    y = (b0_10 +/- b0_10_err) * (x)**(b1 +/- b1_err)
    """

    b0_10 = 57.9
    b0_10_err = 7.03

    b1 = 0.61
    b1_err = 0.08

    y_fit = (b0_10 * (x**b1))
    y_fit_lower = (b0_10 - b0_10_err) * (x**(b1-b1_err))
    y_fit_upper = (b0_10 + b0_10_err) * (x**(b1+b1_err))

    return y_fit, y_fit_lower, y_fit_upper

#Howard et al. 2025
#normalised metallicity
def Zp_rel_Howard(x):
    """
    y = (b0_10 +/- b0_10_err) * (x)**(b1 +/- b1_err)
    """

    b0_10 = 5.09
    b0_10_err = 0.95

    b1 = -0.57
    b1_err = 0.13

    y_fit = (b0_10 * (x**b1))
    y_fit_lower = (b0_10 - b0_10_err) * (x**(b1-b1_err))
    y_fit_upper = (b0_10 + b0_10_err) * (x**(b1+b1_err))

    return y_fit, y_fit_lower, y_fit_upper


# %%


#Plot 1) all planets for AKZM24 vs MH25
alpha = 1
df = df_all
for i in range(len(y_obs_labels)):
    # Get labels for this iteration
    y_obs_label = y_obs_labels[i]
    y_obs_err_label = y_obs_err_labels[i]

    y_mod_label = y_mod_labels[i]
    y_mod_err_label = y_mod_err_labels[i]

    y_name = y_names[i]
    y_symbol = y_symbols[i]
    y_unit = y_units[i]

    # Data
    lin_x_obs = df[x_obs_label]
    lin_x_obs_err = pd.Series(
        [max(df.loc[index][x_obs_label+'_err1'], np.abs(df.loc[index][x_obs_label+'_err2'])) for index in df.index],
        index=df.index, name=x_obs_label+'_err')
    lin_y_obs = df[y_obs_label]
    lin_y_obs_err = df[y_obs_err_label]
    lin_y_mod = df[y_mod_label]
    lin_y_mod_err = df[y_mod_err_label]

    # ODR for observed
    log_x_obs = np.log10(lin_x_obs)
    log_x_obs_err = lin_x_obs_err / (lin_x_obs * np.log(10))
    log_y_obs = np.log10(lin_y_obs)
    log_y_obs_err = lin_y_obs_err / (lin_y_obs * np.log(10))

    loglog_data_obs = odr.RealData(log_x_obs, log_y_obs, sx=log_x_obs_err, sy=log_y_obs_err)
    odr_loglog_obs = odr.ODR(data=loglog_data_obs, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_obs = odr_loglog_obs.run()
    b0_obs, b1_obs = output_loglog_obs.beta
    b0_obs_err, b1_obs_err = output_loglog_obs.sd_beta

    # ODR for model
    log_y_mod = np.log10(lin_y_mod)
    log_y_mod_err = lin_y_mod_err / (lin_y_mod * np.log(10))
    loglog_data_mod = odr.RealData(log_x_obs, log_y_mod, sx=log_x_obs_err, sy=log_y_mod_err)
    odr_loglog_mod = odr.ODR(data=loglog_data_mod, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_mod = odr_loglog_mod.run()
    b0_mod, b1_mod = output_loglog_mod.beta
    b0_mod_err, b1_mod_err = output_loglog_mod.sd_beta

    # Fit lines
    x_fit = np.logspace(np.log10(min(lin_x_obs)), np.log10(max(lin_x_obs)), 200)
    print(f"y = {y_name}\nMass range: [{min(lin_x_obs):.2f}, {max(lin_x_obs):.2f}] M_J\n")

    print(f"Observed: b0 = {b0_obs:.2f} +/- {b0_obs_err:.2f}, b1 = {b1_obs:.2f} +/- {b1_obs_err:.2f}")
    print(f"y = ({10**b0_obs:.2f} +/- {np.log(10) * 10**b0_obs * b0_obs_err:.2f}) * M_p ^ ({b1_obs:.2f} +/- {b1_obs_err:.2f})\n")

    print(f"Model: b0 = {b0_mod:.2f} +/- {b0_mod_err:.2f}, b1 = {b1_mod:.2f} +/- {b1_mod_err:.2f}")
    print(f"y = ({10**b0_mod:.2f} +/- {np.log(10) * 10**b0_mod * b0_mod_err:.2f}) * M_p ^ ({b1_mod:.2f} +/- {b1_mod_err:.2f})\n")

    y_fit_obs = (x_fit**b1_obs) * (10**b0_obs)
    y_fit_obs_lower = ((x_fit)**(b1_obs-b1_obs_err)) * (10**(b0_obs-b0_obs_err))
    y_fit_obs_upper = ((x_fit)**(b1_obs+b1_obs_err)) * (10**(b0_obs+b0_obs_err))

    y_fit_mod = (x_fit**b1_mod) * (10**b0_mod)
    y_fit_mod_lower = ((x_fit)**(b1_mod-b1_mod_err)) * (10**(b0_mod-b0_mod_err))
    y_fit_mod_upper = ((x_fit)**(b1_mod+b1_mod_err)) * (10**(b0_mod+b0_mod_err))

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios': ratio, 'hspace': 0.05},
                            #  sharex=True
                             )
    ax0 = axes[0]
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_title(f"All planets")
    ax0.set_ylabel(f"{y_name} {y_symbol} {y_unit}".strip())

    ax0.errorbar(lin_x_obs, lin_y_obs, 
                 xerr=lin_x_obs_err, yerr=lin_y_obs_err,
                 fmt='s',
                 color=planetsynth_color,
                 ecolor=planetsynth_color,
                 markeredgecolor='black',
                 markeredgewidth=1,
                 capsize=3,
                 alpha=alpha
                )   

    ax0.errorbar(lin_x_obs, lin_y_mod,
                 xerr=lin_x_obs_err, yerr=lin_y_mod_err,
                 fmt='s',
                 color=GASTLI_color,
                 ecolor=GASTLI_color,
                 markeredgecolor='black',
                 markeredgewidth=1,
                 capsize=3,
                 alpha=alpha
                )
    
    #Calculated fits
    ax0.plot(x_fit, y_fit_mod, GASTLI_color,
                    label=fr"$\propto M_p^{{{b1_mod:.2f}}}$ [{GASTLI_label}]")
    ax0.fill_between(x_fit, y_fit_mod_lower, y_fit_mod_upper, color=GASTLI_color, alpha=0.2)

    ax0.plot(x_fit, y_fit_obs, planetsynth_color,
                    label=fr"$\propto M_p^{{{b1_obs:.2f}}}$ [{planetsynth_label}]")
    ax0.fill_between(x_fit, y_fit_obs_lower, y_fit_obs_upper, color=planetsynth_color, alpha=0.2)




    # # Thorngren & Howard fits
    # if y_symbol == y_symbols[0]:
    #     y_fit_thorngren, y_fit_thorngren_lower, y_fit_thorngren_upper = Zp_rel_Thorngren(x_fit)
    #     ax0.plot(x_fit, y_fit_thorngren, 'g-',
    #                 label=fr"$\propto M_p^{{-0.45}}$ [T16]")
    #     ax0.fill_between(x_fit, y_fit_thorngren_lower, y_fit_thorngren_upper, color='g', alpha=0.2)

    #     y_fit_howard, y_fit_howard_lower, y_fit_howard_upper = Zp_rel_Howard(x_fit)
    #     ax0.plot(x_fit, y_fit_howard, 'r-',
    #                 label=fr"$\propto M_p^{{-0.57}}$ [H25]")
    #     ax0.fill_between(x_fit, y_fit_howard_lower, y_fit_howard_upper, color='r', alpha=0.2)

    # elif y_symbol == y_symbols[1]:
    #     y_fit_thorngren, y_fit_thorngren_lower, y_fit_thorngren_upper = Mz_Thorngren(x_fit)
    #     ax0.plot(x_fit, y_fit_thorngren, 'g-',
    #                 label=fr"$\propto M_p^{{0.61}}$ [T16]")
    #     ax0.fill_between(x_fit, y_fit_thorngren_lower, y_fit_thorngren_upper, color='g', alpha=0.2)

    if y_symbol == y_symbols[0]:
        ax0.set_ylim(y_lim_Zp_rel_upper)
        ax0.legend(loc="upper left", 
                #    ncol=2
                   )
    elif y_symbol == y_symbols[1]:
        ax0.set_ylim(y_lim_Mz_upper)
        ax0.legend(loc="upper left", 
                #    ncol=2
                   )
    # ax0.legend()

    # Bottom plot: differences
    ax1 = axes[1]
    ax1.scatter(lin_x_obs, lin_y_mod - lin_y_obs, color='k', s=30,
                # label=f"[{GASTLI_label}] - [{planetsynth_label}]"
                )
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    if y_symbol == y_symbols[0]:
        ax1.set_ylabel(r"$(Z_{p,\text{mod}}- Z_p)/Z_*$ " f"{y_unit}")
        # ax1.set_ylim(1e-1, 2e1)
        ax1.set_ylim(y_lim_Zp_rel)
        # ax1.set_yticks([1e-1, 1e0, 1e1])
        # ax1.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
        ax1.set_yticks([1e0, 1e1])
        ax1.set_yticklabels([r'$10^{0}$', r'$10^{1}$'])
    elif y_symbol == y_symbols[1]:
        ax1.set_ylabel(r"$(M_{Z,\text{mod}} - M_Z)$ " f"{y_unit}")
        # ax1.set_ylim(1e0, 3e2)
        ax1.set_ylim(y_lim_Mz)
        ax1.set_yticks([1e0, 1e1, 1e2])
        ax1.set_yticklabels([r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'])
    ax1.set_xlim(x_lim)
    # ax1.set_ylabel(r"$\Delta$" fr"{y_symbol}")
    ax1.set_xlabel(fr"Mass $M_p$ [$M_J$]")
    # ax1.legend()

    plt.tight_layout()
    plt.show()
    # fig.savefig(f"img/3_all_{y_obs_label}.pdf", bbox_inches='tight')
    # fig.savefig(f"img/3_all_{y_obs_label}.png", bbox_inches='tight', dpi=300)
    

    # #creating similar plots, just without the datapoints and bottom panel. so just the trendlines
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_title(f"All planets")
    # ax.set_ylabel(f"{y_name} {y_symbol} {y_unit}".strip())
    # ax.set_xlabel(fr"Mass $M_p$ [$M_J$]")
    # ax.plot(x_fit, y_fit_mod, GASTLI_color,
    #                 label=fr"$\propto M_p^{{{b1_mod:.2f}}}$ [{GASTLI_label}]")
    # ax.fill_between(x_fit, y_fit_mod_lower, y_fit_mod_upper, color=GASTLI_color, alpha=0.2)
    # ax.plot(x_fit, y_fit_obs, planetsynth_color,
    #                 label=fr"$\propto M_p^{{{b1_obs:.2f}}}$ [{planetsynth_label}]")
    # ax.fill_between(x_fit, y_fit_obs_lower, y_fit_obs_upper, color=planetsynth_color, alpha=0.2)
    # if y_symbol == y_symbols[0]:
    #     ax.set_ylim(y_lim_Zp_rel_upper)
    #     ax.legend(loc="upper left", ncol=2)
    # elif y_symbol == y_symbols[1]:
    #     ax.set_ylim(y_lim_Mz_upper)
    #     ax.legend(loc="upper left", ncol=2)
    # # Thorngren & Howard fits
    # if y_symbol == y_symbols[0]:
    #     ax.plot(x_fit, y_fit_thorngren, 'g-',
    #                 label=fr"$\propto M_p^{{-0.45}}$ [T16]")
    #     ax.fill_between(x_fit, y_fit_thorngren_lower, y_fit_thorngren_upper, color='g', alpha=0.2)

    #     ax.plot(x_fit, y_fit_howard, 'r-',
    #                 label=fr"$\propto M_p^{{-0.57}}$ [H25]")
    #     ax.fill_between(x_fit, y_fit_howard_lower, y_fit_howard_upper, color='r', alpha=0.2)

    # elif y_symbol == y_symbols[1]:
    #     ax.plot(x_fit, y_fit_thorngren, 'g-',
    #                 label=fr"$\propto M_p^{{0.61}}$ [T16]")
    #     ax.fill_between(x_fit, y_fit_thorngren_lower, y_fit_thorngren_upper, color='g', alpha=0.2)
    
    # if y_symbol == y_symbols[0]:
    #     ax.set_ylim(y_lim_Zp_rel_upper)
    #     ax.legend(loc="upper left", ncol=2)
    # elif y_symbol == y_symbols[1]:
    #     ax.set_ylim(y_lim_Mz_upper)
    #     ax.legend(loc="upper left", ncol=2)

    # plt.tight_layout()
    # plt.show()


# %%
#Plot 1) all planets for AKZM24 vs MH25 (emtpty)
alpha = 0.4
df = df_all
for i in range(len(y_obs_labels)):
    # Get labels for this iteration
    y_obs_label = y_obs_labels[i]
    y_obs_err_label = y_obs_err_labels[i]

    y_mod_label = y_mod_labels[i]
    y_mod_err_label = y_mod_err_labels[i]

    y_name = y_names[i]
    y_symbol = y_symbols[i]
    y_unit = y_units[i]

    # Data
    lin_x_obs = df[x_obs_label]
    lin_x_obs_err = pd.Series(
        [max(df.loc[index][x_obs_label+'_err1'], np.abs(df.loc[index][x_obs_label+'_err2'])) for index in df.index],
        index=df.index, name=x_obs_label+'_err')
    lin_y_obs = df[y_obs_label]
    lin_y_obs_err = df[y_obs_err_label]
    lin_y_mod = df[y_mod_label]
    lin_y_mod_err = df[y_mod_err_label]

    # ODR for observed
    log_x_obs = np.log10(lin_x_obs)
    log_x_obs_err = lin_x_obs_err / (lin_x_obs * np.log(10))
    log_y_obs = np.log10(lin_y_obs)
    log_y_obs_err = lin_y_obs_err / (lin_y_obs * np.log(10))

    loglog_data_obs = odr.RealData(log_x_obs, log_y_obs, sx=log_x_obs_err, sy=log_y_obs_err)
    odr_loglog_obs = odr.ODR(data=loglog_data_obs, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_obs = odr_loglog_obs.run()
    b0_obs, b1_obs = output_loglog_obs.beta
    b0_obs_err, b1_obs_err = output_loglog_obs.sd_beta

    # ODR for model
    log_y_mod = np.log10(lin_y_mod)
    log_y_mod_err = lin_y_mod_err / (lin_y_mod * np.log(10))
    loglog_data_mod = odr.RealData(log_x_obs, log_y_mod, sx=log_x_obs_err, sy=log_y_mod_err)
    odr_loglog_mod = odr.ODR(data=loglog_data_mod, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_mod = odr_loglog_mod.run()
    b0_mod, b1_mod = output_loglog_mod.beta
    b0_mod_err, b1_mod_err = output_loglog_mod.sd_beta

    # Fit lines
    x_fit = np.logspace(np.log10(min(lin_x_obs)), np.log10(max(lin_x_obs)), 200)
    print(f"y = {y_name}\nMass range: [{min(lin_x_obs):.2f}, {max(lin_x_obs):.2f}] M_J\n")

    print(f"Observed: b0 = {b0_obs:.2f} +/- {b0_obs_err:.2f}, b1 = {b1_obs:.2f} +/- {b1_obs_err:.2f}")
    print(f"y = ({10**b0_obs:.2f} +/- {np.log(10) * 10**b0_obs * b0_obs_err:.2f}) * M_p ^ ({b1_obs:.2f} +/- {b1_obs_err:.2f})\n")

    print(f"Model: b0 = {b0_mod:.2f} +/- {b0_mod_err:.2f}, b1 = {b1_mod:.2f} +/- {b1_mod_err:.2f}")
    print(f"y = ({10**b0_mod:.2f} +/- {np.log(10) * 10**b0_mod * b0_mod_err:.2f}) * M_p ^ ({b1_mod:.2f} +/- {b1_mod_err:.2f})\n")

    y_fit_obs = (x_fit**b1_obs) * (10**b0_obs)
    y_fit_obs_lower = ((x_fit)**(b1_obs-b1_obs_err)) * (10**(b0_obs-b0_obs_err))
    y_fit_obs_upper = ((x_fit)**(b1_obs+b1_obs_err)) * (10**(b0_obs+b0_obs_err))

    y_fit_mod = (x_fit**b1_mod) * (10**b0_mod)
    y_fit_mod_lower = ((x_fit)**(b1_mod-b1_mod_err)) * (10**(b0_mod-b0_mod_err))
    y_fit_mod_upper = ((x_fit)**(b1_mod+b1_mod_err)) * (10**(b0_mod+b0_mod_err))


    # Plot
    fig, axes = plt.subplots(1, 1, figsize=(8,6), 
                            #  gridspec_kw={'height_ratios': ratio, 'hspace': 0.05},
                            #  sharex=True
                             )
    
    ax0 = axes
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_title(f"All planets")
    ax0.set_ylabel(f"{y_name} {y_symbol} {y_unit}".strip())

        #Calculated fits
    ax0.plot(x_fit, y_fit_mod, GASTLI_color,
                    label=fr"$\propto M_p^{{{b1_mod:.2f}}}$ [{GASTLI_label}]")
    ax0.fill_between(x_fit, y_fit_mod_lower, y_fit_mod_upper, color=GASTLI_color, alpha=0.2)

    ax0.plot(x_fit, y_fit_obs, planetsynth_color,
                    label=fr"$\propto M_p^{{{b1_obs:.2f}}}$ [{planetsynth_label}]")
    ax0.fill_between(x_fit, y_fit_obs_lower, y_fit_obs_upper, color=planetsynth_color, alpha=0.2)

    # # Thorngren & Howard fits
    # if y_symbol == y_symbols[0]:
    #     y_fit_thorngren, y_fit_thorngren_lower, y_fit_thorngren_upper = Zp_rel_Thorngren(x_fit)
    #     ax0.plot(x_fit, y_fit_thorngren, 'g-',
    #                 label=fr"$\propto M_p^{{-0.45}}$ [T16]")
    #     ax0.fill_between(x_fit, y_fit_thorngren_lower, y_fit_thorngren_upper, color='g', alpha=0.2)

    #     y_fit_howard, y_fit_howard_lower, y_fit_howard_upper = Zp_rel_Howard(x_fit)
    #     ax0.plot(x_fit, y_fit_howard, 'r-',
    #                 label=fr"$\propto M_p^{{-0.57}}$ [H25]")
    #     ax0.fill_between(x_fit, y_fit_howard_lower, y_fit_howard_upper, color='r', alpha=0.2)

    # elif y_symbol == y_symbols[1]:
    #     y_fit_thorngren, y_fit_thorngren_lower, y_fit_thorngren_upper = Mz_Thorngren(x_fit)
    #     ax0.plot(x_fit, y_fit_thorngren, 'g-',
    #                 label=fr"$\propto M_p^{{0.61}}$ [T16]")
    #     ax0.fill_between(x_fit, y_fit_thorngren_lower, y_fit_thorngren_upper, color='g', alpha=0.2)

    if y_symbol == y_symbols[0]:
        # ax0.set_ylim(y_lim_Zp_rel_upper)
        ax0.legend(loc="upper left", 
                #    ncol=2
                   )
    elif y_symbol == y_symbols[1]:
        # ax0.set_ylim(y_lim_Mz_upper)
        ax0.legend(loc="upper left", 
                #    ncol=2
                   )

    ax0.set_xlim(x_lim)
    ax0.set_xlabel(fr"Mass $M_p$ [$M_J$]")

    # fig.savefig(f"img/3_all_{y_obs_label}_empty.pdf", bbox_inches='tight')
    # fig.savefig(f"img/3_all_{y_obs_label}_empty.png", bbox_inches='tight', dpi=300)
# 

# %%

# %%
#Plot 2) only FGK for both AKZM24 and MH25

df = df_FGK
for i in range(len(y_obs_labels)):
    # Get labels for this iteration
    y_obs_label = y_obs_labels[i]
    y_obs_err_label = y_obs_err_labels[i]

    y_mod_label = y_mod_labels[i]
    y_mod_err_label = y_mod_err_labels[i]

    y_name = y_names[i]
    y_symbol = y_symbols[i]
    y_unit = y_units[i]

    # Data
    lin_x_obs = df[x_obs_label]
    lin_x_obs_err = pd.Series(
        [max(df.loc[index][x_obs_label+'_err1'], np.abs(df.loc[index][x_obs_label+'_err2'])) for index in df.index],
        index=df.index, name=x_obs_label+'_err')
    lin_y_obs = df[y_obs_label]
    lin_y_obs_err = df[y_obs_err_label]
    lin_y_mod = df[y_mod_label]
    lin_y_mod_err = df[y_mod_err_label]

    # ODR for observed
    log_x_obs = np.log10(lin_x_obs)
    log_x_obs_err = lin_x_obs_err / (lin_x_obs * np.log(10))
    log_y_obs = np.log10(lin_y_obs)
    log_y_obs_err = lin_y_obs_err / (lin_y_obs * np.log(10))

    loglog_data_obs = odr.RealData(log_x_obs, log_y_obs, sx=log_x_obs_err, sy=log_y_obs_err)
    odr_loglog_obs = odr.ODR(data=loglog_data_obs, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_obs = odr_loglog_obs.run()
    b0_obs, b1_obs = output_loglog_obs.beta
    b0_obs_err, b1_obs_err = output_loglog_obs.sd_beta

    # ODR for model
    log_y_mod = np.log10(lin_y_mod)
    log_y_mod_err = lin_y_mod_err / (lin_y_mod * np.log(10))
    loglog_data_mod = odr.RealData(log_x_obs, log_y_mod, sx=log_x_obs_err, sy=log_y_mod_err)
    odr_loglog_mod = odr.ODR(data=loglog_data_mod, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_mod = odr_loglog_mod.run()
    b0_mod, b1_mod = output_loglog_mod.beta
    b0_mod_err, b1_mod_err = output_loglog_mod.sd_beta

    # Fit lines
    x_fit = np.logspace(np.log10(min(lin_x_obs)), np.log10(max(lin_x_obs)), 200)

    y_fit_obs = (x_fit**b1_obs) * (10**b0_obs)
    y_fit_obs_lower = ((x_fit)**(b1_obs-b1_obs_err)) * (10**(b0_obs-b0_obs_err))
    y_fit_obs_upper = ((x_fit)**(b1_obs+b1_obs_err)) * (10**(b0_obs+b0_obs_err))

    y_fit_mod = (x_fit**b1_mod) * (10**b0_mod)
    y_fit_mod_lower = ((x_fit)**(b1_mod-b1_mod_err)) * (10**(b0_mod-b0_mod_err))
    y_fit_mod_upper = ((x_fit)**(b1_mod+b1_mod_err)) * (10**(b0_mod+b0_mod_err))

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': ratio, 'hspace': 0.05}, sharex=True)
    ax0 = axes[0]
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_title(f"Planets around FGK Stars")
    ax0.set_ylabel(f"{y_name} {y_symbol} {y_unit}".strip())

    ax0.errorbar(lin_x_obs, lin_y_obs, 
                 xerr=lin_x_obs_err, yerr=lin_y_obs_err,
                 fmt='s',
                 color=planetsynth_color,
                 ecolor=planetsynth_color,
                 markeredgecolor='black',
                 markeredgewidth=1,
                 capsize=3
                )   

    ax0.plot(x_fit, y_fit_obs, planetsynth_color,
                    label=fr"$\propto M_p^{{{b1_obs:.2f}}}$ [{planetsynth_label}]")
    ax0.fill_between(x_fit, y_fit_obs_lower, y_fit_obs_upper, color=planetsynth_color, alpha=0.2)

    ax0.errorbar(lin_x_obs, lin_y_mod,
                 xerr=lin_x_obs_err, yerr=lin_y_mod_err,
                 fmt='s',
                 color=GASTLI_color,
                 ecolor=GASTLI_color,
                 markeredgecolor='black',
                 markeredgewidth=1,
                 capsize=3
                )

    ax0.plot(x_fit, y_fit_mod, GASTLI_color,
                    label=fr"$\propto M_p^{{{b1_mod:.2f}}}$ [{GASTLI_label}]")
    ax0.fill_between(x_fit, y_fit_mod_lower, y_fit_mod_upper, color=GASTLI_color, alpha=0.2)

    if y_symbol == y_symbols[0]:
        ax0.set_ylim(y_lim_Zp_rel_upper)
        ax0.legend(loc="lower left")
    elif y_symbol == y_symbols[1]:
        ax0.legend(loc="lower right")
    # ax0.legend()

    # Bottom plot: differences
    ax1 = axes[1]
    ax1.scatter(lin_x_obs, lin_y_mod - lin_y_obs, color='k', s=30, label=f"[{GASTLI_label}] - [{planetsynth_label}]")
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    if y_symbol == y_symbols[0]:
        ax1.set_ylabel(r"$(Z_{p,\text{mod}}- Z_p)/Z_*$ " f"{y_unit}")
        # ax1.set_ylim(1e-1, 2e1)
        ax1.set_ylim(y_lim_Zp_rel)
        # ax1.set_yticks([1e-1, 1e0, 1e1])
        # ax1.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
        ax1.set_yticks([1e0, 1e1])
        ax1.set_yticklabels([r'$10^{0}$', r'$10^{1}$'])
    elif y_symbol == y_symbols[1]:
        ax1.set_ylabel(r"$(M_{Z,\text{mod}} - M_Z)$ " f"{y_unit}")
        # ax1.set_ylim(1e0, 3e2)
        ax1.set_ylim(y_lim_Mz)
        ax1.set_yticks([1e0, 1e1, 1e2])
        ax1.set_yticklabels([r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'])
    ax1.set_xlim(x_lim)
    # ax1.set_ylabel(r"$\Delta$" fr"{y_symbol}")
    ax1.set_xlabel(fr"Mass $M_p$ [$M_J$]")
    # ax1.legend()

    plt.tight_layout()
    plt.show()
    fig.savefig(f"img/3_FGK_{y_obs_label}.pdf", bbox_inches='tight')




#Plot 3) only M for both AKZM24 and MH25

df = df_M
for i in range(len(y_obs_labels)):
    # Get labels for this iteration
    y_obs_label = y_obs_labels[i]
    y_obs_err_label = y_obs_err_labels[i]

    y_mod_label = y_mod_labels[i]
    y_mod_err_label = y_mod_err_labels[i]

    y_name = y_names[i]
    y_symbol = y_symbols[i]
    y_unit = y_units[i]

    # Data
    lin_x_obs = df[x_obs_label]
    lin_x_obs_err = pd.Series(
        [max(df.loc[index][x_obs_label+'_err1'], np.abs(df.loc[index][x_obs_label+'_err2'])) for index in df.index],
        index=df.index, name=x_obs_label+'_err')
    lin_y_obs = df[y_obs_label]
    lin_y_obs_err = df[y_obs_err_label]
    lin_y_mod = df[y_mod_label]
    lin_y_mod_err = df[y_mod_err_label]

    # ODR for observed
    log_x_obs = np.log10(lin_x_obs)
    log_x_obs_err = lin_x_obs_err / (lin_x_obs * np.log(10))
    log_y_obs = np.log10(lin_y_obs)
    log_y_obs_err = lin_y_obs_err / (lin_y_obs * np.log(10))

    loglog_data_obs = odr.RealData(log_x_obs, log_y_obs, sx=log_x_obs_err, sy=log_y_obs_err)
    odr_loglog_obs = odr.ODR(data=loglog_data_obs, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_obs = odr_loglog_obs.run()
    b0_obs, b1_obs = output_loglog_obs.beta
    b0_obs_err, b1_obs_err = output_loglog_obs.sd_beta

    # ODR for model
    log_y_mod = np.log10(lin_y_mod)
    log_y_mod_err = lin_y_mod_err / (lin_y_mod * np.log(10))
    loglog_data_mod = odr.RealData(log_x_obs, log_y_mod, sx=log_x_obs_err, sy=log_y_mod_err)
    odr_loglog_mod = odr.ODR(data=loglog_data_mod, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_mod = odr_loglog_mod.run()
    b0_mod, b1_mod = output_loglog_mod.beta
    b0_mod_err, b1_mod_err = output_loglog_mod.sd_beta

    # Fit lines
    x_fit = np.logspace(np.log10(min(lin_x_obs)), np.log10(max(lin_x_obs)), 200)

    y_fit_obs = (x_fit**b1_obs) * (10**b0_obs)
    y_fit_obs_lower = ((x_fit)**(b1_obs-b1_obs_err)) * (10**(b0_obs-b0_obs_err))
    y_fit_obs_upper = ((x_fit)**(b1_obs+b1_obs_err)) * (10**(b0_obs+b0_obs_err))

    y_fit_mod = (x_fit**b1_mod) * (10**b0_mod)
    y_fit_mod_lower = ((x_fit)**(b1_mod-b1_mod_err)) * (10**(b0_mod-b0_mod_err))
    y_fit_mod_upper = ((x_fit)**(b1_mod+b1_mod_err)) * (10**(b0_mod+b0_mod_err))

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': ratio, 'hspace': 0.05}, sharex=True)
    ax0 = axes[0]
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_title(f"Planets around M Dwarfs")
    ax0.set_ylabel(f"{y_name} {y_symbol} {y_unit}".strip())

    ax0.errorbar(lin_x_obs, lin_y_obs, 
                 xerr=lin_x_obs_err, yerr=lin_y_obs_err,
                 fmt='s',
                 color=planetsynth_color,
                 ecolor=planetsynth_color,
                 markeredgecolor='black',
                 markeredgewidth=1,
                 capsize=3
                )   

    ax0.plot(x_fit, y_fit_obs, planetsynth_color,
                    label=fr"$\propto M_p^{{{b1_obs:.2f}}}$ [{planetsynth_label}]")
    ax0.fill_between(x_fit, y_fit_obs_lower, y_fit_obs_upper, color=planetsynth_color, alpha=0.2)

    ax0.errorbar(lin_x_obs, lin_y_mod,
                 xerr=lin_x_obs_err, yerr=lin_y_mod_err,
                 fmt='s',
                 color=GASTLI_color,
                 ecolor=GASTLI_color,
                 markeredgecolor='black',
                 markeredgewidth=1,
                 capsize=3
                )

    ax0.plot(x_fit, y_fit_mod, GASTLI_color,
                    label=fr"$\propto M_p^{{{b1_mod:.2f}}}$ [{GASTLI_label}]")
    ax0.fill_between(x_fit, y_fit_mod_lower, y_fit_mod_upper, color=GASTLI_color, alpha=0.2)

    if y_symbol == y_symbols[0]:
        ax0.legend(loc="lower left")
    elif y_symbol == y_symbols[1]:
        ax0.legend(loc="lower right")
    # ax0.legend()

    # Bottom plot: differences
    ax1 = axes[1]
    ax1.scatter(lin_x_obs, lin_y_mod - lin_y_obs, color='k', s=30, label=f"[{GASTLI_label}] - [{planetsynth_label}]")
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    if y_symbol == y_symbols[0]:
        ax1.set_ylabel(r"$(Z_{p,\text{mod}}- Z_p)/Z_*$ " f"{y_unit}")
        # ax1.set_ylim(1e-1, 2e1)
        ax1.set_ylim(y_lim_Zp_rel)
        # ax1.set_yticks([1e-1, 1e0, 1e1])
        # ax1.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
        ax1.set_yticks([1e0, 1e1])
        ax1.set_yticklabels([r'$10^{0}$', r'$10^{1}$'])
    elif y_symbol == y_symbols[1]:
        ax1.set_ylabel(r"$(M_{Z,\text{mod}} - M_Z)$ " f"{y_unit}")
        # ax1.set_ylim(1e0, 3e2)
        ax1.set_ylim(y_lim_Mz)
        ax1.set_yticks([1e0, 1e1, 1e2])
        ax1.set_yticklabels([r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'])
    ax1.set_xlim(x_lim)
    # ax1.set_ylabel(r"$\Delta$" fr"{y_symbol}")
    ax1.set_xlabel(fr"Mass $M_p$ [$M_J$]")
    # ax1.legend()

    plt.tight_layout()
    plt.show()
    fig.savefig(f"img/3_M_{y_obs_label}.pdf", bbox_inches='tight')
# %%
#Plot 4) FKG, M for both A24 and MH25

alpha=1
for i in range(len(y_obs_labels)):
    # Get labels for this iteration
    y_obs_label = y_obs_labels[i]
    y_obs_err_label = y_obs_err_labels[i]

    y_mod_label = y_mod_labels[i]
    y_mod_err_label = y_mod_err_labels[i]

    y_name = y_names[i]
    y_symbol = y_symbols[i]
    y_unit = y_units[i]


    # Data for FGK
    lin_x_obs_FGK = df_FGK[x_obs_label]
    lin_x_obs_err_FGK = pd.Series(
        [max(df_FGK.loc[index][x_obs_label+'_err1'], np.abs(df_FGK.loc[index][x_obs_label+'_err2'])) for index in df_FGK.index],
        index=df_FGK.index, name=x_obs_label+'_err')
    lin_y_obs_FGK = df_FGK[y_obs_label]
    lin_y_obs_err_FGK = df_FGK[y_obs_err_label]
    lin_y_mod_FGK = df_FGK[y_mod_label]
    lin_y_mod_err_FGK = df_FGK[y_mod_err_label]

    # ODR for observed FGK
    log_x_obs_FGK = np.log10(lin_x_obs_FGK) 
    log_x_obs_err_FGK = lin_x_obs_err_FGK / (lin_x_obs_FGK * np.log(10))
    log_y_obs_FGK = np.log10(lin_y_obs_FGK)
    log_y_obs_err_FGK = lin_y_obs_err_FGK / (lin_y_obs_FGK * np.log(10))

    loglog_data_obs_FGK = odr.RealData(log_x_obs_FGK, log_y_obs_FGK, sx=log_x_obs_err_FGK, sy=log_y_obs_err_FGK)
    odr_loglog_obs_FGK = odr.ODR(data=loglog_data_obs_FGK, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_obs_FGK = odr_loglog_obs_FGK.run()
    b0_obs_FGK, b1_obs_FGK = output_loglog_obs_FGK.beta
    b0_obs_err_FGK, b1_obs_err_FGK = output_loglog_obs_FGK.sd_beta

    # ODR for model FGK
    log_y_mod_FGK = np.log10(lin_y_mod_FGK)
    log_y_mod_err_FGK = lin_y_mod_err_FGK / (lin_y_mod_FGK * np.log(10))
    loglog_data_mod_FGK = odr.RealData(log_x_obs_FGK, log_y_mod_FGK, sx=log_x_obs_err_FGK, sy=log_y_mod_err_FGK)
    odr_loglog_mod_FGK = odr.ODR(data=loglog_data_mod_FGK, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_mod_FGK = odr_loglog_mod_FGK.run()
    b0_mod_FGK, b1_mod_FGK = output_loglog_mod_FGK.beta
    b0_mod_err_FGK, b1_mod_err_FGK = output_loglog_mod_FGK.sd_beta

    # Fit lines for FGK
    x_fit_FGK = np.logspace(np.log10(min(lin_x_obs_FGK)), np.log10(max(lin_x_obs_FGK)), 200)
    print(f"y FGK= {y_name}\nMass range: [{min(lin_x_obs_FGK):.2f}, {max(lin_x_obs_FGK):.2f}] M_J\n")
    
    print(f"Observed FGK: b0 = {b0_obs_FGK:.2f} +/- {b0_obs_err_FGK:.2f}, b1 = {b1_obs_FGK:.2f} +/- {b1_obs_err_FGK:.2f}")
    print(f"y = ({10**b0_obs_FGK:.2f} +/- {np.log(10) * 10**b0_obs_FGK * b0_obs_err_FGK:.2f}) * M_p ^ ({b1_obs_FGK:.2f} +/- {b1_obs_err_FGK:.2f})\n")

    print(f"Model FGK: b0 = {b0_mod_FGK:.2f} +/- {b0_mod_err_FGK:.2f}, b1 = {b1_mod_FGK:.2f} +/- {b1_mod_err_FGK:.2f}")
    print(f"y = ({10**b0_mod_FGK:.2f} +/- {np.log(10) * 10**b0_mod_FGK * b0_mod_err_FGK:.2f}) * M_p ^ ({b1_mod_FGK:.2f} +/- {b1_mod_err_FGK:.2f})\n") 

    y_fit_obs_FGK = (x_fit_FGK**b1_obs_FGK) * (10**b0_obs_FGK)
    y_fit_obs_lower_FGK = ((x_fit_FGK)**(b1_obs_FGK-b1_obs_err_FGK)) * (10**(b0_obs_FGK-b0_obs_err_FGK))
    y_fit_obs_upper_FGK = ((x_fit_FGK)**(b1_obs_FGK+b1_obs_err_FGK)) * (10**(b0_obs_FGK+b0_obs_err_FGK))

    y_fit_mod_FGK = (x_fit_FGK**b1_mod_FGK) * (10**b0_mod_FGK)
    y_fit_mod_lower_FGK = ((x_fit_FGK)**(b1_mod_FGK-b1_mod_err_FGK)) * (10**(b0_mod_FGK-b0_mod_err_FGK))
    y_fit_mod_upper_FGK = ((x_fit_FGK)**(b1_mod_FGK+b1_mod_err_FGK)) * (10**(b0_mod_FGK+b0_mod_err_FGK))


    # Data for M
    lin_x_obs_M = df_M[x_obs_label]
    lin_x_obs_err_M = pd.Series(
        [max(df_M.loc[index][x_obs_label+'_err1'], np.abs(df_M.loc[index][x_obs_label+'_err2'])) for index in df_M.index],
        index=df_M.index, name=x_obs_label+'_err')
    lin_y_obs_M = df_M[y_obs_label]
    lin_y_obs_err_M = df_M[y_obs_err_label]
    lin_y_mod_M = df_M[y_mod_label]
    lin_y_mod_err_M = df_M[y_mod_err_label]

    # ODR for observed M
    log_x_obs_M = np.log10(lin_x_obs_M)
    log_x_obs_err_M = lin_x_obs_err_M / (lin_x_obs_M * np.log(10))
    log_y_obs_M = np.log10(lin_y_obs_M)
    log_y_obs_err_M = lin_y_obs_err_M / (lin_y_obs_M * np.log(10))

    loglog_data_obs_M = odr.RealData(log_x_obs_M, log_y_obs_M, sx=log_x_obs_err_M, sy=log_y_obs_err_M)
    odr_loglog_obs_M = odr.ODR(data=loglog_data_obs_M, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_obs_M = odr_loglog_obs_M.run()
    b0_obs_M, b1_obs_M = output_loglog_obs_M.beta
    b0_obs_err_M, b1_obs_err_M = output_loglog_obs_M.sd_beta

    # ODR for model M
    log_y_mod_M = np.log10(lin_y_mod_M)
    log_y_mod_err_M = lin_y_mod_err_M / (lin_y_mod_M * np.log(10))
    loglog_data_mod_M = odr.RealData(log_x_obs_M, log_y_mod_M, sx=log_x_obs_err_M, sy=log_y_mod_err_M)
    odr_loglog_mod_M = odr.ODR(data=loglog_data_mod_M, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_mod_M = odr_loglog_mod_M.run()
    b0_mod_M, b1_mod_M = output_loglog_mod_M.beta
    b0_mod_err_M, b1_mod_err_M = output_loglog_mod_M.sd_beta

    # Fit lines for M
    x_fit_M = np.logspace(np.log10(min(lin_x_obs_M)), np.log10(max(lin_x_obs_M)), 200)
    print(f"y M = {y_name}\nMass range: [{min(lin_x_obs_M):.2f}, {max(lin_x_obs_M):.2f}] M_J\n")

    print(f"Observed M: b0 = {b0_obs_M:.2f} +/- {b0_obs_err_M:.2f}, b1 = {b1_obs_M:.2f} +/- {b1_obs_err_M:.2f}")
    print(f"y = ({10**b0_obs_M:.2f} +/- {np.log(10) * 10**b0_obs_M * b0_obs_err_M:.2f}) * M_p ^ ({b1_obs_M:.2f} +/- {b1_obs_err_M:.2f})\n")

    print(f"Model M: b0 = {b0_mod_M:.2f} +/- {b0_mod_err_M:.2f}, b1 = {b1_mod_M:.2f} +/- {b1_mod_err_M:.2f}")
    print(f"y = ({10**b0_mod_M:.2f} +/- {np.log(10) * 10**b0_mod_M * b0_mod_err_M:.2f}) * M_p ^ ({b1_mod_M:.2f} +/- {b1_mod_err_M:.2f})\n")

    
    y_fit_obs_M = (x_fit_M**b1_obs_M) * (10**b0_obs_M)
    y_fit_obs_lower_M = ((x_fit_M)**(b1_obs_M-b1_obs_err_M)) * (10**(b0_obs_M-b0_obs_err_M))
    y_fit_obs_upper_M = ((x_fit_M)**(b1_obs_M+b1_obs_err_M)) * (10**(b0_obs_M+b0_obs_err_M))

    y_fit_mod_M = (x_fit_M**b1_mod_M) * (10**b0_mod_M)
    y_fit_mod_lower_M = ((x_fit_M)**(b1_mod_M-b1_mod_err_M)) * (10**(b0_mod_M-b0_mod_err_M))
    y_fit_mod_upper_M = ((x_fit_M)**(b1_mod_M+b1_mod_err_M)) * (10**(b0_mod_M+b0_mod_err_M))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), 
                             gridspec_kw={'height_ratios': ratio, 'hspace': 0.05, 'wspace': 0.1},
                             sharex='col', sharey='row')
    

    # --- FGK (left column) ---
    ax0 = axes[0, 0]
    ax0.set_xscale('log')
    ax0.set_yscale('log')   
    ax0.set_title(f"Planets around FGK Stars")
    ax0.set_ylabel(f"{y_name} {y_symbol} {y_unit}".strip())

    ax0.errorbar(lin_x_obs_FGK, lin_y_obs_FGK,
                 xerr=lin_x_obs_err_FGK, yerr=lin_y_obs_err_FGK,
                 fmt='s',
                 color=planetsynth_color,
                 ecolor=planetsynth_color,
                 markeredgecolor='black',
                 markeredgewidth=1,
                 capsize=3,
                 alpha=alpha
                )
    
 

    ax0.errorbar(lin_x_obs_FGK, lin_y_mod_FGK,
                xerr=lin_x_obs_err_FGK, yerr=lin_y_mod_err_FGK,
                fmt='s',
                color=GASTLI_color,
                ecolor=GASTLI_color,
                markeredgecolor='black',
                markeredgewidth=1,
                capsize=3,
                alpha=alpha
                )
    
    ax0.plot(x_fit_FGK, y_fit_mod_FGK, GASTLI_color,
                    label=fr"$\propto M_p^{{{b1_mod_FGK:.2f}}}$ [{GASTLI_label}]")
    ax0.fill_between(x_fit_FGK, y_fit_mod_lower_FGK, y_fit_mod_upper_FGK, color=GASTLI_color, alpha=0.2)

    ax0.plot(x_fit_FGK, y_fit_obs_FGK, planetsynth_color,
                    label=fr"$\propto M_p^{{{b1_obs_FGK:.2f}}}$ [{planetsynth_label}]") 
    ax0.fill_between(x_fit_FGK, y_fit_obs_lower_FGK, y_fit_obs_upper_FGK, color=planetsynth_color, alpha=0.2)


    if y_symbol == y_symbols[0]:
        ax0.legend(loc="upper left")
    elif y_symbol == y_symbols[1]:
        ax0.legend(loc="upper left")

    # --- Bottom FGK plot (row 1, col 0) ---
    ax1 = axes[1, 0]
    ax1.scatter(lin_x_obs_FGK, lin_y_mod_FGK - lin_y_obs_FGK, color='k', s=30,
                label=f"[{GASTLI_label}] - [{planetsynth_label}]")
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    if y_symbol == y_symbols[0]:
        ax1.set_ylabel(r"$(Z_{p,\text{mod}}- Z_p)/Z_*$ " f"{y_unit}")
        ax1.set_ylim(y_lim_Zp_rel)
        ax1.set_yticks([1e0, 1e1])
        ax1.set_yticklabels([r'$10^{0}$', r'$10^{1}$'])
    elif y_symbol == y_symbols[1]:
        ax1.set_ylabel(r"$(M_{Z,\text{mod}} - M_Z)$ " f"{y_unit}")
        ax1.set_ylim(y_lim_Mz)
        ax1.set_yticks([1e0, 1e1, 1e2])
        ax1.set_yticklabels([r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'])
    ax1.set_xlim(x_lim)
    ax1.set_xlabel(fr"Mass $M_p$ [$M_J$]")



    # --- M Dwarfs (right column) ---
    ax2 = axes[0, 1]
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title(f"Planets around M Dwarfs")
    # ax2.set_ylabel(f"{y_name} {y_symbol} {y_unit}".strip())

    ax2.errorbar(lin_x_obs_M, lin_y_obs_M,
                 xerr=lin_x_obs_err_M, yerr=lin_y_obs_err_M,
                 fmt='s',
                 color=planetsynth_color,
                 ecolor=planetsynth_color,
                 markeredgecolor='black',
                 markeredgewidth=1,
                 capsize=3,
                 alpha=alpha
                )


    ax2.errorbar(lin_x_obs_M, lin_y_mod_M,
                 xerr=lin_x_obs_err_M, yerr=lin_y_mod_err_M,
                 fmt='s',
                 color=GASTLI_color,
                 ecolor=GASTLI_color,
                 markeredgecolor='black',
                 markeredgewidth=1,
                 capsize=3,
                 alpha=alpha
                )
    
    ax2.plot(x_fit_M, y_fit_mod_M, GASTLI_color,
                    label=fr"$\propto M_p^{{{b1_mod_M:.2f}}}$ [{GASTLI_label}]")
    ax2.fill_between(x_fit_M, y_fit_mod_lower_M, y_fit_mod_upper_M, color=GASTLI_color, alpha=0.2)

    ax2.plot(x_fit_M, y_fit_obs_M, planetsynth_color,
                    label=fr"$\propto M_p^{{{b1_obs_M:.2f}}}$ [{planetsynth_label}]")
    ax2.fill_between(x_fit_M, y_fit_obs_lower_M, y_fit_obs_upper_M, color=planetsynth_color, alpha=0.2)

    if y_symbol == y_symbols[0]:
        ax2.set_ylim(y_lim_Zp_rel_upper)
        ax2.legend(loc="upper left")
    elif y_symbol == y_symbols[1]:
        ax2.set_ylim(y_lim_Mz_upper)
        ax2.legend(loc="upper left")


    # --- Bottom M plot (row 1, col 1) ---
    ax3 = axes[1, 1]
    ax3.scatter(lin_x_obs_M, lin_y_mod_M - lin_y_obs_M, color='k', s=30,
                label=f"[{GASTLI_label}] - [{planetsynth_label}]")
    ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    if y_symbol == y_symbols[0]:
        # ax3.set_ylabel(r"$(Z_{p,\text{mod}}- Z_p)/Z_*$ " f"{y_unit}")
        ax3.set_ylim(y_lim_Zp_rel)
        ax3.set_yticks([1e0, 1e1])
        ax3.set_yticklabels([r'$10^{0}$', r'$10^{1}$'])
    elif y_symbol == y_symbols[1]:
        # ax3.set_ylabel(r"$(M_{Z,\text{mod}} - M_Z)$ " f"{y_unit}")
        ax3.set_ylim(y_lim_Mz)
        ax3.set_yticks([1e0, 1e1, 1e2])
        ax3.set_yticklabels([r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'])
    ax3.set_xlim(x_lim)
    ax3.set_xlabel(fr"Mass $M_p$ [$M_J$]")

    plt.tight_layout()
    plt.show()
    # fig.savefig(f"img/3_FGK_M_{y_obs_label}.pdf", bbox_inches='tight')
    # fig.savefig(f"img/3_FGK_M_{y_obs_label}.png", bbox_inches='tight', dpi=300)
    

# %%
#Plot 4) FKG, M for both A24 and MH25 (empty)

for i in range(len(y_obs_labels)):
    # Get labels for this iteration
    y_obs_label = y_obs_labels[i]
    y_obs_err_label = y_obs_err_labels[i]

    y_mod_label = y_mod_labels[i]
    y_mod_err_label = y_mod_err_labels[i]

    y_name = y_names[i]
    y_symbol = y_symbols[i]
    y_unit = y_units[i]


    # Data for FGK
    lin_x_obs_FGK = df_FGK[x_obs_label]
    lin_x_obs_err_FGK = pd.Series(
        [max(df_FGK.loc[index][x_obs_label+'_err1'], np.abs(df_FGK.loc[index][x_obs_label+'_err2'])) for index in df_FGK.index],
        index=df_FGK.index, name=x_obs_label+'_err')
    lin_y_obs_FGK = df_FGK[y_obs_label]
    lin_y_obs_err_FGK = df_FGK[y_obs_err_label]
    lin_y_mod_FGK = df_FGK[y_mod_label]
    lin_y_mod_err_FGK = df_FGK[y_mod_err_label]

    # ODR for observed FGK
    log_x_obs_FGK = np.log10(lin_x_obs_FGK) 
    log_x_obs_err_FGK = lin_x_obs_err_FGK / (lin_x_obs_FGK * np.log(10))
    log_y_obs_FGK = np.log10(lin_y_obs_FGK)
    log_y_obs_err_FGK = lin_y_obs_err_FGK / (lin_y_obs_FGK * np.log(10))

    loglog_data_obs_FGK = odr.RealData(log_x_obs_FGK, log_y_obs_FGK, sx=log_x_obs_err_FGK, sy=log_y_obs_err_FGK)
    odr_loglog_obs_FGK = odr.ODR(data=loglog_data_obs_FGK, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_obs_FGK = odr_loglog_obs_FGK.run()
    b0_obs_FGK, b1_obs_FGK = output_loglog_obs_FGK.beta
    b0_obs_err_FGK, b1_obs_err_FGK = output_loglog_obs_FGK.sd_beta

    # ODR for model FGK
    log_y_mod_FGK = np.log10(lin_y_mod_FGK)
    log_y_mod_err_FGK = lin_y_mod_err_FGK / (lin_y_mod_FGK * np.log(10))
    loglog_data_mod_FGK = odr.RealData(log_x_obs_FGK, log_y_mod_FGK, sx=log_x_obs_err_FGK, sy=log_y_mod_err_FGK)
    odr_loglog_mod_FGK = odr.ODR(data=loglog_data_mod_FGK, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_mod_FGK = odr_loglog_mod_FGK.run()
    b0_mod_FGK, b1_mod_FGK = output_loglog_mod_FGK.beta
    b0_mod_err_FGK, b1_mod_err_FGK = output_loglog_mod_FGK.sd_beta

    # Fit lines for FGK
    x_fit_FGK = np.logspace(np.log10(min(lin_x_obs_FGK)), np.log10(max(lin_x_obs_FGK)), 200)
    print(f"y FGK= {y_name}\nMass range: [{min(lin_x_obs_FGK):.2f}, {max(lin_x_obs_FGK):.2f}] M_J\n")
    
    print(f"Observed FGK: b0 = {b0_obs_FGK:.2f} +/- {b0_obs_err_FGK:.2f}, b1 = {b1_obs_FGK:.2f} +/- {b1_obs_err_FGK:.2f}")
    print(f"y = ({10**b0_obs_FGK:.2f} +/- {np.log(10) * 10**b0_obs_FGK * b0_obs_err_FGK:.2f}) * M_p ^ ({b1_obs_FGK:.2f} +/- {b1_obs_err_FGK:.2f})\n")

    print(f"Model FGK: b0 = {b0_mod_FGK:.2f} +/- {b0_mod_err_FGK:.2f}, b1 = {b1_mod_FGK:.2f} +/- {b1_mod_err_FGK:.2f}")
    print(f"y = ({10**b0_mod_FGK:.2f} +/- {np.log(10) * 10**b0_mod_FGK * b0_mod_err_FGK:.2f}) * M_p ^ ({b1_mod_FGK:.2f} +/- {b1_mod_err_FGK:.2f})\n") 

    y_fit_obs_FGK = (x_fit_FGK**b1_obs_FGK) * (10**b0_obs_FGK)
    y_fit_obs_lower_FGK = ((x_fit_FGK)**(b1_obs_FGK-b1_obs_err_FGK)) * (10**(b0_obs_FGK-b0_obs_err_FGK))
    y_fit_obs_upper_FGK = ((x_fit_FGK)**(b1_obs_FGK+b1_obs_err_FGK)) * (10**(b0_obs_FGK+b0_obs_err_FGK))

    y_fit_mod_FGK = (x_fit_FGK**b1_mod_FGK) * (10**b0_mod_FGK)
    y_fit_mod_lower_FGK = ((x_fit_FGK)**(b1_mod_FGK-b1_mod_err_FGK)) * (10**(b0_mod_FGK-b0_mod_err_FGK))
    y_fit_mod_upper_FGK = ((x_fit_FGK)**(b1_mod_FGK+b1_mod_err_FGK)) * (10**(b0_mod_FGK+b0_mod_err_FGK))


    # Data for M
    lin_x_obs_M = df_M[x_obs_label]
    lin_x_obs_err_M = pd.Series(
        [max(df_M.loc[index][x_obs_label+'_err1'], np.abs(df_M.loc[index][x_obs_label+'_err2'])) for index in df_M.index],
        index=df_M.index, name=x_obs_label+'_err')
    lin_y_obs_M = df_M[y_obs_label]
    lin_y_obs_err_M = df_M[y_obs_err_label]
    lin_y_mod_M = df_M[y_mod_label]
    lin_y_mod_err_M = df_M[y_mod_err_label]

    # ODR for observed M
    log_x_obs_M = np.log10(lin_x_obs_M)
    log_x_obs_err_M = lin_x_obs_err_M / (lin_x_obs_M * np.log(10))
    log_y_obs_M = np.log10(lin_y_obs_M)
    log_y_obs_err_M = lin_y_obs_err_M / (lin_y_obs_M * np.log(10))

    loglog_data_obs_M = odr.RealData(log_x_obs_M, log_y_obs_M, sx=log_x_obs_err_M, sy=log_y_obs_err_M)
    odr_loglog_obs_M = odr.ODR(data=loglog_data_obs_M, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_obs_M = odr_loglog_obs_M.run()
    b0_obs_M, b1_obs_M = output_loglog_obs_M.beta
    b0_obs_err_M, b1_obs_err_M = output_loglog_obs_M.sd_beta

    # ODR for model M
    log_y_mod_M = np.log10(lin_y_mod_M)
    log_y_mod_err_M = lin_y_mod_err_M / (lin_y_mod_M * np.log(10))
    loglog_data_mod_M = odr.RealData(log_x_obs_M, log_y_mod_M, sx=log_x_obs_err_M, sy=log_y_mod_err_M)
    odr_loglog_mod_M = odr.ODR(data=loglog_data_mod_M, model=loglog_model, beta0=[-0.5, 0.5])
    output_loglog_mod_M = odr_loglog_mod_M.run()
    b0_mod_M, b1_mod_M = output_loglog_mod_M.beta
    b0_mod_err_M, b1_mod_err_M = output_loglog_mod_M.sd_beta

    # Fit lines for M
    x_fit_M = np.logspace(np.log10(min(lin_x_obs_M)), np.log10(max(lin_x_obs_M)), 200)
    print(f"y M = {y_name}\nMass range: [{min(lin_x_obs_M):.2f}, {max(lin_x_obs_M):.2f}] M_J\n")

    print(f"Observed M: b0 = {b0_obs_M:.2f} +/- {b0_obs_err_M:.2f}, b1 = {b1_obs_M:.2f} +/- {b1_obs_err_M:.2f}")
    print(f"y = ({10**b0_obs_M:.2f} +/- {np.log(10) * 10**b0_obs_M * b0_obs_err_M:.2f}) * M_p ^ ({b1_obs_M:.2f} +/- {b1_obs_err_M:.2f})\n")

    print(f"Model M: b0 = {b0_mod_M:.2f} +/- {b0_mod_err_M:.2f}, b1 = {b1_mod_M:.2f} +/- {b1_mod_err_M:.2f}")
    print(f"y = ({10**b0_mod_M:.2f} +/- {np.log(10) * 10**b0_mod_M * b0_mod_err_M:.2f}) * M_p ^ ({b1_mod_M:.2f} +/- {b1_mod_err_M:.2f})\n")

    
    y_fit_obs_M = (x_fit_M**b1_obs_M) * (10**b0_obs_M)
    y_fit_obs_lower_M = ((x_fit_M)**(b1_obs_M-b1_obs_err_M)) * (10**(b0_obs_M-b0_obs_err_M))
    y_fit_obs_upper_M = ((x_fit_M)**(b1_obs_M+b1_obs_err_M)) * (10**(b0_obs_M+b0_obs_err_M))

    y_fit_mod_M = (x_fit_M**b1_mod_M) * (10**b0_mod_M)
    y_fit_mod_lower_M = ((x_fit_M)**(b1_mod_M-b1_mod_err_M)) * (10**(b0_mod_M-b0_mod_err_M))
    y_fit_mod_upper_M = ((x_fit_M)**(b1_mod_M+b1_mod_err_M)) * (10**(b0_mod_M+b0_mod_err_M))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), 
                             gridspec_kw={
                            # 'height_ratios': ratio,
                            # 'hspace': 0.05,
                            'wspace': 0.1},
                            #  sharex='col', 
                             sharey='row')
    
    # --- FGK (left column) ---
    ax0 = axes[0]
    ax0.set_xscale('log')
    ax0.set_yscale('log')   
    ax0.set_title(f"Planets around FGK Stars")
    ax0.set_ylabel(f"{y_name} {y_symbol} {y_unit}".strip())
    
    ax0.plot(x_fit_FGK, y_fit_mod_FGK, GASTLI_color,
                    label=fr"$\propto M_p^{{{b1_mod_FGK:.2f}}}$ [{GASTLI_label}]")
    ax0.fill_between(x_fit_FGK, y_fit_mod_lower_FGK, y_fit_mod_upper_FGK, color=GASTLI_color, alpha=0.2)

    ax0.plot(x_fit_FGK, y_fit_obs_FGK, planetsynth_color,
                    label=fr"$\propto M_p^{{{b1_obs_FGK:.2f}}}$ [{planetsynth_label}]") 
    ax0.fill_between(x_fit_FGK, y_fit_obs_lower_FGK, y_fit_obs_upper_FGK, color=planetsynth_color, alpha=0.2)


    if y_symbol == y_symbols[0]:
        ax0.legend(loc="upper left")
    elif y_symbol == y_symbols[1]:
        ax0.legend(loc="upper left")

    ax0.set_xlim(x_lim)
    ax0.set_xlabel(fr"Mass $M_p$ [$M_J$]")


# --- M Dwarfs (right column) ---
    ax2 = axes[1]
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title(f"Planets around M Dwarfs")
    # ax2.set_ylabel(f"{y_name} {y_symbol} {y_unit}".strip())

    ax2.plot(x_fit_M, y_fit_mod_M, GASTLI_color,
                    label=fr"$\propto M_p^{{{b1_mod_M:.2f}}}$ [{GASTLI_label}]")
    ax2.fill_between(x_fit_M, y_fit_mod_lower_M, y_fit_mod_upper_M, color=GASTLI_color, alpha=0.2)

    ax2.plot(x_fit_M, y_fit_obs_M, planetsynth_color,
                    label=fr"$\propto M_p^{{{b1_obs_M:.2f}}}$ [{planetsynth_label}]")
    ax2.fill_between(x_fit_M, y_fit_obs_lower_M, y_fit_obs_upper_M, color=planetsynth_color, alpha=0.2)

    if y_symbol == y_symbols[0]:
        ax2.set_ylim(y_lim_Zp_rel_upper)
        ax2.legend(loc="upper left")
    elif y_symbol == y_symbols[1]:
        ax2.set_ylim(y_lim_Mz_upper)
        ax2.legend(loc="upper left")

    ax2.set_xlim(x_lim)
    ax2.set_xlabel(fr"Mass $M_p$ [$M_J$]")

    plt.tight_layout()
    plt.show()

    # fig.savefig(f"img/3_FGK_M_{y_obs_label}_empty.pdf", bbox_inches='tight')
    # fig.savefig(f"img/3_FGK_M_{y_obs_label}_empty.png", bbox_inches='tight', dpi=300)





# %%


# %%
# %%
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(
#     4, 2, figsize=(14, 14),
#     sharex='col', sharey='row',
#     gridspec_kw={'hspace': 0.08, 'wspace': 0.1}
# )

# # --- FGK (left column) ---
# df = df_FGK
# for i in range(len(y_obs_labels)):
#     # Main plot (row 2*i, col 0)
#     # ax_main = axes[2*i, 0]
#     ax0 = axes[2*i, 0]
#     # ... your plotting code for FGK main plot ...
#     if i == 0:
#         ax0.set_title("FGK")
#     ax0.set_ylabel(f"{y_names[i]} {y_symbols[i]} {y_units[i]}".strip())
#     # ... errorbar, plot, fill_between, legend, etc. ...
#     # Get labels for this iteration
#     y_obs_label = y_obs_labels[i]
#     y_obs_err_label = y_obs_err_labels[i]

#     y_mod_label = y_mod_labels[i]
#     y_mod_err_label = y_mod_err_labels[i]

#     y_name = y_names[i]
#     y_symbol = y_symbols[i]
#     y_unit = y_units[i]

#     # Data
#     lin_x_obs = df[x_obs_label]
#     lin_x_obs_err = pd.Series(
#         [max(df.loc[index][x_obs_label+'_err1'], np.abs(df.loc[index][x_obs_label+'_err2'])) for index in df.index],
#         index=df.index, name=x_obs_label+'_err')
#     lin_y_obs = df[y_obs_label]
#     lin_y_obs_err = df[y_obs_err_label]
#     lin_y_mod = df[y_mod_label]
#     lin_y_mod_err = df[y_mod_err_label]

#     # ODR for observed
#     log_x_obs = np.log10(lin_x_obs)
#     log_x_obs_err = lin_x_obs_err / (lin_x_obs * np.log(10))
#     log_y_obs = np.log10(lin_y_obs)
#     log_y_obs_err = lin_y_obs_err / (lin_y_obs * np.log(10))

#     loglog_data_obs = odr.RealData(log_x_obs, log_y_obs, sx=log_x_obs_err, sy=log_y_obs_err)
#     odr_loglog_obs = odr.ODR(data=loglog_data_obs, model=loglog_model, beta0=[-0.5, 0.5])
#     output_loglog_obs = odr_loglog_obs.run()
#     b0_obs, b1_obs = output_loglog_obs.beta
#     b0_obs_err, b1_obs_err = output_loglog_obs.sd_beta

#     # ODR for model
#     log_y_mod = np.log10(lin_y_mod)
#     log_y_mod_err = lin_y_mod_err / (lin_y_mod * np.log(10))
#     loglog_data_mod = odr.RealData(log_x_obs, log_y_mod, sx=log_x_obs_err, sy=log_y_mod_err)
#     odr_loglog_mod = odr.ODR(data=loglog_data_mod, model=loglog_model, beta0=[-0.5, 0.5])
#     output_loglog_mod = odr_loglog_mod.run()
#     b0_mod, b1_mod = output_loglog_mod.beta
#     b0_mod_err, b1_mod_err = output_loglog_mod.sd_beta

#     # Fit lines
#     x_fit = np.logspace(np.log10(min(lin_x_obs)), np.log10(max(lin_x_obs)), 200)

#     y_fit_obs = (x_fit**b1_obs) * (10**b0_obs)
#     y_fit_obs_lower = ((x_fit)**(b1_obs-b1_obs_err)) * (10**(b0_obs-b0_obs_err))
#     y_fit_obs_upper = ((x_fit)**(b1_obs+b1_obs_err)) * (10**(b0_obs+b0_obs_err))

#     y_fit_mod = (x_fit**b1_mod) * (10**b0_mod)
#     y_fit_mod_lower = ((x_fit)**(b1_mod-b1_mod_err)) * (10**(b0_mod-b0_mod_err))
#     y_fit_mod_upper = ((x_fit)**(b1_mod+b1_mod_err)) * (10**(b0_mod+b0_mod_err))

#     # Plot
#     fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': ratio, 'hspace': 0.05}, sharex=True)
#     ax0 = axes[0]
#     ax0.set_xscale('log')
#     ax0.set_yscale('log')
#     ax0.set_title(f"FGK")
#     ax0.set_ylabel(f"{y_name} {y_symbol} {y_unit}".strip())

#     ax0.errorbar(lin_x_obs, lin_y_obs, 
#                  xerr=lin_x_obs_err, yerr=lin_y_obs_err,
#                  fmt='s',
#                  color=planetsynth_color,
#                  ecolor=planetsynth_color,
#                  markeredgecolor='black',
#                  markeredgewidth=1,
#                  capsize=3
#                 )   

#     ax0.plot(x_fit, y_fit_obs, planetsynth_color,
#                     label=fr"$\propto M_p^{{{b1_obs:.2f}}}$ [{planetsynth_label}]")
#     ax0.fill_between(x_fit, y_fit_obs_lower, y_fit_obs_upper, color=planetsynth_color, alpha=0.2)

#     ax0.errorbar(lin_x_obs, lin_y_mod,
#                  xerr=lin_x_obs_err, yerr=lin_y_mod_err,
#                  fmt='s',
#                  color=GASTLI_color,
#                  ecolor=GASTLI_color,
#                  markeredgecolor='black',
#                  markeredgewidth=1,
#                  capsize=3
#                 )

#     ax0.plot(x_fit, y_fit_mod, GASTLI_color,
#                     label=fr"$\propto M_p^{{{b1_mod:.2f}}}$ [{GASTLI_label}]")
#     ax0.fill_between(x_fit, y_fit_mod_lower, y_fit_mod_upper, color=GASTLI_color, alpha=0.2)

#     ax0.legend()

#     # Difference plot (row 2*i+1, col 0)
#     ax1 = axes[2*i+1, 0]
#     # ... your plotting code for FGK diff plot ...
#     ax1.scatter(lin_x_obs, lin_y_mod - lin_y_obs, color='k', s=30, label=f"[{GASTLI_label}] - [{planetsynth_label}]")
#     ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
#     ax1.set_xscale('log')
#     ax1.set_yscale('log')
#     if y_symbol == y_symbols[0]:
#         ax1.set_ylabel(r"$(Z_{p,\text{mod}}- Z_p)/Z_*$ " f"{y_unit}")
#         ax1.set_ylim(1e-1, 1e1)
#         ax1.set_yticks([1e-1, 1e0, 1e1])
#         ax1.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
#     elif y_symbol == y_symbols[1]:
#         ax1.set_ylabel(r"$M_{Z,\text{mod}} - M_Z$ " f"{y_unit}")
#         ax1.set_ylim(1e0, 1e2)
#         ax1.set_yticks([1e0, 1e1, 1e2])
#         ax1.set_yticklabels([r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'])
#     ax1.set_xlim(left=1e-1)
#     # ax1.set_ylabel(r"$\Delta$" fr"{y_symbol}")
#     ax1.set_ylabel("Difference")  # or your custom label
#     ax1.set_xlabel(fr"Mass $M_p$ [$M_J$]")


# # --- M (right column) ---
# df = df_M
# for i in range(len(y_obs_labels)):
#     # Main plot (row 2*i, col 1)
#     ax0 = axes[2*i, 1]
#     # ... your plotting code for M main plot ...
#     # Get labels for this iteration
#     y_obs_label = y_obs_labels[i]
#     y_obs_err_label = y_obs_err_labels[i]

#     y_mod_label = y_mod_labels[i]
#     y_mod_err_label = y_mod_err_labels[i]

#     y_name = y_names[i]
#     y_symbol = y_symbols[i]
#     y_unit = y_units[i]

#     # Data
#     lin_x_obs = df[x_obs_label]
#     lin_x_obs_err = pd.Series(
#         [max(df.loc[index][x_obs_label+'_err1'], np.abs(df.loc[index][x_obs_label+'_err2'])) for index in df.index],
#         index=df.index, name=x_obs_label+'_err')
#     lin_y_obs = df[y_obs_label]
#     lin_y_obs_err = df[y_obs_err_label]
#     lin_y_mod = df[y_mod_label]
#     lin_y_mod_err = df[y_mod_err_label]

#     # ODR for observed
#     log_x_obs = np.log10(lin_x_obs)
#     log_x_obs_err = lin_x_obs_err / (lin_x_obs * np.log(10))
#     log_y_obs = np.log10(lin_y_obs)
#     log_y_obs_err = lin_y_obs_err / (lin_y_obs * np.log(10))

#     loglog_data_obs = odr.RealData(log_x_obs, log_y_obs, sx=log_x_obs_err, sy=log_y_obs_err)
#     odr_loglog_obs = odr.ODR(data=loglog_data_obs, model=loglog_model, beta0=[-0.5, 0.5])
#     output_loglog_obs = odr_loglog_obs.run()
#     b0_obs, b1_obs = output_loglog_obs.beta
#     b0_obs_err, b1_obs_err = output_loglog_obs.sd_beta

#     # ODR for model
#     log_y_mod = np.log10(lin_y_mod)
#     log_y_mod_err = lin_y_mod_err / (lin_y_mod * np.log(10))
#     loglog_data_mod = odr.RealData(log_x_obs, log_y_mod, sx=log_x_obs_err, sy=log_y_mod_err)
#     odr_loglog_mod = odr.ODR(data=loglog_data_mod, model=loglog_model, beta0=[-0.5, 0.5])
#     output_loglog_mod = odr_loglog_mod.run()
#     b0_mod, b1_mod = output_loglog_mod.beta
#     b0_mod_err, b1_mod_err = output_loglog_mod.sd_beta

#     # Fit lines
#     x_fit = np.logspace(np.log10(min(lin_x_obs)), np.log10(max(lin_x_obs)), 200)

#     y_fit_obs = (x_fit**b1_obs) * (10**b0_obs)
#     y_fit_obs_lower = ((x_fit)**(b1_obs-b1_obs_err)) * (10**(b0_obs-b0_obs_err))
#     y_fit_obs_upper = ((x_fit)**(b1_obs+b1_obs_err)) * (10**(b0_obs+b0_obs_err))

#     y_fit_mod = (x_fit**b1_mod) * (10**b0_mod)
#     y_fit_mod_lower = ((x_fit)**(b1_mod-b1_mod_err)) * (10**(b0_mod-b0_mod_err))
#     y_fit_mod_upper = ((x_fit)**(b1_mod+b1_mod_err)) * (10**(b0_mod+b0_mod_err))

#     # Plot
#     fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': ratio, 'hspace': 0.05}, sharex=True)
#     ax0 = axes[0]
#     ax0.set_xscale('log')
#     ax0.set_yscale('log')
#     ax0.set_title(f"FGK")
#     ax0.set_ylabel(f"{y_name} {y_symbol} {y_unit}".strip())

#     ax0.errorbar(lin_x_obs, lin_y_obs, 
#                  xerr=lin_x_obs_err, yerr=lin_y_obs_err,
#                  fmt='s',
#                  color=planetsynth_color,
#                  ecolor=planetsynth_color,
#                  markeredgecolor='black',
#                  markeredgewidth=1,
#                  capsize=3
#                 )   

#     ax0.plot(x_fit, y_fit_obs, planetsynth_color,
#                     label=fr"$\propto M_p^{{{b1_obs:.2f}}}$ [{planetsynth_label}]")
#     ax0.fill_between(x_fit, y_fit_obs_lower, y_fit_obs_upper, color=planetsynth_color, alpha=0.2)

#     ax0.errorbar(lin_x_obs, lin_y_mod,
#                  xerr=lin_x_obs_err, yerr=lin_y_mod_err,
#                  fmt='s',
#                  color=GASTLI_color,
#                  ecolor=GASTLI_color,
#                  markeredgecolor='black',
#                  markeredgewidth=1,
#                  capsize=3
#                 )

#     ax0.plot(x_fit, y_fit_mod, GASTLI_color,
#                     label=fr"$\propto M_p^{{{b1_mod:.2f}}}$ [{GASTLI_label}]")
#     ax0.fill_between(x_fit, y_fit_mod_lower, y_fit_mod_upper, color=GASTLI_color, alpha=0.2)

#     ax0.legend()

#     if i == 0:
#         ax0.set_title("M")
#     # Only set y-label for leftmost plots
#     # ... errorbar, plot, fill_between, legend, etc. ...

#     # Difference plot (row 2*i+1, col 1)
#     ax1 = axes[2*i+1, 1]
#     # ... your plotting code for M diff plot ...
#     ax1.scatter(lin_x_obs, lin_y_mod - lin_y_obs, color='k', s=30, label=f"[{GASTLI_label}] - [{planetsynth_label}]")
#     ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
#     ax1.set_xscale('log')
#     ax1.set_yscale('log')
#     if y_symbol == y_symbols[0]:
#         ax1.set_ylabel(r"$(Z_{p,\text{mod}}- Z_p)/Z_*$ " f"{y_unit}")
#         ax1.set_ylim(1e-1, 1e1)
#         ax1.set_yticks([1e-1, 1e0, 1e1])
#         ax1.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
#     elif y_symbol == y_symbols[1]:
#         ax1.set_ylabel(r"$M_{Z,\text{mod}} - M_Z$ " f"{y_unit}")
#         ax1.set_ylim(1e0, 1e2)
#         ax1.set_yticks([1e0, 1e1, 1e2])
#         ax1.set_yticklabels([r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'])
#     ax1.set_xlim(left=1e-1)
#     # ax1.set_ylabel(r"$\Delta$" fr"{y_symbol}")
#     ax1.set_xlabel(fr"Mass $M_p$ [$M_J$]")

# # Remove y-axis labels from right column for clarity
# for i in range(4):
#     axes[i, 1].set_ylabel("")

# # Optionally, hide x-tick labels for all but the bottom row
# for i in range(3):
#     for j in range(2):
#         plt.setp(axes[i, j].get_xticklabels(), visible=False)

# plt.tight_layout()
# plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import odr

# --- Dummy data and functions for demonstration ---
# Remove or replace with your actual data and functions
df_FGK = pd.DataFrame({'x': np.logspace(-1, 1, 10), 'x_err1': 0.1, 'x_err2': 0.1,
                       'y1': np.logspace(-1, 1, 10), 'y1_err': 0.1, 'y2': np.logspace(0, 2, 10), 'y2_err': 0.2,
                       'y1_mod': np.logspace(-1, 1, 10)*1.1, 'y1_mod_err': 0.15, 'y2_mod': np.logspace(0, 2, 10)*0.9, 'y2_mod_err': 0.25})
df_M = df_FGK.copy()  # For demo, use same data

y_obs_labels = ['y1', 'y2']
y_obs_err_labels = ['y1_err', 'y2_err']
y_mod_labels = ['y1_mod', 'y2_mod']
y_mod_err_labels = ['y1_mod_err', 'y2_mod_err']
y_names = ['Zp', 'Mz']
y_symbols = ['Z_p', 'M_Z']
y_units = ['[$Z_\odot$]', '[M$_\oplus$]']
x_obs_label = 'x'
planetsynth_color = 'tab:blue'
GASTLI_color = 'tab:orange'
planetsynth_label = 'MH25'
GASTLI_label = 'AKZM24'
figsize = (12, 14)
ratio = [3, 1]

def loglog_model(B, x):
    return B[0] + B[1]*x

# --- End dummy data ---

fig, axes = plt.subplots(
    4, 2, figsize=figsize,
    sharex='col', sharey='row',
    gridspec_kw={'hspace': 0.08, 'wspace': 0.1}
)

for col, (df, col_title) in enumerate(zip([df_FGK, df_M], ["FGK", "M"])):
    for i in range(len(y_obs_labels)):
        # Get labels for this iteration
        y_obs_label = y_obs_labels[i]
        y_obs_err_label = y_obs_err_labels[i]
        y_mod_label = y_mod_labels[i]
        y_mod_err_label = y_mod_err_labels[i]
        y_name = y_names[i]
        y_symbol = y_symbols[i]
        y_unit = y_units[i]

        # Data
        lin_x_obs = df[x_obs_label]
        lin_x_obs_err = pd.Series(
            [max(df.loc[index][x_obs_label+'_err1'], np.abs(df.loc[index][x_obs_label+'_err2'])) for index in df.index],
            index=df.index, name=x_obs_label+'_err')
        lin_y_obs = df[y_obs_label]
        lin_y_obs_err = df[y_obs_err_label]
        lin_y_mod = df[y_mod_label]
        lin_y_mod_err = df[y_mod_err_label]

        # ODR for observed
        log_x_obs = np.log10(lin_x_obs)
        log_x_obs_err = lin_x_obs_err / (lin_x_obs * np.log(10))
        log_y_obs = np.log10(lin_y_obs)
        log_y_obs_err = lin_y_obs_err / (lin_y_obs * np.log(10))

        loglog_data_obs = odr.RealData(log_x_obs, log_y_obs, sx=log_x_obs_err, sy=log_y_obs_err)
        odr_loglog_obs = odr.ODR(data=loglog_data_obs, model=odr.Model(loglog_model), beta0=[-0.5, 0.5])
        output_loglog_obs = odr_loglog_obs.run()
        b0_obs, b1_obs = output_loglog_obs.beta
        b0_obs_err, b1_obs_err = output_loglog_obs.sd_beta

        # ODR for model
        log_y_mod = np.log10(lin_y_mod)
        log_y_mod_err = lin_y_mod_err / (lin_y_mod * np.log(10))
        loglog_data_mod = odr.RealData(log_x_obs, log_y_mod, sx=log_x_obs_err, sy=log_y_mod_err)
        odr_loglog_mod = odr.ODR(data=loglog_data_mod, model=odr.Model(loglog_model), beta0=[-0.5, 0.5])
        output_loglog_mod = odr_loglog_mod.run()
        b0_mod, b1_mod = output_loglog_mod.beta
        b0_mod_err, b1_mod_err = output_loglog_mod.sd_beta

        # Fit lines
        x_fit = np.logspace(np.log10(min(lin_x_obs)), np.log10(max(lin_x_obs)), 200)
        y_fit_obs = (x_fit**b1_obs) * (10**b0_obs)
        y_fit_obs_lower = ((x_fit)**(b1_obs-b1_obs_err)) * (10**(b0_obs-b0_obs_err))
        y_fit_obs_upper = ((x_fit)**(b1_obs+b1_obs_err)) * (10**(b0_obs+b0_obs_err))
        y_fit_mod = (x_fit**b1_mod) * (10**b0_mod)
        y_fit_mod_lower = ((x_fit)**(b1_mod-b1_mod_err)) * (10**(b0_mod-b0_mod_err))
        y_fit_mod_upper = ((x_fit)**(b1_mod+b1_mod_err)) * (10**(b0_mod+b0_mod_err))

        # Main plot
        ax0 = axes[2*i, col]
        ax0.set_xscale('log')
        ax0.set_yscale('log')
        if col == 0:
            ax0.set_ylabel(f"{y_name} {y_symbol} {y_unit}".strip())
        if i == 0:
            ax0.set_title(col_title)
        ax0.errorbar(lin_x_obs, lin_y_obs, 
                     xerr=lin_x_obs_err, yerr=lin_y_obs_err,
                     fmt='s',
                     color=planetsynth_color,
                     ecolor=planetsynth_color,
                     markeredgecolor='black',
                     markeredgewidth=1,
                     capsize=3
                    )   
        ax0.plot(x_fit, y_fit_obs, planetsynth_color,
                        label=fr"$\propto M_p^{{{b1_obs:.2f}}}$ [{planetsynth_label}]")
        ax0.fill_between(x_fit, y_fit_obs_lower, y_fit_obs_upper, color=planetsynth_color, alpha=0.2)

        ax0.errorbar(lin_x_obs, lin_y_mod,
                     xerr=lin_x_obs_err, yerr=lin_y_mod_err,
                     fmt='s',
                     color=GASTLI_color,
                     ecolor=GASTLI_color,
                     markeredgecolor='black',
                     markeredgewidth=1,
                     capsize=3
                    )
        ax0.plot(x_fit, y_fit_mod, GASTLI_color,
                        label=fr"$\propto M_p^{{{b1_mod:.2f}}}$ [{GASTLI_label}]")
        ax0.fill_between(x_fit, y_fit_mod_lower, y_fit_mod_upper, color=GASTLI_color, alpha=0.2)
        ax0.legend(fontsize=10)

        # Difference plot
        ax1 = axes[2*i+1, col]
        ax1.scatter(lin_x_obs, lin_y_mod - lin_y_obs, color='k', s=30, label=f"[{GASTLI_label}] - [{planetsynth_label}]")
        ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        if col == 0:
            if y_symbol == y_symbols[0]:
                ax1.set_ylabel(r"$(Z_{p,\text{mod}}- Z_p)/Z_*$ " f"{y_unit}")
                ax1.set_ylim(1e-1, 1e1)
                ax1.set_yticks([1e-1, 1e0, 1e1])
                ax1.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
            elif y_symbol == y_symbols[1]:
                ax1.set_ylabel(r"$M_{Z,\text{mod}} - M_Z$ " f"{y_unit}")
                ax1.set_ylim(1e0, 1e2)
                ax1.set_yticks([1e0, 1e1, 1e2])
                ax1.set_yticklabels([r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'])
        if i == len(y_obs_labels)-1:
            ax1.set_xlabel(fr"Mass $M_p$ [$M_J$]")

# Remove y-axis labels from right column for clarity
for i in range(4):
    axes[i, 1].set_ylabel("")

# Hide x-tick labels for all but the bottom row
for i in range(3):
    for j in range(2):
        plt.setp(axes[i, j].get_xticklabels(), visible=False)

plt.tight_layout()
plt.show()
# %%
