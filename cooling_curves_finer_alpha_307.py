# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random

import pandas as pd
import h5py

from scipy.interpolate import RegularGridInterpolator
import emcee

# %%
#constants 
Z0 = 0.014
Mjup = 318 #M_E
Rjup = 11.2 #R_E
sigma_b = 5.67e-5 # erg/s/cm2/K4
conv_AU_to_RS = 215.032 #AU to RS

# %%
#finer grid
filename = "my_forward_model_grid_finer_alpha_307.hdf5"
file = h5py.File(filename, "r")

# filename = "my_forward_model_grid_finer.hdf5"
# file = h5py.File(filename, "r")

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

#arrays
CMFs = file["CMF"][()]
# Zenvs = file["Zenv"][()]
masses = file["mass"][()]
Teqpls = file["Teqpl"][()]
k = 0
Tint_array = file["Tint"][()][k:]

#datasets
#splicing such that we only use the solar zenv and ignore first Tint (corresponds to too old)
data_set_Rtot = file["Rtot"][()][:, :, :, k:]
data_set_Mtot = file["Mtot"][()][:, :, :, k:]
data_set_age = file["age"][()][:, :, :, k:]
data_set_Rbulk = file["Rbulk"][()][:, :, :, k:]
data_set_Tsurf = file["Tsurf"][()][:, :, :, k:]
data_set_Zplanet = file["Zplanet"][()][:, :, :, k:]

#interpolated functions
method='pchip'
Rtot_interp = RegularGridInterpolator((CMFs, masses, Teqpls, Tint_array), data_set_Rtot, bounds_error=False, fill_value=None, method=method)
Mtot_interp = RegularGridInterpolator((CMFs, masses, Teqpls, Tint_array), data_set_Mtot, bounds_error=False, fill_value=None, method=method)
age_interp = RegularGridInterpolator((CMFs, masses, Teqpls, Tint_array), data_set_age, bounds_error=False, fill_value=None, method=method)
Rbulk_interp = RegularGridInterpolator((CMFs, masses, Teqpls, Tint_array), data_set_Rbulk, bounds_error=False, fill_value=None, method=method)
Tsurf_interp = RegularGridInterpolator((CMFs, masses, Teqpls, Tint_array), data_set_Tsurf, bounds_error=False, fill_value=None, method=method)
Zplanet_interp = RegularGridInterpolator((CMFs, masses, Teqpls, Tint_array), data_set_Zplanet, bounds_error=False, fill_value=None, method=method)

# %%
#dataframe
df_original = pd.read_csv("giant_planets_dataset_v2.csv")
df_original

# %%
"""
Function to get planet parameters from dataframe
"""
def planet_parameters(dataframe, index):

    planet = dataframe.iloc[index]

    #planet's name
    name = planet['pl_name']

    #planet's observed radius (normal distribution)
    rad = planet['pl_rad'] #Rjup
    rad_err1 = planet['pl_rad_err1'] #Rjup
    rad_err2 = np.abs(planet['pl_rad_err2']) #Rjup

    #planet's observed mass (normal distribution)
    #TODO
    #if mass outside mass grid bounds, skip and flag "mass out of bounds"
    mass = planet['pl_mass'] #Mjup
    mass_err1 = planet['pl_mass_err1'] #Mjup
    mass_err2 = np.abs(planet['pl_mass_err2']) #Mjup

    #star's observed age (uniform distribution)
    #TODO
    #if age not available, skip and flag "age not available"
    #eliif age less than 1 Gyr, skip and flag "age too young"
    age = planet['st_age'] #Gyr

    #TODO
    #if upper and lower bounds not available, set them to 0
    age_upper  = planet['st_age_upper'] #Gyr
    age_lower = planet['st_age_lower'] #Gyr

    #planet's observed equilibrium temperature (fixed value)
    # Teq = (planet['pl_instell']/(4*sigma_b))**(0.25) #K
    # dataframe.at[index, 'pl_teq'] = Teq
    Teq = planet['pl_teq'] #K

    #planet's bulk metallicity (based on planetsynth)
    Zp = planet['pl_zbulk'] #Zsun
    Zp_err = planet['pl_zbulk_err'] #Zsun


    return name, rad, rad_err1, rad_err2, mass, mass_err1, mass_err2, age, age_upper, age_lower, Teq, Zp, Zp_err

# %%
#data of planet 14 from MCMC method
data = pd.read_csv('Output3/samples/planet_85_samples.dat', sep='\s+')

CMF_plot = np.quantile(data['CMF'], 0.5)
# rad_plot = Rtot_interp((CMF_plot, mass*Mjup, Teq, Tint_array))
# age_plot = age_interp((CMF_plot, mass*Mjup, Teq, Tint_array))

# Tint_plot = np.linspace(min(Tint_array), max(Tint_array), 100)[::-1][:65]
planetsynth_color = '#1f77b4'
GASTLI_color = '#ff7f0e'

planetsynth_label = 'MH25'
GASTLI_label = 'AKZM24'

cmap = plt.get_cmap('rainbow', len(CMFs))
Tint_plot = np.linspace(min(Tint_array), max(Tint_array), 100)
# print(Tint_plot[-1])
max_Teq = max(Teqpls) #K
min_age = 1 #Gyr

# df = df_original[df_original.index == 0]

df = df_original[
    (df_original['pl_mass'] >= mass_min) &
    (df_original['pl_mass'] <= mass_max) &
    (df_original['pl_teq'] >= Teq_min) &
    (df_original['pl_teq'] <= Teq_max) &
    (df_original['st_age'] >= age_min) &
    (df_original['st_age'].notna())
]

# num_planets = 12 # Set the number of random planets you want
# random_indices = np.random.choice(df.index, size=num_planets, replace=False)
# df = df[:5]
df = df.loc[[10, 11, 24, 32, 36]]

# %%
x_attr = "Age[Gyr]"
y_attr = "Rtot[R_J]"
# y_attr = "M[M_J]"
# y_attr = "Zplanet"

cmap = plt.get_cmap('rainbow', len(CMFs))

if len(df) == 1:
    fig, ax = plt.subplots()
    for idx, index in enumerate(df.index):
        planet = df.loc[index]
        name, rad, rad_err1, rad_err2, mass, mass_err1, mass_err2, age, age_upper, age_lower, Teq, Zp, Zp_err = planet_parameters(df_original, index)

        ax.set_title(fr"{index}) {name}" "\n" fr"{mass:.2f} $M_J$ | {age:.2f} Gyr | {Teq:.2f} K")
        ax.set_xlabel(x_attr)
        ax.set_ylabel(y_attr)

        ax.set_xlim(0, 15)
        ax.set_xticks([0, 3, 6, 9, 12, 15])

        
        for i, CMF in enumerate(CMFs):
            ax.plot(age_interp((CMF, mass*Mjup, Teq, Tint_plot)),
                    Rtot_interp((CMF, mass*Mjup, Teq, Tint_plot)),
                    '-', linewidth=3,
                    #  alpha=0.1,
                    color=cmap(i/len(CMFs)),
                    label=f'CMF = {CMF:.2f}'
                    )

        #true observed value
        ax.errorbar(
            age, rad,
            xerr=[[age-age_lower], [age_upper-age]],
            yerr=[[rad_err2], [rad_err1]],
            fmt="s",
            # label="Model value",
            # alpha=0.7,
            color=planetsynth_color,
            markeredgecolor='black',      # Add this
            markeredgewidth=1.5,          # And this
            label=planetsynth_label,
            capsize=5,
            elinewidth=2
        )

            # ax.plot(age_interp((CMF, mass*Mjup, Teq, Tint_array)),
            #         Rtot_interp((CMF, mass*Mjup, Teq, Tint_array)),
            #         'o-', 
            #         #  alpha=0.1,
            #         color=cmap(i/len(CMFs_new)),
            #         label=f'CMF = {CMF:.2f}'
            #         )

        # ax.plot(age_interp_035((mass*Mjup, Teq, Tint_array_035)),
        #         Rtot_interp_035((mass*Mjup, Teq, Tint_array_035)),
        #         'o-', 
        #         #  alpha=0.1,
        #         color='grey',
        #         label=f'CMF = 0.35'
        #         )


        ax.legend(fontsize='6', loc='upper left', ncol=3)


else:
    # Define the number of rows and columns for subplots
    num_planets = len(df)
    cols = 3  # Number of columns you want (adjust as needed)
    rows = (num_planets + cols) // cols  # Calculate rows dynamically


    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), 
                            #  sharex='col', sharey='row'
                            )  # Adjust figsize as needed
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    # Loop through the planets and plot on each subplot
    for idx, index in enumerate(df.index):
        # print(index)
        planet = df.loc[index]
        name, rad, rad_err1, rad_err2, mass, mass_err1, mass_err2, age, age_upper, age_lower, Teq, Zp, Zp_err = planet_parameters(df_original, index)

        # print(f"----------\n{index}) {name} \n----------")
        # print(f"M = {mass:.2f} (+{mass_err1:.2f}, -{mass_err2:.2f}) Mjup")
        # print(f"R = {rad:.2f} (+{rad_err1:.2f}, -{rad_err2:.2f}) Rjup")
        # print(f"Age = {age:.2f} [{age_lower:.2f}, {age_upper:.2f}] Gyr")
        # print(f"Teq = {Teq:.2f} K \n")

        ax = axes[idx+1]  # Select the current subplot
        ax.set_title(fr"{index}) {name}" "\n" fr"{rad:.2f} $R_J$ | {mass:.2f} $M_J$ | {age:.2f} Gyr | {Teq:.2f} K", fontsize=14)
        # ax.set_xlabel(x_attr)
        # ax.set_ylabel(y_attr)
        # ax.set_xlabel(r"Age $\tau_p$ [Gyr]", fontsize=13)
        # ax.set_ylabel(r"Radius $R_p$ [$R_J$]", fontsize=13)
        ax.set_xlabel(r"Age $\tau_{\text{interp}}$ [Gyr]", fontsize=13)
        ax.set_ylabel(r"Radius $R_{\text{interp}}$ [$R_J$]", fontsize=13)


        ax.set_xlim(0, 15)
        ax.set_xticks([0, 3, 6, 9, 12, 15])

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # ax.set_ylim(0.9, 1.15)

        for i, CMF in enumerate(CMFs):
            # if CMF == 0.01:
            #     continue
            ax.plot(age_interp((CMF, mass*Mjup, Teq, Tint_plot)),
                    Rtot_interp((CMF, mass*Mjup, Teq, Tint_plot)),
                    # Mtot_interp((CMF, mass*Mjup, Teq, Tint_plot)),
                    # Zplanet_interp((CMF, mass*Mjup, Teq, Tint_plot)),
                    '-', linewidth=3,
                    #  alpha=0.1,
                    color=cmap(i/len(CMFs)),
                    label=f'CMF = {CMF:.2f}'
                    )


        #true observed value
        ax.errorbar(
            age,
            rad,
            # mass,
            # Zp,   
            xerr=[[age-age_lower], [age_upper-age]],
            yerr=[[rad_err2], [rad_err1]],
            # yerr=[[mass_err2], [mass_err1]],
            # yerr=[[Zp_err], [Zp_err]],
            fmt="s",
            markersize=8,
            # label="Observed value",
            # alpha=0.7,
            color=planetsynth_color,
            markeredgecolor='black',      # Add this
            markeredgewidth=1.5,          # And this
            # label='MH25',
            capsize=5,
            elinewidth=2
        )


            # ax.plot(age_interp((CMF, mass*Mjup, Teq, Tint_plot)),
            #         Rtot_interp((CMF, mass*Mjup, Teq, Tint_plot)),
            #         'o-', 
            #         #  alpha=0.1,
            #         color=cmap(i/len(CMFs_new)),
            #         label=f'CMF = {CMF:.2f}'
            #         )
        
        # ax.plot(age_interp_035((mass*Mjup, Teq, Tint_array_035)),
        #         Rtot_interp_035((mass*Mjup, Teq, Tint_array_035)),
        #         'o-', 
        #         #  alpha=0.1,
        #         color='grey',
        #         label=f'CMF = 0.35'
        #         )


        # ax.legend(fontsize='6', loc='upper left', ncol=3)

    legend_elements = [
    # Line2D([0], [0], color='black', linestyle='--', linewidth=5, label='Best-fit CMF'),
    # Line2D([0], [0], color='black', alpha=0.2, linewidth=10, label='CMF error range'),
    Line2D([0], [0], color=planetsynth_color, marker='s', linestyle='', markeredgecolor='black', markeredgewidth=1.5, markersize=8, label='MH25'),
    # Line2D([0], [0], color=GASTLI_color, marker='s', linestyle='', markeredgecolor='black', markeredgewidth=1.5, markersize=8, label='AKZM24'),
    ]

    # Add all CMF curves to the legend (optional, or just show a colorbar)
    for i, CMF in enumerate(CMFs):
        legend_elements.append(
            Line2D([0], [0], color=cmap(i/len(CMFs)), linewidth=3, label=f'CMF = {CMF:.2f}')
        )

    # # Turn off unused subplots
    # for ax in axes[num_planets:]:
    #     ax.axis('off')

    axes[0].legend(
        handles=legend_elements,
        loc='center',
        frameon=True,              # Show the box
        fontsize=14,
        framealpha=1,              # Opaque box
        edgecolor='black',         # Black border
        facecolor='white',         # White background
        fancybox=True,             # Rounded corners
        borderpad=1                # Padding inside the box
    )

    axes[0].axis('off')
    # Add a single legend for the entire figure
    # handles, labels = ax.get_legend_handles_labels()  # Get handles and labels from the last subplot
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize='10', ncol=3)

    # Adjust layout
    plt.tight_layout()  # Leave space for the legend at the top
    # plt.savefig("img/old_cooling_curves.pdf")
    # plt.savefig("img/cooling_curves.pdf")
    # plt.savefig("img/filtered_cooling_curves.pdf")
    # plt.savefig("img/filtered_cooling_curves_finer.pdf")
    # plt.savefig("img/filtered_cooling_curves_finer2.pdf")
    # plt.savefig("img/filtered_cooling_curves_finer2_without_CMF_001.pdf")
    # plt.savefig("img/filtered_age_vs_mass_finer2.pdf")
    # plt.savefig("img/filtered_age_vs_Zp_finer2.pdf")
    # plt.savefig("img/filtered_cooling_curves_finer_alpha_307_correct.pdf")
    plt.savefig("img/2_filtered_cooling_curves_without_report.pdf")

# %%
