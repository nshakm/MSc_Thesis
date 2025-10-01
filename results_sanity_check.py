# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import h5py
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

#TODO: add filter for FGK type stars

df = df_updated[
    (df_updated['pl_mass'] >= mass_min) &
    (df_updated['pl_mass'] <= mass_max) &
    (df_updated['pl_teq'] >= Teq_min) &
    (df_updated['pl_teq'] <= Teq_max) &
    (df_updated['st_age'] >= age_min) &
    (df_updated['st_age'].notna()) &
    (df_updated['pl_zbulk'].notna())
# ].drop(124)
].drop([124, 130])

# k = 1
# N = 5
# [index for index in df[(k)*int(len(df)/N):(k+1)*int(len(df)/N)].index]
#     # print(index)

# %%
x = 'pl_mass'
y = 'M_mod'
x_linspace = np.linspace(min(df[x]), max(df[x]), 100)

plt.xlabel(x)
plt.ylabel(y)

plt.plot(df[x], df[y], 'o', alpha=0.5)
plt.plot(x_linspace, x_linspace, 'r--', alpha=0.5)

# %%
x = 'pl_rad'
y = 'R_mod'
x_linspace = np.linspace(min(df[x]), max(df[x]), 100)

plt.xlabel(x)
plt.ylabel(y)

plt.plot(df[x], df[y], 'o', alpha=0.5)
plt.plot(x_linspace, x_linspace, 'r--', alpha=0.5)

# %%
x = 'st_age'
y = 'age_mod'
x_linspace = np.linspace(min(df[x]), max(df[x]), 100)

plt.xlabel(x)
plt.ylabel(y)

plt.plot(df[x], df[y], 'o', alpha=0.5)
plt.plot(x_linspace, x_linspace, 'r--', alpha=0.5)

# %%
x = 'pl_zbulk_rel'
y = 'Zp_rel'
x_linspace = np.linspace(min(df[x]), max(df[x]), 100)

plt.xlabel(x)
plt.ylabel(y)

plt.plot(df[x], df[y], 'o', alpha=0.5)
plt.plot(x_linspace, x_linspace, 'r--', alpha=0.5)

# # %%
# fig, axes = plt.subplots(1, 2, figsize=(12, 8))

# x1 = 'pl_mass'
# y1 = 'pl_zbulk_rel'
# axes[0].set_title("Relative bulk metallicity vs Mass (planetsynth)")
# axes[0].set_xlabel(x1)
# axes[0].set_ylabel(y1)

# axes[0].set_xscale("log")
# axes[0].set_yscale("log")

# axes[0].errorbar(df[x1], df[y1], fmt='o', alpha=0.7)


# x2 = 'M_mod'
# y2 = 'Zp_rel'
# axes[1].set_title("Relative bulk metallicity vs Mass (GASTLI)")
# axes[1].set_xlabel(x2)
# axes[1].set_ylabel(y2)

# axes[1].set_xscale("log")
# axes[1].set_yscale("log")

# axes[1].errorbar(df[x2], df[y2], fmt='o', alpha=0.7)
# %%
mass_labels = ['pl_mass', 'M_mod']
mass_err_labels = ['pl_mass_err1', 'pl_mass_err2', 'M_mod_err1', 'M_mod_err2']
mass_unit = '[Mjup]'

Zp_labels = ['pl_zbulk_rel', 'Zp_rel']
Zp_err_labels = ['pl_zbulk_rel_err', 'Zp_rel_err']
# Zp_units = ['']

Mbulk_labels = ['pl_mbulk', 'Mz_mod_obs']
Mbulk_err_labels = ['pl_mbulk_err', 'Mz_mod_obs_err']
# Mbulk_units = ['[Mearth]']


for i, (y_labels, y_err_labels, y_unit) in enumerate(zip([Mbulk_labels, Zp_labels,], [Mbulk_err_labels, Zp_err_labels], ['[Mearth]',''])):
    
    # print(y_units)

    x_obs = mass_labels[0]
    x_mod = mass_labels[1]
    # print(x_obs, x_mod)

    x_obs_err1, x_obs_err2 = mass_err_labels[:2]
    x_mod_err1, x_mod_err2 = mass_err_labels[2:]
    # print(x_obs_err1, x_mod_err1)

    y_obs = y_labels[0]
    y_mod = y_labels[1]
    # print(y_obs, y_mod)

    y_obs_err = y_err_labels[0]
    y_mod_err = y_err_labels[1]
    # print(y_obs_err, y_mod_err)


    #Combined plot
    plt.figure()
    # plt.title(f"nsteps = {nsteps}")

    plt.errorbar(df[x_obs],
                df[y_obs],
                # xerr=[df[x_obs_err1], np.abs(df[x_obs_err2])],
                xerr=[np.abs(df[x_obs_err2]), np.abs(df[x_obs_err1])],
                yerr=df[y_obs_err],
                fmt='o',
                alpha=0.7, 
                label='planetsynth')

    plt.errorbar(df[x_obs],
                df[y_mod],
                # xerr=[df[x_obs_err1], np.abs(df[x_obs_err2])],
                xerr=[np.abs(df[x_obs_err2]), np.abs(df[x_obs_err1])],
                yerr=df[y_mod_err],
                fmt='o',
                alpha=0.7, 
                label='GASTLI')

    # plt.errorbar(df['pl_mass'],
    #             np.abs(df['Zp_rel']-df['pl_zbulk_rel']),
    #             # xerr=[df['M_mod_err1'], np.abs(df['M_mod_err2'])],
    #             # yerr=df['Zp_rel_err'],
    #             fmt='o',
    #             alpha=0.7, 
    #             label=r'|GASTLI-planetsynth|')

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(f"{x_obs} {mass_unit}")
    plt.ylabel(f"{y_obs} {y_unit}")
    # if i == 0:
    #     plt.ylabel(f"{y_obs}")
    # elif i == 1:
    #     plt.ylabel(f"{y_obs} {Mbulk_units[i]}")

    # x_min = 1e-1
    # x_max = 1e1
    # y_min = min(df['pl_zbulk_rel'].min(), df['Zp_rel'].min())
    # y_max = max(df['pl_zbulk_rel'].max(), df['Zp_rel'].max())

    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)

    plt.legend()







    # for x_label, x_err_label in zip(mass_labels, mass_err_labels):
    #     df[y_labels[0]] = df[y_labels[0]].astype(float)
    #     df[y_labels[1]] = df[y_labels[1]].astype(float)
    #     df[x_label] = df[x_label].astype(float)
    #     df[x_err_label] = df[x_err_label].astype(float)

    #     # Ensure error columns are numeric
    #     df[y_err_labels[0]] = pd.to_numeric(df[y_err_labels[0]], errors='coerce')
    #     df[y_err_labels[1]] = pd.to_numeric(df[y_err_labels[1]], errors='coerce')

# %%
# %%

# %%
plt.figure()
plt.title(f"nsteps = {nsteps}")

plt.errorbar(df['pl_mass'],
             df['pl_zbulk_rel'],
            #  xerr=[df['pl_mass_err'], np.abs(df['pl_mass_err2'])],
             xerr=[np.abs(df['pl_mass_err2']), df['pl_mass_err1']],
             yerr=df['pl_zbulk_rel_err'],
             fmt='o',
             alpha=0.7, 
             label='planetsynth')

plt.errorbar(df['pl_mass'],
            df['Zp_rel'],
            # xerr=[df['M_mod_err1'], np.abs(df['M_mod_err2'])],
            xerr=[np.abs(df['M_mod_err2']), df['M_mod_err1']],
            yerr=df['Zp_rel_err'],
            fmt='o',
            alpha=0.7, 
            label='GASTLI')

# plt.errorbar(df['pl_mass'],
#             np.abs(df['Zp_rel']-df['pl_zbulk_rel']),
#             # xerr=[df['M_mod_err1'], np.abs(df['M_mod_err2'])],
#             # yerr=df['Zp_rel_err'],
#             fmt='o',
#             alpha=0.7, 
#             label=r'|GASTLI-planetsynth|')

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Mass [MJup]")
plt.ylabel("Relative bulk metallicity Zp/Z*")

x_min = min(df['pl_mass'].min(), df['M_mod'].min())
x_max = max(df['pl_mass'].max(), df['M_mod'].max())
y_min = min(df['pl_zbulk_rel'].min(), df['Zp_rel'].min())
y_max = max(df['pl_zbulk_rel'].max(), df['Zp_rel'].max())

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend()


fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=False)
plt.suptitle(f"nsteps = {nsteps}")

# Planetsynth subplot
axes[0].errorbar(
    df['pl_mass'],
    df['pl_zbulk_rel'],
    xerr=[df['pl_mass_err1'], np.abs(df['pl_mass_err2'])],
    yerr=df['pl_zbulk_rel_err'],
    fmt='o',
    alpha=0.7,
    # label='planetsynth'
    color='#1f77b4'
)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Mass [MJup]")
axes[0].set_ylabel("Relative bulk metallicity Zp/Z*")
axes[0].set_title("planetsynth")
# axes[0].legend()

# Annotate points with index for Planetsynth (with boxes)
for i, row in df.iterrows():
    axes[0].annotate(
        str(i),
        (row['pl_mass'], row['pl_zbulk_rel']),
        textcoords="offset points",
        xytext=(5, 0),
        ha='left',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.8)
    )

# GASTLI subplot
axes[1].errorbar(
    df['pl_mass'],
    df['Zp_rel'],
    # xerr=[df['M_mod_err1'], np.abs(df['M_mod_err2'])],
    xerr=[np.abs(df['M_mod_err2']), df['M_mod_err1']],
    yerr=df['Zp_rel_err'],
    fmt='o',
    alpha=0.7,
    color='#ff7f0e',
    # label='GASTLI'
)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Mass [MJup]")
axes[1].set_title("GASTLI")
# axes[1].legend()

# Annotate points with index for GASTLI (with boxes)
for i, row in df.iterrows():
    axes[1].annotate(
        str(i),
        (row['M_mod'], row['Zp_rel']),
        textcoords="offset points",
        xytext=(5, 0),
        ha='left',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.8)
    )

# Set same axis limits for both subplots
x_min = min(df['pl_mass'].min(), df['M_mod'].min())
x_max = max(df['pl_mass'].max(), df['M_mod'].max())
y_min = min(df['pl_zbulk_rel'].min(), df['Zp_rel'].min())
y_max = max(df['pl_zbulk_rel'].max(), df['Zp_rel'].max())

axes[0].set_xlim(x_min, x_max)
axes[1].set_xlim(x_min, x_max)
axes[0].set_ylim(y_min, y_max)
axes[1].set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()
# ...existing code...
# %%
# ...existing code...

import matplotlib.gridspec as gridspec

fig = plt.figure(constrained_layout=True, figsize=(16, 7))
gs = gridspec.GridSpec(2, 3, figure=fig)
# Make the first plot span two rows and two columns (large)
ax_main = fig.add_subplot(gs[:, :2])
# The other two plots on the right, one above the other
ax_top = fig.add_subplot(gs[0, 2])
ax_bottom = fig.add_subplot(gs[1, 2])

plt.suptitle(f"nsteps = {nsteps}")

# --- Main large plot: Both datasets together ---
ax_main.errorbar(
    df['pl_mass'],
    df['pl_zbulk_rel'],
    # xerr=[df['pl_mass_err1'], np.abs(df['pl_mass_err2'])],
    xerr=[np.abs(df['pl_mass_err2']), np.abs(df['pl_mass_err1'])],
    yerr=df['pl_zbulk_rel_err'],
    fmt='o',
    alpha=0.7,
    label='planetsynth',
    color='#1f77b4'
)
ax_main.errorbar(
    df['M_mod'],
    df['Zp_rel'],
    # xerr=[df['M_mod_err1'], np.abs(df['M_mod_err2'])],
    xerr=[np.abs(df['M_mod_err2']), np.abs(df['M_mod_err1'])],
    yerr=df['Zp_rel_err'],
    fmt='o',
    alpha=0.7,
    label='GASTLI',
    color='#ff7f0e'
)
ax_main.set_xscale("log")
ax_main.set_yscale("log")
ax_main.set_xlabel("Mass [MJup]")
ax_main.set_ylabel("Relative bulk metallicity Zp/Z*")
ax_main.set_title("All Data")
ax_main.legend()

# Annotate points with index for main plot (planetsynth)
for i, row in df.iterrows():
    ax_main.annotate(
        str(i),
        (row['pl_mass'], row['pl_zbulk_rel']),
        textcoords="offset points",
        xytext=(5, 0),
        ha='left',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.8)
    )

# --- Top right: Planetsynth only ---
ax_top.errorbar(
    df['pl_mass'],
    df['pl_zbulk_rel'],
    # xerr=[df['pl_mass_err1'], np.abs(df['pl_mass_err2'])],
    xerr=[np.abs(df['pl_mass_err2']), np.abs(df['pl_mass_err1'])],
    yerr=df['pl_zbulk_rel_err'],
    fmt='o',
    alpha=0.7,
    color='#1f77b4'
)
ax_top.set_xscale("log")
ax_top.set_yscale("log")
ax_top.set_xlabel("Mass [MJup]")
ax_top.set_ylabel("Zp/Z*")
ax_top.set_title("planetsynth")
for i, row in df.iterrows():
    ax_top.annotate(
        str(i),
        (row['pl_mass'], row['pl_zbulk_rel']),
        textcoords="offset points",
        xytext=(5, 0),
        ha='left',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.8)
    )

# --- Bottom right: GASTLI only ---
ax_bottom.errorbar(
    df['M_mod'],
    df['Zp_rel'],
    # xerr=[df['M_mod_err1'], np.abs(df['M_mod_err2'])],
    xerr=[np.abs(df['M_mod_err2']), np.abs(df['M_mod_err1'])],
    yerr=df['Zp_rel_err'],
    fmt='o',
    alpha=0.7,
    color='#ff7f0e'
)
ax_bottom.set_xscale("log")
ax_bottom.set_yscale("log")
ax_bottom.set_xlabel("Mass [MJup]")
ax_bottom.set_ylabel("Zp/Z*")
ax_bottom.set_title("GASTLI")
for i, row in df.iterrows():
    ax_bottom.annotate(
        str(i),
        (row['M_mod'], row['Zp_rel']),
        textcoords="offset points",
        xytext=(5, 0),
        ha='left',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.8)
    )

# Set same axis limits for all
x_min = min(df['pl_mass'].min(), df['M_mod'].min())
x_max = max(df['pl_mass'].max(), df['M_mod'].max())
y_min = min(df['pl_zbulk_rel'].min(), df['Zp_rel'].min())
y_max = max(df['pl_zbulk_rel'].max(), df['Zp_rel'].max())

for ax in [ax_main, ax_top, ax_bottom]:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()
# ...existing code...

# %%
