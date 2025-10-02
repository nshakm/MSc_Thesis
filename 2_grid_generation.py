# %%
import numpy as np
import pandas as pd
import h5py

# %%
#constants
Mjup = 318.
Rjup = 11.2

#solar metallicity
Z0 = 0.014
# %%
#arrays of grid

#CMF
CMFs = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
n_CMF = len(CMFs)

# Zenvs
# Zenvs = np.array([1e-2, 1e-1, 1e0, 1e1])*Z0
# Zenvs = np.array([1e-2, 1e-1, 1e0])*Z0
Zenvs = np.array([Z0])
n_Zenv = len(Zenvs)

#mass
mass = (np.concatenate([
    np.array([0.06, 0.1]),
    np.arange(0.15, 1, 0.05),
    np.arange(1, 2, 0.1),
    np.arange(2, 3, 0.2)
])*Mjup)
n_mass = len(mass)

#equilibrium temperature
Teqpls = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000])
n_Teqpl = len(Teqpls)

#internal temperature
Tint_array = np.array([40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500])
n_Tint = len(Tint_array)

n_mrel = n_CMF * n_Zenv * n_mass * n_Teqpl
print(n_mrel)

# %%
#create file
grid_filename = "my_forward_model_grid_finer_alpha_307.hdf5"
f = h5py.File(grid_filename, "w")

#assign arrays for grid
f['CMF'] = CMFs
f['Zenv'] = Zenvs
f['mass'] = mass
f['Teqpl'] = Teqpls
f['Tint'] = Tint_array

#datasets
data_set_Rtot = f.create_dataset("Rtot", (n_CMF, n_mass, n_Teqpl, n_Tint), dtype='f')
data_set_Rbulk = f.create_dataset("Rbulk", (n_CMF, n_mass, n_Teqpl, n_Tint), dtype='f')
data_set_age = f.create_dataset("age", (n_CMF, n_mass, n_Teqpl, n_Tint), dtype='f')
data_set_Tsurf = f.create_dataset("Tsurf", (n_CMF, n_mass, n_Teqpl, n_Tint), dtype='f')
data_set_Mtot = f.create_dataset("Mtot", (n_CMF, n_mass, n_Teqpl, n_Tint), dtype='f')
data_set_Zplanet = f.create_dataset("Zplanet", (n_CMF, n_mass, n_Teqpl, n_Tint), dtype='f')
# data_set_Zenv = f.create_dataset("Zenv", (n_CMF, n_Zenv, n_mass, n_Teqpl, n_Tint), dtype='f')

# %%
#loop to read output files and fill in the datasets
for i_cmf, CMF in enumerate(CMFs):
    for i_Zenv, Zenv in enumerate(Zenvs):
        for i_mass, M_P in enumerate(mass):
                for i_teq, Teqpl in enumerate(Teqpls):
                     
                        file_name = "Output2_alpha_307/thermal_sequence_CMF_" + "{:4.2f}".format(CMF) + "_Zenv_" + "{:4.1e}".format(Zenv) + "_mass_" + "{:4.2f}".format(M_P) + "_Teq_" + "{:4.2f}".format(Teqpl) + ".dat"
                        # print(file_name)

                        #read file
                        data = pd.read_csv(file_name, sep='\s+', header=0)
                        rtot = data['Rtot_earth']
                        rbulk = data['Rbulk_earth']
                        age = data['Age_Gyrs']
                        tsurf = data['Tsurf_K']
                        mtot = data['Mtot_earth']
                        zplanet = data['Zplanet']
                        # zenv = data['Zenv']

                        #fill datasets
                        data_set_Rtot[i_cmf, i_mass, i_teq, :] = rtot/Rjup
                        data_set_Rbulk[i_cmf, i_mass, i_teq, :] = rbulk/Rjup
                        data_set_age[i_cmf, i_mass, i_teq, :] = age
                        data_set_Tsurf[i_cmf, i_mass, i_teq, :] = tsurf
                        data_set_Mtot[i_cmf, i_mass, i_teq, :] = mtot/Mjup
                        data_set_Zplanet[i_cmf, i_mass, i_teq, :] = zplanet
                        # data_set_Zenv[i_cmf, i_Zenv, i_mass, i_teq, :] = zenv


# %%
attrs = [attr for attr in list(f.keys()) \
         if ((attr != 'CMF') and (attr != 'Zenv') and (attr != 'mass') and (attr != 'Teqpl') and (attr != 'Tint'))]


for attr in attrs:
    f[attr].dims[0].attach_scale(f['CMF'])
    # f[attr].dims[1].attach_scale(f['Zenv'])
    f[attr].dims[1].attach_scale(f['mass'])
    f[attr].dims[2].attach_scale(f['Teqpl'])
    f[attr].dims[3].attach_scale(f['Tint'])

    f[attr].dims[0].label = 'CMF'
    # f[attr].dims[1].label = 'Zenv'
    f[attr].dims[1].label = 'mass'
    f[attr].dims[2].label = 'Teqpl'
    f[attr].dims[3].label = 'Tint'

# %%
f.close()
# %%
