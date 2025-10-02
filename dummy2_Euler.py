# %%
import gastli.Thermal_evolution as therm
import numpy as np
import os

# #%%
#earth to jupiter conversions
Mjup = 318
Rjup = 11.2

#solar metallicity
Z0 = 0.014

# # %%
#final input parameters for interior
# CMFs = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])[3:4]
# CMFs = np.array([0.349255]) #CMF from MCMC sampling for '14) HATS-48 A b'
# CMFs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# CMFs = np.array([0.01, 0.025, 0.05])
CMFs = np.array([0.025])

# Zenvs = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2])*Z0
# Zenvs = np.array([1e-2, 1e-1, 1e0, 1e1])*Z0
Zenvs = np.array([Z0])

mass = (np.concatenate([
    np.array([0.06, 0.1]),
    np.arange(0.15, 1, 0.05),
    np.arange(1, 2, 0.1),
    np.arange(2, 3, 0.2)
])*Mjup)
# mass

Teqpls = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000])
# Teqpls = np.array([400, 500, 600, 700, 800, 900, 1000])
# Teqpls = np.array([200, 300])


#internal temperatures
# Tint_array = np.array([50, 100, 200, 300, 400, 500])
Tint_array = np.array([40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500])

n_mrel = len(CMFs) * len(Zenvs) * len(mass) * len(Teqpls)
print(n_mrel)

# %%
# #(potential) bad parameters for interior
# CMFs = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])[:1]

# Z0 = 0.014
# # Zenvs = (np.array([1e-2, 1e-1, 1e0, 1e1, 1e2])*Z0)[:2]
# Zenvs = (np.array([0.01, 0.1, 1, 10])*Z0)[3:4]


# mass = (np.concatenate([
#     np.array([0.06, 0.1]),
#     np.arange(0.15, 1, 0.05),
#     np.arange(1, 2, 0.1),
#     np.arange(2, 3, 0.2)
# ])*Mjup)[:2]
# # mass

# Teqpls = np.array([400, 500, 600, 700, 800, 900, 1000])[:2]

# #internal temperatures
# Tint_array = np.array([50, 100, 200, 300, 400, 500])[:2]
# # Tint_array = np.array([400, 500])

# n_mrel = len(CMFs) * len(Zenvs) * len(mass) * len(Teqpls)
# print(n_mrel)

# # %%
# #(potential) bad parameters for interior
# CMFs = np.array([0.2, 0.3])

# Z0 = 0.014
# # Zenvs = (np.array([1e-2, 1e-1, 1e0, 1e1, 1e2])*Z0)[:2]
# Zenvs = (np.array([0.01, 0.1, 1, 10])*Z0)[2:3]


# mass = (np.concatenate([
#     np.array([0.06, 0.1]),
#     np.arange(0.15, 1, 0.05),
#     np.arange(1, 2, 0.1),
#     np.arange(2, 3, 0.2)
# ])*Mjup)[3:5]
# # print(mass)

# Teqpls = np.array([400, 500, 600, 700, 800, 900, 1000])[:1]

# #internal temperatures
# Tint_array = np.array([50, 100, 200, 300, 400, 500])[2:4]
# # Tint_array = np.array([400, 500])

# n_mrel = len(CMFs) * len(Zenvs) * len(mass) * len(Teqpls)
# print(n_mrel)


# %%

#Initialize the mask array with zeros
# mask = np.zeros((len(CMFs), len(Zenvs), len(mass), len(Teqpls)), dtype=int)
mask = np.ones((len(CMFs), len(Zenvs), len(mass), len(Teqpls), 5), dtype=float)
# mask

# %%
#loop for thermal evolution
counter = 0

for i_cmf, CMF in enumerate(CMFs):
    for i_Zenv, Zenv in enumerate(Zenvs):
        for i_mass, M_P in enumerate(mass):
                for i_teq, Teqpl in enumerate(Teqpls):
                
                    mask[i_cmf, i_Zenv, i_mass, i_teq, 0] = CMF
                    mask[i_cmf, i_Zenv, i_mass, i_teq, 1] = Zenv
                    mask[i_cmf, i_Zenv, i_mass, i_teq, 2] = M_P
                    mask[i_cmf, i_Zenv, i_mass, i_teq, 3] = Teqpl

                    counter = counter+1
                    mask_flag = False

                    #File with thermal evolution
                    # file_name = "Output/thermal_sequence_CMF_" + "{:4.2f}".format(CMF) + "_Zenv_" + "{:4.5f}".format(Zenv) + "_mass_" + "{:4.2f}".format(M_P) + "_Teq_" + "{:4.2f}".format(Teqpl) + ".dat"
                    # file_name = "Output2/thermal_sequence_CMF_" + "{:4.2f}".format(CMF) + "_Zenv_" + "{:4.1e}".format(Zenv) + "_mass_" + "{:4.2f}".format(M_P) + "_Teq_" + "{:4.2f}".format(Teqpl) + ".dat"
                    file_name = "Output2_alpha_307/thermal_sequence_CMF_" + "{:4.2f}".format(CMF) + "_Zenv_" + "{:4.1e}".format(Zenv) + "_mass_" + "{:4.2f}".format(M_P) + "_Teq_" + "{:4.2f}".format(Teqpl) + ".dat"


                    #Check that file does not exist to avoid overwrite
                    if os.path.isfile(file_name):
                        continue

                    #Print message to monitor progress
                    # print('---------------')
                    # print('CMF = ', CMF)
                    # print('Zenv = ', Zenv)
                    # print('Mass [Mearth] = ', M_P)
                    # print('Teq [K] = ', Teqpl)
                    # print('Sequence = ', counter, ' out of ', n_mrel)
                    # print('---------------')

                    print('---------------')
                    print(f"CMF = {CMF:.2f}")
                    print(f"Zenv = {Zenv:.1e}")
                    print(f"Mass = {M_P:.2f}")
                    print(f"Teq = {Teqpl:.2f}")
                    print(f"Sequence = {counter} out of {n_mrel}")
                    print('---------------')

                    #thermal class evolution object
                    if (CMF >= 0.1) and (CMF <= 0.9):
                        alpha = 0.315

                    elif (CMF == 0.01) or (CMF == 0.025) or (CMF == 0.05):
                        alpha = 0.307

                    my_therm_obj = therm.thermal_evolution(pow_law_formass=alpha)

                    try:
                        print("----------------------------------------------------")
                        print("interior_retrieval.py: Initializing thermal evolution object's main function")
                        my_therm_obj.main(M_P=M_P,
                                        x_core=CMF,
                                        Teq=Teqpl,
                                        Tint_array=Tint_array,
                                        CO=0.55,
                                        log_FeH=0.,
                                        Zenv=Zenv,
                                        FeH_flag=False)
                        print("interior_retrieval.py: Main function executed")
                        print("----------------------------------------------------")
                    
                        #Solve thermal evolution equation
                        print("----------------------------------------------------")
                        print("interior_retrieval.py: Solving thermal evolution equation")
                        my_therm_obj.solve_thermal_evol_eq()
                        print("interior_retrieval.py: Thermal evolution equation solved")
                        print("----------------------------------------------------")

                        # 42/0
                        
                        # # # If successful, set the success flag to 1
                        # # mask[i_cmf, i_Zenv, i_mass, i_teq, 4] = 1
                        # print(f"interior_retrieval.py: mask_flag = {mask_flag}")
                        # mask_flag = True
                        # print(f"interior_retrieval.py: mask_flag = {mask_flag}")
                        # # print("interior_retrieval.py: Success flag set to 1")
                    
                    except ValueError as e:
                        print(f"interior_retrieval.py: Error in sequence -> {e}")
                        # print("interior_retrieval.py: 1st line")
                        # raise ValueError('Error in sequence')
                        # print(type(my_therm_obj.f_S))
                        # print('Error in sequence')
                        # print("2nd line")
                        
                        # #Save sequence of interior models with NaNs
                        # data = np.zeros((len(my_therm_obj.f_S), 11))
                        # print("Data initialized with shape: ", data.shape)
                        # data[:,0] = np.nan #my_therm_obj.f_S
                        # data[:,1] = np.nan #my_therm_obj.s_mean_TE
                        # data[:,2] = np.nan #my_therm_obj.s_top_TE 
                        # data[:,3] = np.nan #my_therm_obj.Tint_array
                        # data[:,4] = 5 #my_therm_obj.Rtot_TE*Rjup
                        # data[:,5] = np.nan #my_therm_obj.Rbulk_TE*Rjup
                        # data[:,6] = np.nan #my_therm_obj.Tsurf_TE
                        # data[:,7] = np.nan #my_therm_obj.age_points
                        # data[:,8] = np.nan #my_therm_obj.Zenv_TE
                        # data[:,9] = np.nan #my_therm_obj.Mtot_TE
                        # # tmm = CMF + (1-CMF)*my_therm_obj.Zenv_TE
                        # data[:,10] = np.nan #tmm
                        # header0 = 'f_S s_mean_TE s_top_TE Tint_K Rtot_earth Rbulk_earth Tsurf_K Age_Gyrs Zenv Mtot_earth Zplanet'

                        # np.savetxt(file_name, data, header=header0, comments='', fmt='%1.4e')
                        # print("File saved")
                        # continue

                    if np.any(np.isnan(my_therm_obj.f_S)):
                        # If successful, set the success flag to 1
                        mask[i_cmf, i_Zenv, i_mass, i_teq, 4] = 0

                    #Save sequence of interior models
                    data = np.zeros((len(my_therm_obj.f_S), 11))
                    data[:,0] = my_therm_obj.f_S
                    data[:,1] = my_therm_obj.s_mean_TE
                    data[:,2] = my_therm_obj.s_top_TE
                    data[:,3] = my_therm_obj.Tint_array
                    data[:,4] = my_therm_obj.Rtot_TE*Rjup
                    data[:,5] = my_therm_obj.Rbulk_TE*Rjup
                    data[:,6] = my_therm_obj.Tsurf_TE
                    data[:,7] = my_therm_obj.age_points
                    data[:,8] = my_therm_obj.Zenv_TE
                    data[:,9] = my_therm_obj.Mtot_TE
                    tmm = CMF + (1-CMF)*my_therm_obj.Zenv_TE
                    data[:,10] = tmm
                    header0 = 'f_S s_mean_TE s_top_TE Tint_K Rtot_earth Rbulk_earth Tsurf_K Age_Gyrs Zenv Mtot_earth Zplanet'

                    np.savetxt(file_name, data, header=header0, comments='', fmt='%1.4e')


print(f"dummy2.py: successful for {CMF}")
#  %%
# my_therm_obj.age_points


# %%

# #mask analysis
# print(f"mask:\n {mask} \n")
# print('-----')

# print(f"mask[..., 4]: \n {mask[..., 4]} \n")
# print('-----')

# print(f"mask[..., 4] == 1: \n {mask[..., 4] == 1} \n")
# print('-----')
# # mask[..., 4] == 1

# #successful indices
# successful_indices = np.argwhere(mask[..., 4] == 1)
# print(f"successful indices: \n {successful_indices} \n")
# # successful_indices

# # Extract parameters for successful runs
# successful_parameters = []
# for idx in successful_indices:
#     # print(idx[0])
#     CMF = mask[idx[0], idx[1], idx[2], idx[3], 0]
#     # print(CMF)
#     Zenv = mask[idx[0], idx[1], idx[2], idx[3], 1]
#     # print(Zenv)
#     mass = mask[idx[0], idx[1], idx[2], idx[3], 2]
#     # print(mass)
#     Teqpl = mask[idx[0], idx[1], idx[2], idx[3], 3]
#     # print(Teqpl)
#     successful_parameters.append((CMF, Zenv, mass, Teqpl))
#     print(successful_parameters)

# print("Successful parameter combinations:")
# for params in successful_parameters:
#     print(f"CMF = {params[0]:.2f}, Zenv = {params[1]:.2e}, Mass = {params[2]:.2f}, Teq = {params[3]:.2f}")

# # # %%
# # # Save the mask array to a file
# # flattened_mask = mask.reshape(-1, mask.shape[-1])
# # # flattened_mask

# # #header for the file
# # header = 'CMF Zenv M_P [M_E] Teq [K] Success'

# # # Save the flattened mask as a table
# # np.savetxt("mask_table.txt", flattened_mask, header=header, comments='', fmt='%1.4e')

# # print("Mask saved as 'mask_table.txt'")

# # %%
# import pandas as pd

# flattened_mask = mask.reshape(-1, mask.shape[-1])

# # Convert the mask to a DataFrame
# columns = ["CMF", "Zenv", "M_P [M_E]", "Teq [K]", "Success"]
# df = pd.DataFrame(flattened_mask, columns=columns)

# # Save as a CSV file

# table_name = f"mask_table_CMF_{str({CMFs[0]})}.csv"
# df.to_csv(table_name, index=False)

# print(f"Mask saved as '{table_name}'")

# # np.save("mask.npy", mask)
# %%
