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

##
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
                    print(f"Mass [Mearth] = {M_P:.2f}")
                    print(f"Teq = {Teqpl:.2f}")
                    print(f"Sequence [K] = {counter} out of {n_mrel}")
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

                    
                    except ValueError as e:
                        print(f"interior_retrieval.py: Error in sequence -> {e}")
                    

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
