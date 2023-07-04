# -*- coding: utf-8 -*-

"""
Created on Sun Aug 29 16:09:47 2021

@author: Lukas
"""

import os.path
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
from matplotlib import rcParams

import refnx
from refnx.dataset import ReflectDataset, Data1D
from refnx.analysis import Transform, CurveFitter, Objective, Model, Parameter
from refnx.reflect import SLD, Slab, ReflectModel
from refnx.reflect import Linear, Tanh, Interface, Erf, Sinusoidal, Exponential
from refnx_transform_fresnel import Transform_Fresnel

save_file = ()


def fresnel(qc, qz, roughness=0):
    return (np.exp(-qz**2 * roughness**2) *
            abs((qz - np.sqrt((qz**2 - qc**2) + 0j)) /
            (qz + np.sqrt((qz**2 - qc**2)+0j)))**2)

settings = dict(file_path = #"./processed/reflectivity/",
                            "./processed/reflectivity/",
                file_name = [
                            #"rbbr_Reflectivity_rbbr_5mol_s0002_1_off",
                    
                            #"rbbr_Reflectivity_3mol_s0001_1_off",
                            #rbbr_Reflectivity_rbbr_3mol_alk_s0001_3_off"
                            
                            #"h2o_Reflectivity_h2o_s2_1_off",
                            #"rbbr_h2o_Reflectivity_h2o_s1_1_off_noff",
                            #"rbbr_h2o_Reflectivity_h2o_s1_2_on_noff",
                    
                            #"rbbr_Reflectivity_5mol_2_test_noflatffield",
                            #"rbbr_Reflectivity_5mol_3_test_noflatffield"
                            
                            "rbbr_Reflectivity_rbbr_1mol_s1_1_off",
                            "rbbr_Reflectivity_rbbr_1mol_s1_2_on",
                            
                            #"rbbr_Reflectivity_rbbr_3mol_s1_1_off",
                            
                            #"rbbr_Reflectivity_rbbr_5mol_s1_1_off",
                            #"rbbr_Reflectivity_rbbr_5mol_s1_1_off_noff",
                            #"rbbr_Reflectivity_rbbr_5mol_s1_2_on",
                    
                            #"rbbr_Reflectivity_rbbr_3mol_s0001_1_off" #
                            #"rbbr_Reflectivity_rbbr_3mol_s0001_2_off" #
                            #"rbbr_Reflectivity_rbbr_3mol_s0001_3_off" #
                            #"rbbr_Reflectivity_rbbr_3mol_s0001_4_on"  #
                            #"rbbr_Reflectivity_rbbr_3mol_s0001_5_on"  #
                            
                            #"rbbr_Reflectivity_rbbr_3mol_s0002_1_off" #
                            #"rbbr_Reflectivity_rbbr_3mol_s0002_2_off" #
                            
                            #"rbbr_Reflectivity_rbbr_3mol_s0003_1_off" # 
                            #"rbbr_Reflectivity_rbbr_3mol_s0003_2_on" #
                    
                            #"rbbr_Reflectivity_rbbr_3mol_acid_s0001_3_off" #
                            #"rbbr_Reflectivity_rbbr_3mol_acid_s0001_4_off" # Scan abborted
                            #"rbbr_Reflectivity_rbbr_3mol_acid_s0001_5_off" # Maybe a layer
                            
                            #"rbbr_Reflectivity_rbbr_3mol_alk_s0001_3_off" #
                            
                            #"rbbr_Reflectivity_rbbr_5mol_s0001_3_off" #
                            #"rbbr_Reflectivity_rbbr_5mol_s0002_1_off" #
                            
                            #"rbbr_Reflectivity_rbbr_1mol_s0002_1_off" #
                    
                    
                    
                            #"rbbr_Reflectivity_rbbr_old_beamtime",     
                            #"rbbr_Reflectivity_rbbr_3mol_s001_1_off",    
                            #"rbbr_Reflectivity_rbbr_3mol_s001_2_off",    
                            #"rbbr_Reflectivity_rbbr_3mol_s001_1_off",    
                            #"rbbr_Reflectivity_rbbr_3mol_s0_1_off",    
                            #"rbbr_Reflectivity_rbbr_3mol_s0_2_off",    
                            #"rbbr_Reflectivity_rbbr_3mol_s1_1_off",    
                            #"rbbr_Reflectivity_rbbr_3mol_s1_2_on",     
                            #"rbbr_Reflectivity_rbbr_3mol_s1_3_on",     
                            #"rbbr_Reflectivity_rbbr_3mol_s1_4_off",    
                            #"rbbr_Reflectivity_rbbr_3mol_ac_s1_1_off", # have a layer --> nachjustieren
                            #"rbbr_Reflectivity_rbbr_3mol_ac_s1_2_on",  # have a layer --> nachjustieren
                            #"rbbr_Reflectivity_rbbr_3mol_ac_s1_3_off", # have a layer --> nachjustieren
                            #"rbbr_Reflectivity_rbbr_3mol_ac_s1_4_off", # have a layer --> nachjustieren
                            #"rbbr_Reflectivity_rbbr_3mol_ac_s2_1_off", # have a layer --> nachjustieren
                            #"rbbr_Reflectivity_rbbr_3mol_ac_s2_2_off", # have a layer --> nachjustieren
                            #"rbbr_Reflectivity_rbbr_3mol_ac_s2_3_on",  # have a layer --> nachjustieren
                            #"rbbr_Reflectivity_rbbr_3mol_alk_s1_1_off",
                            #"rbbr_Reflectivity_rbbr_3mol_alk_s1_2_on", 
                            #"rbbr_Reflectivity_rbbr_3mol_alk_s1_3_off",
                            
                            #"rbbr_Reflectivity_rbbr_5mol_s0_1_off"
                            #"rbbr_Reflectivity_rbbr_5mol_s1_1_off",    
                            #"rbbr_Reflectivity_rbbr_5mol_s1_2_on",     
                            #"rbbr_Reflectivity_rbbr_5mol_s1_3_off",    
                            #"rbbr_Reflectivity_rbbr_1mol_s1_1_off",    
                            #"rbbr_Reflectivity_rbbr_1mol_s1_2_on",     
                            
                            #"srcl_Reflectivity_srcl_3mol_s1_1_off",
                            #"srcl_Reflectivity_srcl_3mol_s1_2_on",
                            
                            #"ercl_Reflectivity_ercl_1mol_s1_1_off",
                            #"ercl_Reflectivity_ercl_1mol_s1_2_on",
                            #"ercl_Reflectivity_ercl_1mol_s1_3_off",
                            #"ercl_Reflectivity_ercl_1mol_s1_4_off",
                            
                            #"rbbr_Reflectivity_srcl2_3mol_s1_1_off",   
                            #"rbbr_Reflectivity_srcl2_3mol_s1_2_on",    
                            #"rbbr_Reflectivity_srcl2_3mol_s1_3_off",   
                            #"rbbr_Reflectivity_ercl3_1mol_s1_1_off",   
                            #"rbbr_Reflectivity_ercl3_1mol_s1_2_on,     
                            #"rbbr_Reflectivity_ercl3_1mol_s1_3_off",   
                            #"rbbr_Reflectivity_ercl3_1mol_s1_4_off"    
                            #"rbbr_Reflectivity_rbbr_5mol_"             
                            ],
                save_figures = True,
                save_obj_struc = True,
                file_save = "./processed4/reflectivity_fits_batching/variant_082/",
                figure_save = "./processed4/reflectivity_fits_batching/variant_082/",
                # file_save = "./processed3/reflectivity_fits/3mol_acid_1_off/",
                # figure_save = "./processed3/reflectivity_fits/3mol_acid_1_off/",
                objective_struc_subname = None,
                q_c = 0.0271,    # rbbr 3 mol
                #q_c = 0.0271,    # rbbr 5 mol  
                #q_c = 0.023,    # rbbr 1 mol 
                #q_c = 0.0248,    # srcl 3 mol     
                #q_c = 0.0241,    # ercl 3 mol     
                            )



file = settings["file_path"] + settings["file_name"][0] + ".dat"
data_unpolished = ReflectDataset(file)
x_unpolished = data_unpolished.x
y_err_unpolished = data_unpolished.y_err
y_unpolished = data_unpolished.y

masky = np.logical_and(data_unpolished.x < 0.8, data_unpolished.x > 0.06)
data = ReflectDataset(file, mask = masky)

# 1 Mol RbBr: 10.234
# 3 Mol RbBr: 11.626
# 5 Mol RbBr: 12.782
# 3 Mol SrCl: 10.425

# ========================================================================================================================

#3 Mol fit no layer

air = SLD(0, name='air')
h2o_bulk_sld = SLD(15.1, name='h2o_bulk')

h2o_bulk = h2o_bulk_sld(0, 2.55)

structure = air | h2o_bulk

#structure[1].interfaces = Tanh()
#structure[1].interfaces = Erf()
#structure[1].interfaces = Sinusoidal()
#structure[1].interfaces = Exponential()

#h2o_layer.thick.setp(bounds=(15, 5000), vary=True)
h2o_bulk.rough.setp(bounds=(2.2, 3.8), vary=True)
h2o_bulk.sld.real.setp(bounds=(8.0, 15.5), vary=True)

model = ReflectModel(structure, bkg=2e-11, dq=5.0, scale=1)
model.scale.setp(bounds=(0.7, 1.85), vary=False)
model.bkg.setp(bounds=(1e-17, 9e-8), vary=False)


objective = Objective(model, data, transform=Transform_Fresnel('fresnel', use_weights = True, qc = settings["q_c"], roughness = 0))
print(objective.chisqr(), objective.logp(), objective.logl(), objective.logpost())
fitter = CurveFitter(objective)
fitter.fit('differential_evolution')

qc_fit = np.sqrt(16*np.pi*objective.parameters[1]["h2o_bulk"]["h2o_bulk - sld"].value*10**(-6))

# # ---------------------------------------------------------------

# # #3 Mol fit no layer

# air = SLD(0, name='air')
# h2o_bulk_sld = SLD(11.687, name='h2o_bulk')

# h2o_bulk = h2o_bulk_sld(0, 2.4)

# structure = air | h2o_bulk

# #structure[1].interfaces = Erf()

# #h2o_layer.thick.setp(bounds=(15, 5000), vary=True)
# h2o_bulk.rough.setp(bounds=(1, 6.5), vary=True)
# #h2o_bulk.sld.real.setp(bounds=(11, 16), vary=True)

# model = ReflectModel(structure, bkg=2e-11, dq=5.0)
# #model.scale.setp(bounds=(0.6, 1.5), vary=False)
# model.bkg.setp(bounds=(1e-11, 9e-9), vary=True)

# objective = Objective(model, data, transform=Transform_Fresnel('fresnel', qc = settings["q_c"], roughness = 0))
# print(objective.chisqr(), objective.logp(), objective.logl(), objective.logpost())

# fitter = CurveFitter(objective)
# fitter.fit('differential_evolution');
# qc_fit = np.sqrt(16*np.pi*objective.parameters[1]["h2o_bulk"]["h2o_bulk - sld"].value*10**(-6))

# ---------------------------------------------------------------

# #  3 Mol fit one layer
# air = SLD(0, name='air')
# h2o_enrich_sld = SLD(13, name='h2o_enrich')
# h2o_enrich = h2o_enrich_sld(3.5, 2.9)

# #h2o_bulk_sld = SLD(10.989, name='h2o_bulk')
# h2o_bulk_sld = SLD(10.425, name='h2o_bulk')
# #h2o_bulk_sld = SLD(14.103, name='h2o_bulk')
# h2o_bulk = h2o_bulk_sld(10000, 2.5)

# #h2o_layer.thick.setp(bounds=(15, 5000), vary=True)

# structure = air | h2o_enrich | h2o_bulk
# #structure = air | h2o_layer 

# structure[1].interfaces = Tanh()
# structure[2].interfaces = Tanh()
# #structure[2].interfaces = Linear()
# #structure[2].interfaces = Sinusoidal()

# h2o_enrich.rough.setp(bounds=(2.3,3.2), vary=True)
# h2o_enrich.sld.real.setp(bounds=(13,15), vary=True)
# #h2o_enrich.thick.setp(bounds=(2.5,20), vary=True)

# h2o_bulk.rough.setp(bounds=(2.3,3.2), vary=True)
# h2o_bulk.sld.real.setp(bounds=(9.2, 11.3), vary=False)

# model = ReflectModel(structure, bkg=1e-10, dq=5.0, scale=1.21)
# model.scale.setp(bounds=(1.1, 1.22), vary=False)
# model.bkg.setp(bounds=(1e-15, 9e-9), vary=True)

# objective = Objective(model, data, use_weights = True, transform=Transform_Fresnel('fresnel', qc = settings["q_c"], roughness = 0))
# print(objective.chisqr(), objective.logp(), objective.logl(), objective.logpost())

# fitter = CurveFitter(objective)
# fitter.fit('differential_evolution');
# qc_fit = np.sqrt(16*np.pi*objective.parameters[1]["h2o_bulk"]["h2o_bulk - sld"].value*10**(-6))

# # ========================================================================================================================

#3 Mol fit one layer
# air = SLD(0, name='air')
# h2o_2 = SLD(12.6602, name='h2o_2')
# h2o = SLD(12.782, name='h2o_bulk')

# h2o_layer = h2o(10000, 10)
# h2o_enrich = h2o_2(10, 2.4)

# #h2o_layer.thick.setp(bounds=(15, 5000), vary=True)

# structure = air | h2o_enrich | h2o_layer 

# #structure[1].interfaces = Tanh()
# #structure[2].interfaces = Tanh()

# h2o_layer.rough.setp(bounds=(2, 18.5), vary=True)
# #h2o_layer.sld.real.setp(bounds=(11, 13), vary=True)

# h2o_enrich.rough.setp(bounds=(1,3.5), vary=True)
# h2o_enrich.sld.real.setp(bounds=(12,35), vary=True)
# h2o_enrich.thick.setp(bounds=(1,25), vary=True)

# model = ReflectModel(structure, bkg=2e-11, dq=5.0, scale = 1)
# model.scale.setp(bounds=(0.6, 1.5), vary=True)
# model.bkg.setp(bounds=(1e-11, 9e-9), vary=True)

# objective = Objective(model, data, transform=Transform_Fresnel('fresnel', qc = settings["q_c"], roughness = 0))
# print(objective.chisqr(), objective.logp(), objective.logl(), objective.logpost())

# fitter = CurveFitter(objective)
# fitter.fit('differential_evolution');
# qc_fit = np.sqrt(16*np.pi*objective.parameters[1]["h2o_bulk"]["h2o_bulk - sld"].value*10**(-6))

# ========================================================================================================================

# #3 Mol fit two layers
# air = SLD(0, name='air')

# h2o_near_sld = SLD(13.2756, name='h2o_near')
# h2o_near = h2o_near_sld(50,2.4)

# h2o_enrich_sld = SLD(17.2345, name='h2o_enrich')
# h2o_enrich = h2o_enrich_sld(10, 5) 

# h2o_bulk_sld = SLD(12.782, name='h2o_bulk')
# h2o_bulk = h2o_bulk_sld(10000, 30)

# structure = air | h2o_near | h2o_enrich | h2o_bulk

# # structure[1].interfaces = Tanh()
# # structure[2].interfaces = Tanh()
# # structure[3].interfaces = Linear()

# h2o_near.rough.setp(bounds=(2, 3), vary=True)
# h2o_near.sld.real.setp(bounds=(10, 19), vary=True)
# h2o_near.thick.setp(bounds=(0.1,40), vary=True)

# h2o_enrich.rough.setp(bounds=(1,3.5), vary=True)
# h2o_enrich.sld.real.setp(bounds=(12,15), vary=True)
# h2o_enrich.thick.setp(bounds=(0.1,375), vary=True)

# h2o_bulk.rough.setp(bounds=(2, 3.5), vary=True)
# #h2o_bulk.sld.real.setp(bounds=(9, 15), vary=True)

# model = ReflectModel(structure, bkg=2e-11, dq=5.0)
# #model.scale.setp(bounds=(0.6, 1.5), vary=False)
# model.bkg.setp(bounds=(1e-11, 9e-9), vary=True)

# objective = Objective(model, data, transform=Transform_Fresnel('fresnel', qc = settings["q_c"], roughness = 0))
# print(objective.chisqr(), objective.logp(), objective.logl(), objective.logpost())

# fitter = CurveFitter(objective)
# fitter.fit('differential_evolution');
# qc_fit = np.sqrt(16*np.pi*objective.parameters[1]["h2o_bulk"]["h2o_bulk - sld"].value*10**(-6))

# ========================================================================================================================


# #  3 Mol fit one layer
# air = SLD(0, name='air')
# h2o_enrich_sld = SLD(13, name='h2o_enrich')
# h2o_enrich = h2o_enrich_sld(3.5, 2.9)

# #h2o_bulk_sld = SLD(10.989, name='h2o_bulk')
# h2o_bulk_sld = SLD(11.5094, name='h2o_bulk')
# #h2o_bulk_sld = SLD(14.103, name='h2o_bulk')
# h2o_bulk = h2o_bulk_sld(10000, 2.5)

# #h2o_layer.thick.setp(bounds=(15, 5000), vary=True)

# structure = air | h2o_enrich | h2o_bulk
# #structure = air | h2o_layer 

# #structure[1].interfaces = Tanh()
# #structure[2].interfaces = Tanh()
# #structure[2].interfaces = Linear()
# #structure[2].interfaces = Sinusoidal()

# h2o_enrich.rough.setp(bounds=(2.3,4.2), vary=True)
# h2o_enrich.sld.real.setp(bounds=(13,16), vary=True)
# h2o_enrich.thick.setp(bounds=(2.5,20), vary=True)

# h2o_bulk.rough.setp(bounds=(2.0,3.5), vary=True)
# h2o_bulk.sld.real.setp(bounds=(9.2, 14.3), vary=False)

# model = ReflectModel(structure, bkg=2e-11, dq=5.0, scale=1)
# model.scale.setp(bounds=(1.1, 1.22), vary=False)
# model.bkg.setp(bounds=(1e-15, 9e-9), vary=False)

# objective = Objective(model, data, use_weights = True, transform=Transform_Fresnel('fresnel', qc = settings["q_c"], roughness = 0))
# print(objective.chisqr(), objective.logp(), objective.logl(), objective.logpost())

# fitter = CurveFitter(objective)
# fitter.fit('differential_evolution');
# qc_fit = np.sqrt(16*np.pi*objective.parameters[1]["h2o_bulk"]["h2o_bulk - sld"].value*10**(-6))

# ========================================================================================================================


rcParams["figure.figsize"] = 12, 8
rcParams["font.size"] = 20

fig1 = plt.figure()
plt.plot(*structure.sld_profile())
plt.ylabel('SLD in $10^{-6} \AA^{-2}$')
plt.xlabel(r'z in $\AA$')
h2o_half = objective.parameters[1]["h2o_bulk"]["h2o_bulk - sld"].value/2
if len(objective.parameters[1]) == 4:
    h2o_half = objective.parameters[1][1][1].value/2
    sld_min = np.abs(structure.sld_profile()[1]-h2o_half)
    start = np.where(sld_min == sld_min.min())
    step_sld = np.heaviside(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]], 1)*h2o_half*2*np.heaviside(-(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]]) + objective.parameters[1][1][0].value, 1)
    h2o_half = objective.parameters[1][2][1].value/2
    step_sld += np.heaviside(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]] - objective.parameters[1][1][0].value, 1)*h2o_half*2
    
if len(objective.parameters[1]) == 3:
    h2o_half = objective.parameters[1][1][1].value/2
    sld_min = np.abs(structure.sld_profile()[1]-h2o_half)
    start = np.where(sld_min == sld_min.min())
    step_sld = np.heaviside(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]], 1)*h2o_half*2*np.heaviside(-(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]]) + objective.parameters[1][1][0].value, 1)
    h2o_half = objective.parameters[1][2][1].value/2
    step_sld += np.heaviside(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]] - objective.parameters[1][1][0].value, 1)*h2o_half*2
    plt.plot(structure.sld_profile()[0], step_sld,linestyle = "--")
elif len(objective.parameters[1]) == 2:
    h2o_half = objective.parameters[1]["h2o_bulk"]["h2o_bulk - sld"].value/2
    sld_min = np.abs(structure.sld_profile()[1]-h2o_half)
    start = np.where(sld_min == sld_min.min())
    step_sld = np.heaviside(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]], 1)*h2o_half*2
    sld_min = np.abs(structure.sld_profile()[1]-h2o_half)
    start = np.where(sld_min == sld_min.min())
    step_sld = np.heaviside(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]], 1)*h2o_half*2
    plt.plot(structure.sld_profile()[0], step_sld,linestyle = "--")
elif len(objective.parameters[1]) == 1:
    plt.plot(structure.sld_profile()[0], step_sld,linestyle = "--")

plt.show()
plt.close()
if settings["save_figures"] == True:
    if os.path.exists(settings["file_save"]) == False:
        os.mkdir(settings["file_save"])

if settings["save_figures"] == True:
    if settings["save_obj_struc"] == True:
        if settings["objective_struc_subname"] != None:
            fig1.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_sld" +  "_" + settings["objective_struc_subname"])
        else:
            n = 1
            while os.path.exists(settings["file_save"] + settings["file_name"][0] + "_fitted_sld" +  "_" + str(n) + ".png") == True:
                n = n + 1
            fig1.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_sld" +  "_" + str(n))

fig2 = plt.figure()
ax = fig2.gca()
ax.set_yscale("log")
x,y,y_err,model = (data.x, data.y, data.y_err, objective.model(data_unpolished.x))
ax.errorbar(x_unpolished, fresnel(qc_fit, x_unpolished, roughness = 0), c="#424242", ls="--")
ax.errorbar(x_unpolished,y_unpolished,y_err_unpolished,color="blue",marker="o",ms=3,lw=0,elinewidth=2)
ax.errorbar(x_unpolished, model, color="red")
ax.legend(["Fresnel", "Mes. points", "Fit"])
plt.xlabel(r'$q_z\;in\; \AA^{-1}$')
plt.ylabel(r'$R$')
plt.show()
plt.close()

if settings["save_figures"] == True:
    if settings["save_obj_struc"] == True:
        if settings["objective_struc_subname"] != None:
            fig2.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_reflectivtiy" +  "_" + settings["objective_struc_subname"])
        else:
            n = 1
            while os.path.exists(settings["file_save"] + settings["file_name"][0] + "_fitted_reflectivtiy" +  "_" + str(n) + ".png") == True:
                n = n + 1
            fig2.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_reflectivtiy" +  "_" + str(n))

fig3 = plt.figure()
ax = fig3.gca()
#ax.set_ylim([-0.2,1.5])
ax.set_yscale("log")
roughness_fit = objective.parameters[1][1][3].value
x,y,y_err,model = (data.x, data.y, data.y_err, objective.model(data_unpolished.x))
ax.errorbar(x_unpolished,y_unpolished/fresnel(qc_fit, x_unpolished, roughness = 0),y_err_unpolished/fresnel(qc_fit, x_unpolished, roughness = 0),color="blue",marker="o",ms=3,lw=0,elinewidth=2)
ax.errorbar(x_unpolished, model/fresnel(qc_fit, x_unpolished, roughness = 0), color="red")
ax.legend(["Mes. points", "Fit"])
plt.xlabel(r'$q_z\;in\; \AA^{-1}$')
plt.ylabel(r'$R/R_F$')
plt.ylim(-0.5,5)
plt.show()
plt.close()

if settings["save_figures"] == True:
    if settings["save_obj_struc"] == True:
        if settings["objective_struc_subname"] != None:
            fig3.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_r_rf" +  "_" + settings["objective_struc_subname"])
        else:
            n = 1
            while os.path.exists(settings["file_save"] + settings["file_name"][0] + "_fitted_r_rf" +  "_" + str(n) + ".png") == True:
                n = n + 1
            fig3.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_r_rf" +  "_" + str(n))


fig4 = plt.figure()
ax = fig4.gca()
ax.set_xlim([0,0.05])
ax.set_yscale("log")
x,y,y_err,model = (data.x, data.y, data.y_err, objective.model(data_unpolished.x))
ax.errorbar(x_unpolished,y_unpolished/fresnel(qc_fit, x_unpolished, roughness = 0),y_err_unpolished/fresnel(qc_fit, x_unpolished, roughness = 0),color="blue",marker="o",ms=3,lw=0,elinewidth=2)
ax.errorbar(x_unpolished, model/fresnel(qc_fit, x_unpolished, roughness = 0), color="red")
ax.legend(["Mes. points", "Fit"])
plt.xlabel(r'$q_z\;in\; \AA^{-1}$')
plt.ylabel(r'$R/R_F$')
plt.show()
plt.close()
if settings["save_figures"] == True:
    if settings["save_obj_struc"] == True:
        if settings["objective_struc_subname"] != None:
            fig4.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_lowqz_reflectivtiy" +  "_" + settings["objective_struc_subname"])
        else:
            n = 1
            while os.path.exists(settings["file_save"] + settings["file_name"][0] + "_fitted_lowqz_reflectivtiy" +  "_" + str(n) + ".png") == True:
                n = n + 1
            fig4.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_lowqz_reflectivtiy" +  "_" + str(n))

if settings["save_obj_struc"] == True:
    if settings["objective_struc_subname"] != None:
        objetive_file = settings["file_save"] + settings["file_name"][0] + "_" + settings["objective_struc_subname"] + "_object" + ".txt"
        structure_file = settings["file_save"] + settings["file_name"][0] + "_" + settings["objective_struc_subname"] + "_structure" + ".txt"
        data_file = settings["file_save"] + settings["file_name"][0] + "_" + settings["objective_struc_subname"] + "_data" + ".txt"
    else:
        n = 1
        while os.path.exists(settings["file_save"] + settings["file_name"][0] + "_" + str(n) + "_object" + ".txt") == True:
            n = n + 1
        objetive_file = settings["file_save"] + settings["file_name"][0] + "_" + str(n) + "_object" + ".txt"
        structure_file = settings["file_save"] + settings["file_name"][0] + "_" + str(n) + "_structure" + ".txt"
        data_file = settings["file_save"] + settings["file_name"][0] + "_" + str(n) + "_data" + ".txt"
        
    filehandler = open(objetive_file, 'wb')
    pickle.dump(objective, filehandler)
    filehandler.close()
    
    filehandler = open(structure_file, 'wb')
    pickle.dump(structure, filehandler)
    filehandler.close()
    
    filehandler = open(data_file, 'wb')
    pickle.dump(data_unpolished, filehandler)
    filehandler.close()
    
    parameter_file = settings["file_save"] + settings["file_name"][0] + "_parameter" + str(n) + ".txt"
    f = open(parameter_file,'w')
    for i in range(len(objective.parameters[0])):
        f.write(str(objective.parameters[0][i]) + "\n")
    for i in range(len(objective.parameters[1])):
        for j in range(len(objective.parameters[1][i])):
            f.write(str(objective.parameters[1][i][j]) + "\n")
    f.write("Calculation of the critical angle based on SLD: " + str(qc_fit) + "\n")
    f.close()

print(objective)
print("Calculation of the critical angle based on SLD: " + str(qc_fit))