from simulations.parameter import *


with_isotope = True
include_clouds = False
#Define parameters
FeH = Parameter('FEH', uniform_prior(-1.5, 1.5))
CO = Parameter('C_O', uniform_prior(0.1, 1.6))
#log_g = Parameter('log_g', gaussian_prior(log_g_mu, log_g_sigma))
log_g = Parameter('log_g', uniform_prior(2.5, 5.5))
T_int = Parameter('T_int', uniform_prior(300, 3500))

# Change parameter definition to temperature ratios!!
T1 = Parameter('T1', uniform_prior(300, 3500))
T2 = Parameter('T2', uniform_prior(300, 3500))
T3 = Parameter('T3', uniform_prior(300, 3500))
#T1 = Parameter('T1', lambda x : (x*0.5 + 0.5)*T_int.value)
#T2 = Parameter('T2', lambda x : (x*0.5 + 0.5)*T1.value)
#T3 = Parameter('T3', lambda x : (x*0.5 + 0.5)*T2.value)
alpha = Parameter('alpha', uniform_prior(1, 2))
log_delta = Parameter('log_delta', uniform_prior(3, 8))
#P_phot = Parameter('P_phot', uniform_prior(-3, 2))
log_Pquench = Parameter('log_Pquench', uniform_prior(-6, 3))

param_set = ParameterSet([FeH, CO, log_g, T_int, T1, T2, T3, alpha, log_delta, log_Pquench])

param_list = [FeH, CO, log_g, T_int, T1, T2, T3, alpha, log_delta, log_Pquench] 

if include_clouds:
    MgSiO3 = Parameter('log_MgSiO3', uniform_prior(-2.3, 1))
    Fe = Parameter('log_Fe', uniform_prior(-2.3, 1))
    fsed = Parameter('fsed', uniform_prior(0,10))
    Kzz = Parameter('log_Kzz', uniform_prior(5,13))
    sigma_lnorm= Parameter('sigma_lnorm', uniform_prior(1.05, 3))
    param_set.add_params([Fe, fsed, Kzz, sigma_lnorm])

    param_list += [Fe, fsed, Kzz, sigma_lnorm] 


iso_rat = Parameter('log_iso_rat', uniform_prior(-11, -1))
if with_isotope:
    param_set.add_params(iso_rat)

param_list +=  [iso_rat]


ndim = param_set.N_params
species = ['CO_main_iso', 'H2O_main_iso']
if with_isotope:
    species += ['CO_36']

## Non pRT params included here
radius = Parameter('radius', uniform_prior(0.8, 2.0))
rv = Parameter('rv', uniform_prior(10, 30))
limb_dark = Parameter('limb_dark', uniform_prior(0,1))
vsini = Parameter('vsini', uniform_prior(0, 50))
param_set_ext = ParameterSet([radius, rv, vsini, limb_dark])

param_list_ext = [radius, rv, vsini, limb_dark]

def deNormVal(values, param_list):
    values_actual = []
    for i, param in enumerate(param_list):
        param.prior(values[i])   #param.prior takes a value btw [0,1] and converts it to the actual value which can then be accessed using param.value
        values_actual.append(param.value)
    return values_actual


# LABELS, LOWER, UPPER = zip(*[
# [                  r'$FeH$',  -1.5, 1.5],   # temp_node_9
# [                  r'$CO$',  0.1, 1.6],  # CO_mol_scale
# [                  r'$\log g$',   2.5, 5.5],          # log g
# [                  r'$Tint$',  300,   3500],   # temp_node_5
# [                  r'$T1$',  300,   3500],      # T_bottom
# [                  r'$T2$',  300,   3500],   # temp_node_1
# [                  r'$T3$',  300,   3500],   # temp_node_2
# [                  r'$alpha$',  1.0, 2.0],   # temp_node_4
# [                  r'$log_delta$', 3.0, 8.0],   # temp_node_3
# [                  r'$log_Pquench$', -6.0, 3.0],   # temp_node_6
# # [                  r'$log_Fe$',  -2.3, 1.0], # CH4_mol_scale
# # [                  r'$fsed$',  0.0, 10.0],   # temp_node_8
# # [                  r'$logKzz$',  5.0, 13.0], # H2O_mol_scale \_mol\_scale
# # [                  r'$sigmalnorm$',  1.05, 3.0], # C2O_mol_scale
# [                  r'$log\_iso\_rat$',  -11.0, -1.0],   # temp_node_7
# [                  r'$R\_P$', 0.8, 2.0],             # R_P / R_Jupyter
# [                  r'$rv$',  10.0, 30.0], # NH3_mol_scale 20, 35
# [                  r'$vsini$', 0.0, 50 ], # H2S_mol_scale 10.0, 30.0
# [                  r'$limb\_dark$',  0.0, 1.0], # PH3_mol_scale
# [                  r'$b$',  1, 20.0], # PH3_mol_scale

# ])

LABELS, LOWER, UPPER = zip(*[
[                  r'$\left[{\rm Fe/H}\right]$', -1.5,   1.5],  # [Fe/H]
[                  r'${\rm C/O}$',  0.1,   1.6],  # C/O
[                  r'$\log \, g$',   2.5, 6.5],          # log g
[                  r'$T_{\rm int}$',  300,   5500],   # temp_node_5
[                  r'$T_1$',  300,   3500],      # T_bottom
[                  r'$T_2$',  300,   3500],   # temp_node_1
[                  r'$T_3$',  300,   3500],   # temp_node_2
[                  r'$\alpha$',   1.,    2.],  # alpha
[                  r'$\log \delta$',   3.0,    8.0],  # ∝ log delta / alpha
[                  r'$\log P_{\rm q}$',  -6.,    3.],  # log P_quench
[                  r'$log iso rat$',  -11.0, -1.0],   # temp_node_7
[                  r'$R_{\rm P}$', 0.6, 2.0],             # R_P / R_Jupyter
[                  r'$rv$',  10.0, 30.0], # NH3_mol_scale 20, 35
[                  r'$vsini$', 0.0, 50 ], # H2S_mol_scale 10.0, 30.0
[                  r'$limb dark$',  0.0, 1.0], # PH3_mol_scale
[                  r'$b$',  1, 20.0], # PH3_mol_scale

])