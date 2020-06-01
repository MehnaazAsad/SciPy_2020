import matplotlib.gridspec as gridspec  
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import random

def std_func(num_arr):
    """
    Calculates standard deviation from 0

    Parameters
    ----------
    num_arr: np.array
        Array of numbers to calculate standard deviation of

    Returns
    ---------
    std: float
        Standard deviation of input array
    """
    if len(num_arr) == 0:
        return 0
    mean = 0
    diff_sqrd_arr = []
    for value in num_arr:
        diff = value - mean
        diff_sqrd = diff**2
        diff_sqrd_arr.append(diff_sqrd)
    mean_diff_sqrd = np.mean(diff_sqrd_arr)
    std = np.sqrt(mean_diff_sqrd)
    return std

def assign_colour_label_data(df):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """
    catl = df.copy()

    logmstar_arr = catl.logmstar.values
    u_r_arr = catl.modelu_rcorr.values

    colour_label_arr = np.empty(len(catl), dtype='str')
    for idx, value in enumerate(logmstar_arr):

        # Divisions taken from Moffett et al. 2015 equation 1
        if value <= 9.1:
            if u_r_arr[idx] > 1.457:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value > 9.1 and value < 10.1:
            divider = 0.24 * value - 0.7
            if u_r_arr[idx] > divider:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value >= 10.1:
            if u_r_arr[idx] > 1.7:
                colour_label = 'R'
            else:
                colour_label = 'B'
            
        colour_label_arr[idx] = colour_label
    
    catl['colour_label'] = colour_label_arr

    return catl

def get_sigma_error(eco_keys):
    """
    Bootstrap error in sigma using 20 samples and all galaxies in survey

    Parameters
    ----------
    eco_keys: np.array
        Unique group IDs of galaxies in survey

    Returns
    ---------
    std_red_cen_err: np.array
        Bootstrapped error result for groups with red centrals
    std_blue_cen_err: np.array
        Bootstrapped error result for groups with blue centrals
    """

    grp_key_arr = list(eco_keys)
    N_samples = 20
    std_blue_cen_arr_global = []
    std_red_cen_arr_global = []
    for i in range(N_samples):
        deltav_arr = []
        cen_stellar_mass_arr = []
        cen_colour_arr = []
        for j in range(len(eco_keys)):
            rng = random.choice(grp_key_arr)
            group = eco_groups.get_group(rng)
            cz_arr = group.cz.values
            grpcz = np.unique(group.grpcz.values)
            deltav = cz_arr-grpcz
            # Since some groups don't have a central
            if 1 in group.fc.values:
                cen_stellar_mass = group.logmstar.loc[group.fc.values == 1].values[0]
                cen_colour_label = group.colour_label.loc[group.fc.values == 1].values[0]
            for val in deltav:
                deltav_arr.append(val)
                cen_stellar_mass_arr.append(cen_stellar_mass)
                cen_colour_arr.append(cen_colour_label)
        data = {'deltav': deltav_arr, 'log_cen_stellar_mass': cen_stellar_mass_arr,\
            'cen_colour_label': cen_colour_arr}
        vel_col_mass_df = pd.DataFrame(data)
        df_blue_cen = vel_col_mass_df.loc[vel_col_mass_df['cen_colour_label'] == 'B']
        df_red_cen = vel_col_mass_df.loc[vel_col_mass_df['cen_colour_label'] == 'R']
        stellar_mass_bins = np.arange(7,12,0.5)
        i = 0
        std_blue_cen_arr = []
        for index1,bin_edge in enumerate(stellar_mass_bins):
            df_blue_cen_deltav_arr = []
            if index1 == 9:
                break
            for index2,stellar_mass in enumerate(df_blue_cen.log_cen_stellar_mass.values):
                if stellar_mass >= bin_edge and stellar_mass < stellar_mass_bins[index1+1]:
                    df_blue_cen_deltav_arr.append(df_blue_cen.deltav.values[index2])
            std = std_func(df_blue_cen_deltav_arr)
            std_blue_cen_arr.append(std)
        i = 0
        std_red_cen_arr = []
        for index1,bin_edge in enumerate(stellar_mass_bins):
            df_red_cen_deltav_arr = []
            if index1 == 9:
                break
            for index2,stellar_mass in enumerate(df_red_cen.log_cen_stellar_mass.values):
                if stellar_mass >= bin_edge and stellar_mass < stellar_mass_bins[index1+1]:
                    df_red_cen_deltav_arr.append(df_red_cen.deltav.values[index2])
            std = std_func(df_red_cen_deltav_arr)
            std_red_cen_arr.append(std)
        std_blue_cen_arr_global.append(std_blue_cen_arr)
        std_red_cen_arr_global.append(std_red_cen_arr)

    std_red_cen_arr_global = np.array(std_red_cen_arr_global)
    std_blue_cen_arr_global = np.array(std_blue_cen_arr_global)

    std_red_cen_err = np.std(std_red_cen_arr_global, axis=0)
    std_blue_cen_err = np.std(std_blue_cen_arr_global, axis=0)

    return std_red_cen_err, std_blue_cen_err

# Formatting for plots and animation
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}, size=20)
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)

# Reading in data and applying completeness and survey buffer region cuts
eco = pd.read_csv('eco_dr1.csv', delimiter=',', header=0)
eco_nobuff = eco.loc[(eco.grpcz.values >= 3000) & 
    (eco.grpcz.values <= 7000) & (eco.absrmag.values <= -17.33) &
    (eco.logmstar.values >= 8.9)]

# Assigning colour label to galaxies
eco_nobuff = assign_colour_label_data(eco_nobuff)

# Calculating velocity dispersion and combining with other galaxy details used
# for plotting later
eco_groups = eco_nobuff.groupby('grp')
eco_keys = eco_groups.groups.keys()
deltav_arr = []
cen_stellar_mass_arr = []
cen_colour_label_arr = []
cen_colour_arr = []
for index,key in enumerate(eco_keys):
    group = eco_groups.get_group(key)
    cz_arr = group.cz.values
    grpcz = np.unique(group.grpcz.values)
    deltav = cz_arr-grpcz
    # Since some groups don't have a central
    if 1 in group.fc.values:
        cen_stellar_mass = group.logmstar.loc[group.fc.values == 1].values[0]
        cen_colour_label = group.colour_label.loc[group.fc.values == 1].values[0]
        cen_colour = group.modelu_rcorr.loc[group.fc.values == 1].values[0]

    for val in deltav:
        deltav_arr.append(val)
        cen_stellar_mass_arr.append(cen_stellar_mass)
        cen_colour_label_arr.append(cen_colour_label)
        cen_colour_arr.append(cen_colour)

# Creating dataframe of useful properties
data = {'deltav': deltav_arr, 'log_cen_stellar_mass': cen_stellar_mass_arr,
        'cen_colour_label': cen_colour_label_arr, 
        'cen_ur_colour': cen_colour_arr}
vel_col_mass_df = pd.DataFrame(data)
df_blue_cen = vel_col_mass_df.loc[vel_col_mass_df['cen_colour_label'] == 'B']
df_red_cen = vel_col_mass_df.loc[vel_col_mass_df['cen_colour_label'] == 'R']

# Calculating std manually from mean set to 0 km/s
stellar_mass_bins = np.arange(7,12,0.5)
i = 0
std_blue_cen_arr = []
for index1,bin_edge in enumerate(stellar_mass_bins):
    df_blue_cen_deltav_arr = []
    if index1 == 9:
        break
    for index2,stellar_mass in enumerate(df_blue_cen.log_cen_stellar_mass.values):
        if stellar_mass >= bin_edge and stellar_mass < stellar_mass_bins[index1+1]:
            df_blue_cen_deltav_arr.append(df_blue_cen.deltav.values[index2])
    N = len(df_blue_cen_deltav_arr)
    std = std_func(df_blue_cen_deltav_arr)
    std_blue_cen_arr.append(std)

i = 0
std_red_cen_arr = []
for index1,bin_edge in enumerate(stellar_mass_bins):
    df_red_cen_deltav_arr = []
    if index1 == 9:
        break
    for index2,stellar_mass in enumerate(df_red_cen.log_cen_stellar_mass.values):
        if stellar_mass >= bin_edge and stellar_mass < stellar_mass_bins[index1+1]:
            df_red_cen_deltav_arr.append(df_red_cen.deltav.values[index2])
    N = len(df_red_cen_deltav_arr)
    std = std_func(df_red_cen_deltav_arr)
    std_red_cen_arr.append(std)

# Getting error from bootstraping for red and blue galaxies
std_red_cen_err, std_blue_cen_err = get_sigma_error(eco_keys)

fig1 = plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0,0])
plt.scatter(cen_stellar_mass_arr, deltav_arr, s=7, c=cen_colour_arr, 
    cmap='coolwarm')
cbar = plt.colorbar()
cbar.set_label(r'$\mathbf{(u-r)^e_{cen}}$', labelpad=15, fontsize=20)
plt.xlabel(r'$\mathbf{log_{10}\ M_{*,cen}}\ [\mathbf{M_{\odot}}]$', labelpad=10, 
    fontsize=25)
plt.ylabel(r'\boldmath$\Delta v\ \left[km/s\right]$', labelpad=10, fontsize=25)

left, bottom, width, height = [0.25, 0.20, 0.25, 0.25]
ax2 = fig1.add_axes([left, bottom, width, height])
ax2.errorbar(stellar_mass_bins[:-1],std_blue_cen_arr,yerr=std_blue_cen_err,
    color='#2d77e5',fmt='s',ecolor='#2d77e5',markersize=4,capsize=5,
    capthick=0.5)
ax2.errorbar(stellar_mass_bins[:-1],std_red_cen_arr,yerr=std_red_cen_err,
    color='#f46542',fmt='s',ecolor='#f46542',markersize=4,capsize=5,
    capthick=0.5)
ax2.set_ylabel(r'\boldmath$\sigma\ \left[km/s\right]$', fontsize=20)
fig1.tight_layout()
plt.show()