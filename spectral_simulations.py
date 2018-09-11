# Spectral simulations Part A/B

import pandas as pd
import matplotlib.pyplot as plt
import math

# Part A

dir = ''

A = math.pi * math.pow(3, 2)
RAYS = 200000
ZENITH_ANGLES = [-50,-40,-30,-20,-10,0,10,20,30,40,50]
SOLAR_ZENITH = 40
LAI = [0.4,0.8,1.2,1.6,2,2.4]

# A1

def calc_brf(radiance, solar_zenith, rays, a):
    return radiance / (math.cos(math.radians(solar_zenith)) * rays / a) * math.pi

df_A1 = pd.DataFrame()
df_A1 = pd.read_csv(dir + 'test.csv', sep=',', nrows=11, header=None, names=['Radiance'])
df_A1['View-zenith angle'] = ZENITH_ANGLES
df_A1['BRF'] = df_A1['Radiance'].apply(calc_brf, args = (SOLAR_ZENITH, RAYS, A))

fig, ax = plt.subplots()
ax.set(xlabel='View-zenith angle', ylabel='BRF')
df_A1.plot(x='View-zenith angle', y='BRF', ax=ax, style='-o', xticks=ZENITH_ANGLES)

# A2

# With 100000 rays the standard deviation found out to be 0.009013958918108. Lowering ray count from 100000 did end up with higher standard deviation.
# 100000 rays is a good limit to be sure that the standard deviation would stay below 0.01.

# A3

A_3 = math.pow(30, 2) * math.pi
RAYS_OPTIMUM = 100000

# NIR

A3_NIR = pd.read_csv(dir + 'test3.csv', sep=',', header=None, usecols=[1,2,3,4,5,6,7,8,9,10])
A3_NIR = A3_NIR.apply(calc_brf, args = (SOLAR_ZENITH, RAYS_OPTIMUM, A_3))

# RED

A3_RED = pd.read_csv(dir + 'test4.csv', sep=',', header=None, usecols=[1,2,3,4,5,6,7,8,9,10])
A3_RED = A3_RED.apply(calc_brf, args = (SOLAR_ZENITH, RAYS_OPTIMUM, A_3))

df_A3_LAI = pd.DataFrame()
df_A3_ZENITH = pd.DataFrame()

df_A3_LAI['LAI'] = LAI
df_A3_LAI['SumNIR'] = A3_NIR.sum(axis=1)
df_A3_LAI['SumRED'] = A3_RED.sum(axis=1)

# NDVI

df_A3_LAI['NDVI'] = (df_A3_LAI['SumNIR'] - df_A3_LAI['SumRED']) / (df_A3_LAI['SumNIR'] + df_A3_LAI['SumRED'])

fig3, ax3 = plt.subplots()
ax3.set(xlabel='LAI', ylabel='BRF')
df_A3_LAI.plot(x='LAI', y='SumNIR', ax=ax3, style='-o', xticks=LAI)
df_A3_LAI.plot(x='LAI', y='SumRED', ax=ax3, style='-o', xticks=LAI)
df_A3_LAI.plot(x='LAI', y='NDVI', ax=ax3, style='-o', xticks=LAI)

# Low LAI RED

df_A3_ZENITH['View-zenith angle'] = ZENITH_ANGLES
A3_LRED = pd.read_csv(dir + 'test5_red.csv', sep=',', header=None, usecols=[1,2,3,4,5,6,7,8,9,10])
A3_LRED = A3_LRED.apply(calc_brf, args = (SOLAR_ZENITH, RAYS_OPTIMUM, A_3))
df_A3_ZENITH['SumLRED'] = A3_LRED.sum(axis=1)

# Low LAI NIR

A3_LNIR = pd.read_csv(dir + 'test6_NIR.csv', sep=',', header=None, usecols=[1,2,3,4,5,6,7,8,9,10])
A3_LNIR = A3_LNIR.apply(calc_brf, args = (SOLAR_ZENITH, RAYS_OPTIMUM, A_3))
df_A3_ZENITH['SumLNIR'] = A3_LNIR.sum(axis=1)

# High LAI RED

A3_HRED = pd.read_csv(dir + 'test7_highred.csv', sep=',', header=None, usecols=[1,2,3,4,5,6,7,8,9,10])
A3_HRED = A3_HRED.apply(calc_brf, args = (SOLAR_ZENITH, RAYS_OPTIMUM, A_3))
df_A3_ZENITH['SumHRED'] = A3_HRED.sum(axis=1)

# High LAI NIR

A3_HNIR = pd.read_csv(dir + 'test8_highNIR.csv', sep=',', header=None, usecols=[1,2,3,4,5,6,7,8,9,10])
A3_HNIR = A3_HNIR.apply(calc_brf, args = (SOLAR_ZENITH, RAYS_OPTIMUM, A_3))
df_A3_ZENITH['SumHNIR'] = A3_HNIR.sum(axis=1)

# NDVI Low LAI

df_A3_ZENITH['NDVI_LOW'] = (df_A3_ZENITH['SumLNIR'] - df_A3_ZENITH['SumLRED']) / (df_A3_ZENITH['SumLNIR'] + df_A3_ZENITH['SumLRED'])

# NDVI HIGH LAI

df_A3_ZENITH['NDVI_HIGH'] = (df_A3_ZENITH['SumHNIR'] - df_A3_ZENITH['SumHRED']) / (df_A3_ZENITH['SumHNIR'] + df_A3_ZENITH['SumHRED'])

fig4, ax4 = plt.subplots()
ax4.set(xlabel='View-zenith angle',ylabel='BRF')
df_A3_ZENITH.plot(x='View-zenith angle', y='SumHRED', ax=ax4, style='-o', xticks=ZENITH_ANGLES)
df_A3_ZENITH.plot(x='View-zenith angle', y='SumLRED', ax=ax4, style='-o', xticks=ZENITH_ANGLES)

fig5, ax5 = plt.subplots()
ax5.set(xlabel='View-zenith angle', ylabel='BRF')
df_A3_ZENITH.plot(x='View-zenith angle', y='SumLNIR', ax=ax5, style='-o', xticks=ZENITH_ANGLES)
df_A3_ZENITH.plot(x='View-zenith angle', y='SumHNIR', ax=ax5, style='-o', xticks=ZENITH_ANGLES)

fig6, ax6 = plt.subplots()
ax6.set(xlabel='View-zenith angle', ylabel='BRF')
df_A3_ZENITH.plot(x='View-zenith angle', y='NDVI_LOW', ax=ax6, style='-o', xticks=ZENITH_ANGLES)
df_A3_ZENITH.plot(x='View-zenith angle', y='NDVI_HIGH', ax=ax6, style='-o', xticks=ZENITH_ANGLES)
print(df_A3_ZENITH)

# A4

# Part B

filePath = ''
excel_sheets = ['forest_data', 'tree_leaf_spectra', 'forest_floor_spectra']

forestBrfDataFrame = pd.DataFrame()
indiciesDataFrame = pd.DataFrame(columns=('Forest_ID', 'MCARI', 'CI', 'VARIg', 'LAI'))
data = pd.read_excel(filePath, sheet_name=excel_sheets, skiprows=[0])

data_lenght = len(data[excel_sheets[0]])
tree_leaf_spectra_table = data[excel_sheets[1]]
forest_floor_spectra = data[excel_sheets[2]]
bands = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190]

forestBrfDataFrame['Wavelength (nm)'] = bands
fig6, ax6 = plt.subplots() 
ax6.set(ylabel='BRF')

def calculateForestBRF(N, n=0):
    if (N == n):
        return

    forest_table = data[excel_sheets[0]].iloc[[n]]
    ID = forest_table['Forest_ID'].values[0]
    DIFN = forest_table['DIFN'].values[0]
    LAI = forest_table['LAI'].values[0]
    sensor_angle = forest_table['GAPS1'].values[0]
    sun_angle = forest_table['GAPS3'].values[0]
    allBRF = []
    p = 1 - (1 - DIFN) / LAI
    f = 0.0593 * LAI + 0.5

    for band in bands:
        if (forest_table['Tree species'].values[0] == 'Pine'):
            leaf_albedo_W = tree_leaf_spectra_table.loc[tree_leaf_spectra_table['wavelength, nm'] == band]['pine'].values[0]
        else:
            leaf_albedo_W = tree_leaf_spectra_table.loc[tree_leaf_spectra_table['wavelength, nm'] == band]['birch'].values[0]
        pGround = forest_floor_spectra.loc[forest_floor_spectra['wavelength, nm'] == band]['forest floor vegetation BRF (rho_ground)'].values[0]
        #BRF
        allBRF.append(sensor_angle * sun_angle * pGround + f * (1 - sun_angle) * (leaf_albedo_W - p * leaf_albedo_W) / (1 - p * leaf_albedo_W))
    forestBrfDataFrame[ID] = allBRF
    forestBrfDataFrame.plot(x='Wavelength (nm)', y=n, ax=ax6, style='-o')
    calculateForestBRF(N, n + 1)

calculateForestBRF(data_lenght)

def calculateVegetationIndices(N, n=0):
    if (N == n):
        return
    forest_table = data[excel_sheets[0]].iloc[[n]]
    ID = forest_table['Forest_ID'].values[0]
    LAI = forest_table["LAI"].values[0]

    B2 = forestBrfDataFrame[ID].loc[forestBrfDataFrame['Wavelength (nm)'] == bands[1]].values[0]
    B3 = forestBrfDataFrame[ID].loc[forestBrfDataFrame['Wavelength (nm)'] == bands[2]].values[0]
    B4 = forestBrfDataFrame[ID].loc[forestBrfDataFrame['Wavelength (nm)'] == bands[3]].values[0]
    B5 = forestBrfDataFrame[ID].loc[forestBrfDataFrame['Wavelength (nm)'] == bands[4]].values[0]
    B7 = forestBrfDataFrame[ID].loc[forestBrfDataFrame['Wavelength (nm)'] == bands[6]].values[0]

    MCARI = ( (B5 - B4) - 0.2 * (B5 - B3) ) * (B5 - B4)
    CI = (B7 / B5) - 1
    VARIg = (B3 - B4) / (B3 + B4 - B2)

    indiciesDataFrame.loc[n] = [ID, MCARI, CI, VARIg, LAI]

    calculateVegetationIndices(N, n + 1)

calculateVegetationIndices(data_lenght)

figure, (ax7, ax8, ax9) = plt.subplots(1, 3)
ax7.set_xlim(0, 0.005)
indiciesDataFrame.plot.scatter(x='MCARI', y='LAI', ax=ax7)
indiciesDataFrame.plot.scatter(x='CI', y='LAI', ax=ax8)
indiciesDataFrame.plot.scatter(x='VARIg', y='LAI', ax=ax9)



### BONUS ###

from sklearn.linear_model import LinearRegression
import numpy as np

CI_value = 3.8

X = np.asarray(indiciesDataFrame['LAI'])
y = np.asarray(indiciesDataFrame['CI'])

model = LinearRegression()
model.fit(X[:, np.newaxis], y[:, np.newaxis])

X_predict = np.array([CI_value])
y_predict = model.predict(X_predict[:, np.newaxis])
print(y_predict)



