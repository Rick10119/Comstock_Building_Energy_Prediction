import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

upgrade_id = 1
upgrade_total_data_df = pd.read_parquet('C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/downloaded_data/upgrade01_metadata_and_annual_results.parquet')
weather_data_total_df = pd.read_excel('C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/weather_data_total_seasons.xlsx')
upgrade_total_data_df = upgrade_total_data_df.reset_index()
upgrade_total_data_df = upgrade_total_data_df.merge(weather_data_total_df, on='in.nhgis_county_gisjoin')
print(upgrade_total_data_df[['Dry Bulb Temperature [°C]','Relative Humidity [%]','Wind Speed [m/s]']])

#Finding baseload data for HVAC Disaggregation (CNN)
non_hvac_loads = [
        'out.electricity.interior_equipment.energy_consumption',
        'out.electricity.interior_lighting.energy_consumption',
        'out.electricity.exterior_lighting.energy_consumption',
        'out.electricity.water_systems.energy_consumption',
        'out.electricity.refrigeration.energy_consumption'
]
upgrade_total_data_df['augmented_baseload'] = upgrade_total_data_df['out.electricity.total.energy_consumption'] - upgrade_total_data_df[non_hvac_loads].sum(axis=1)
#upgrade_total_data_df = upgrade_total_data_df[upgrade_total_data_df['out.electricity.cooling.energy_consumption'] >= 50000]
upgrade_total_data_df = upgrade_total_data_df[upgrade_total_data_df['season'] == 'summer']



#Saving selected features to parquet (CNN HVAC Dissagregation)
selected_features_df = pd.read_excel('C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/selected_features_CNN.xlsx')
upgrade_selected_data_df = upgrade_total_data_df[selected_features_df['Feature Name']]
Applicability_series = upgrade_selected_data_df['applicability']
upgrade_selected_data_df = upgrade_selected_data_df[Applicability_series == True]
#upgrade_selected_data_df.to_parquet(f'C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/downloaded_data/selected_data_new_NY_seasons.parquet', index=False)
upgrade_selected_data_df.to_parquet(f'C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/downloaded_data/selected_data_new_upgrade{upgrade_id}_seasons_CNN.parquet', index=False)



# #Saving selected features to parquet （CNN DR Response Prediction)
# selected_features_df = pd.read_excel('C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/selected_features.xlsx')
# upgrade_selected_data_df = upgrade_total_data_df[selected_features_df['Feature Name']]
# Applicability_series = upgrade_selected_data_df['applicability']
# upgrade_selected_data_df = upgrade_selected_data_df[Applicability_series == True]
# #upgrade_selected_data_df.to_parquet(f'C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/downloaded_data/selected_data_new_NY_seasons.parquet', index=False)
# upgrade_selected_data_df.to_parquet(f'C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/downloaded_data/selected_data_new_upgrade{upgrade_id}_seasons.parquet', index=False)

