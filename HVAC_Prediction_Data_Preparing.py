import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import optimizers
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import concurrent.futures
import s3fs
from s3fs import S3FileSystem
import boto3
import logging


# 数据集路径 downloaded_data
data_dir = 'C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/downloaded_data'
upgrade_id = 1

#dataset informations
dataset_year = '2024'
dataset_name = 'comstock_amy2018_release_2'
state = 'NY'
#county = 'CO, Jefferson County'
dataset_path = f'nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{dataset_year}/{dataset_name}'


# metadata_df = pd.read_parquet(os.path.join(data_dir,'upgrade01_metadata_and_annual_results.parquet'))
# puma_county_crosswalk = metadata_df[['in.nhgis_county_gisjoin','in.nhgis_puma_gisjoin']] 
bldg_type_df = pd.read_excel('C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/comstock_building_type_list.xlsx')

# For Building Parsing Method
metadata = pd.read_parquet('C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/downloaded_data/NY_upgrade01_agg.parquet')
bldg_ids_in = metadata['bldg_id'].unique().tolist()
bldg_ids_in = bldg_ids_in[:1]
print(bldg_ids_in)


# #Listing all the GIScode (counties)
fs = s3fs.S3FileSystem(anon=True)
# base_path = f'oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{dataset_year}/{dataset_name}/timeseries_aggregates/by_county/upgrade={upgrade_id}/'
# #Listing all directories under the base path
# GIS_dirs = fs.ls(base_path)
# GIScode = [x.split('county=G')[1] for x in GIS_dirs]
# GIScode = GIScode[:5]
# print(GIScode)



# By Singular Building Parsing
path_dir = f'C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/downloaded_data/2024/comstock_amy2018_release_2/upgrade_1/NY'
# Get all files in the directory
bldg_ids = [x for x in os.listdir(path_dir)]
bldg_ids = [x.split('-')[0] for x in bldg_ids]
#bldg_ids = bldg_ids[:5]
# List to hold downloaded DataFrames
df_list = []
    
# Function to download and process a single file
def download_single_file(bldg_id):
    try:
        df = pd.read_parquet(
            os.path.join(path_dir,f'{bldg_id}-1.parquet'),
            columns=['timestamp', 'out.electricity.cooling.energy_consumption']
        )
        # Apply filtering to the downloaded data
        df['bldg_id'] = bldg_id
        df = df[df['out.electricity.cooling.energy_consumption'] > 0]
        return df
    except Exception as e:
        logging.warning(f"Failed to read {f}: {e}")
        df = []
        return df

# Download files in parallel using ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as file_executor:
    # Submit download tasks for all files
    file_futures = {file_executor.submit(download_single_file, bldg_id):  bldg_id for  bldg_id in  bldg_ids}
    
    # Collect results as they complete
    for future in concurrent.futures.as_completed(file_futures):
        result = future.result()
        if result is not None and not result.empty:
            df_list.append(result)
            print(len(df_list))

# Combine files from this county
bldg_df = pd.concat(df_list, ignore_index=True)
metadata['bldg_id'] = metadata['bldg_id'].astype(str).str.strip()

def process_weather_files(bldg_id):
    # Add weather data
    GIScode = metadata[metadata['bldg_id'] == f'{bldg_id}']['in.as_simulated_nhgis_county_gisjoin'].unique()
    weather_file = f's3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{dataset_year}/{dataset_name}/weather/amy2018/{GIScode[0]}_2018.csv'
    weather_data = pd.read_csv(weather_file, storage_options={'anon': True})
    
    # Get top  hottest day by temperature
    weather_data = weather_data[weather_data['Dry Bulb Temperature [°C]'] == weather_data['Dry Bulb Temperature [°C]'].max()].iloc[0]
    weather_data['bldg_id'] = bldg_id
    return weather_data


weather_total_data_df = []
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as bldg_exec:  
    bldg_futures = {bldg_exec.submit(process_weather_files,bldg_id): bldg_id for bldg_id in bldg_ids}
    
    for future in concurrent.futures.as_completed(bldg_futures):
        result = future.result()
        weather_total_data_df.append(result)
        print(len(weather_total_data_df))


weather_total_data_df = pd.DataFrame(weather_total_data_df)       
weather_total_data_df = weather_total_data_df.rename(columns={'date_time':'timestamp'})
weather_total_data_df['timestamp'] = pd.to_datetime(weather_total_data_df['timestamp'])   
bldg_df = bldg_df.merge(weather_total_data_df, on=['timestamp','bldg_id'])


# 4. Prepare input and output
# X: [total load, dry bulb temp], y: HVAC load
bldg_df.to_excel(os.path.join(data_dir,'HVAC_Prediction_selected_.xlsx'))


upgrade_total_data_df = pd.read_parquet('C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/downloaded_data/upgrade01_metadata_and_annual_results.parquet')
upgrade_total_data_df = upgrade_total_data_df.reset_index()
bldg_df['bldg_id'] = bldg_df['bldg_id'].astype(int)
upgrade_total_data_df = upgrade_total_data_df.rename(columns={'out.electricity.cooling.energy_consumption':'out.electricity.cooling.energy_consumption_total'})
upgrade_total_data_df = upgrade_total_data_df.merge(bldg_df, on='bldg_id')
print(upgrade_total_data_df[['Dry Bulb Temperature [°C]','Relative Humidity [%]','Wind Speed [m/s]']])


#Saving selected features to parquet (CNN HVAC Dissagregation)
selected_features_df = pd.read_excel('C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/selected_features_CNN_HVAC_Prediction.xlsx')
upgrade_selected_data_df = upgrade_total_data_df[selected_features_df['Feature Name']]
Applicability_series = upgrade_selected_data_df['applicability']
upgrade_selected_data_df = upgrade_selected_data_df[Applicability_series == True]
upgrade_selected_data_df.to_parquet(f'C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/downloaded_data/selected_data_new_upgrade{upgrade_id}_seasons_CNN_Pred.parquet', index=False)

