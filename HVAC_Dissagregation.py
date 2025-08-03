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

# # For Building Parsing Method
# metadata = pd.read_parquet('C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/downloaded_data/NY_upgrade01_agg.parquet')
# bldg_ids = metadata['bldg_id'].unique().tolist()
# GIScode = metadata['in.as_simulated_nhgis_county_gisjoin'].unique().tolist()


#Listing all the GIScode (counties)
fs = s3fs.S3FileSystem(anon=True)
base_path = f'oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{dataset_year}/{dataset_name}/timeseries_aggregates/by_county/upgrade={upgrade_id}/'
#Listing all directories under the base path
GIS_dirs = fs.ls(base_path)
GIScode = [x.split('county=G')[1] for x in GIS_dirs]
GIScode = GIScode[:5]
print(GIScode)


# By County Parsing
def process_county_files(GIScode):
    file_path = f'oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{dataset_year}/{dataset_name}/timeseries_aggregates/by_county/upgrade={upgrade_id}/county=G{GIScode}/'
    
    # Get all CSV files in the county directory
    all_files = [f's3://{file}' for file in fs.ls(file_path) if file.endswith('csv')]
    
    # List to hold downloaded DataFrames
    df_list = []
    
    # Function to download and process a single file
    def download_single_file(file):
        try:
            df = pd.read_csv(
                file,
                storage_options={'anon': True},
                usecols=['timestamp', 'out.electricity.cooling.energy_consumption.kwh', 
                         'out.electricity.interior_equipment.energy_consumption.kwh',
                         'out.electricity.interior_lighting.energy_consumption.kwh',
                         'out.electricity.exterior_lighting.energy_consumption.kwh',
                         'out.electricity.water_systems.energy_consumption.kwh',
                         'out.electricity.refrigeration.energy_consumption.kwh',
                         'out.electricity.total.energy_consumption.kwh']
            )
            # Apply filtering to the downloaded data
            return df[df['out.electricity.cooling.energy_consumption.kwh'] >= 100]
        except Exception as e:
            logging.warning(f"Failed to read {file}: {e}")
            return None

    # Download files in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as file_executor:
        # Submit download tasks for all files
        file_futures = {file_executor.submit(download_single_file, file): file for file in all_files}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(file_futures):
            result = future.result()
            if result is not None and not result.empty:
                df_list.append(result)
                print(len(df_list))

    if not df_list:
        logging.warning(f"No valid data found for county {GIScode}")
        return None

    # Combine files from this county
    county_df = pd.concat(df_list, ignore_index=True)
    
    # Process non-HVAC loads
    non_hvac_loads = [
        'out.electricity.interior_equipment.energy_consumption.kwh',
        'out.electricity.interior_lighting.energy_consumption.kwh',
        'out.electricity.exterior_lighting.energy_consumption.kwh',
        'out.electricity.water_systems.energy_consumption.kwh',
        'out.electricity.refrigeration.energy_consumption.kwh'
    ]
    
    county_df['augmented_baseload'] = county_df['out.electricity.total.energy_consumption.kwh'] - county_df[non_hvac_loads].sum(axis=1)
    county_df = county_df.drop(columns=non_hvac_loads)

    # Add weather data
    weather_file = f's3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{dataset_year}/{dataset_name}/weather/amy2018/G{GIScode}_2018.csv'
    weather_data = pd.read_csv(weather_file, storage_options={'anon': True}, usecols=['date_time', 'Dry Bulb Temperature [°C]'])
    # weather_data['season'] = pd.cut(weather_data['Dry Bulb Temperature [°C]'],
    #                                 labels=['winter','shoulder','summer'],
    #                                 bins=[-np.inf,12.777,21.111,np.inf])
    #weather_data = weather_data[weather_data['season'] == 'summer']

    # Get top 3 hottest months by temperature
    weather_data = weather_data.sort_values('Dry Bulb Temperature [°C]', ascending=False).head(2160)
    weather_data = weather_data.rename(columns={'date_time': 'timestamp'})                         
    
    return county_df.merge(weather_data, on='timestamp')


upgrade_total_data_df = []
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as county_exec:  # Fewer workers
    county_futures = {county_exec.submit(process_county_files, code): code for code in GIScode}
    
    for future in concurrent.futures.as_completed(county_futures):
        result = future.result()
        if result is not None and not result.empty:
            upgrade_total_data_df.append(result)
            print(len(upgrade_total_data_df))

if upgrade_total_data_df:
    upgrade_total_data_df = pd.concat(upgrade_total_data_df, ignore_index=True)
else:
    upgrade_total_data_df = pd.DataFrame()



# # By Singular Building Parsing
# def process_building_files(bldg_id):
    
#     df = pd.read_parquet(
#         f's3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{dataset_year}/{dataset_name}/timeseries_individual_buildings/by_state/upgrade={upgrade_id}/state={state}/{bldg_id}-1.parquet',
#         storage_options={'anon': True},
#         columns=['timestamp', 'out.electricity.cooling.energy_consumption', 
#                     'out.electricity.interior_equipment.energy_consumption',
#                     'out.electricity.interior_lighting.energy_consumption',
#                     'out.electricity.exterior_lighting.energy_consumption',
#                     'out.electricity.water_systems.energy_consumption',
#                     'out.electricity.refrigeration.energy_consumption',
#                     'out.electricity.total.energy_consumption']
#     )
#     # Apply filtering to the downloaded data
#     df = df[df['out.electricity.cooling.energy_consumption'] >= 0.5]
#     df['bldg_id'] = bldg_id
 
#     # # Process non-HVAC loads
#     # non_hvac_loads = [
#     #                     'out.electricity.interior_equipment.energy_consumption',
#     #                     'out.electricity.interior_lighting.energy_consumption',
#     #                     'out.electricity.exterior_lighting.energy_consumption',
#     #                     'out.electricity.water_systems.energy_consumption',
#     #                     'out.electricity.refrigeration.energy_consumption',
#     #                 ]

#     # df['augmented_baseload'] = df['out.electricity.total.energy_consumption'] - df[non_hvac_loads].sum(axis=1)
#     # df = df.drop(columns=non_hvac_loads)

#     # Add weather data
#     #GIScode = metadata[metadata['bldg_id'] == bldg_id]['in.as_simulated_nhgis_county_gisjoin'].unique()
#     # weather_file = f's3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{dataset_year}/{dataset_name}/weather/amy2018/{GIScode[0]}_2018.csv'
#     # weather_data = pd.read_csv(weather_file, storage_options={'anon': True}, usecols=['date_time', 'Dry Bulb Temperature [°C]'])
#     # weather_data = weather_data.rename(columns={'date_time': 'timestamp'})
#     # weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
#     # df = df.merge(weather_data, on='timestamp')
#     return df


# upgrade_total_data_df = []
# with concurrent.futures.ThreadPoolExecutor(max_workers=12) as county_exec:  # Fewer workers
#     county_futures = {county_exec.submit(process_building_files, bldg_id): bldg_id for bldg_id in bldg_ids}
    
#     for future in concurrent.futures.as_completed(county_futures):
#         result = future.result()
#         upgrade_total_data_df.append(result)
#         print(len(upgrade_total_data_df))

# if upgrade_total_data_df:
#     upgrade_total_data_df = pd.concat(upgrade_total_data_df, ignore_index=True)
# else:
#     upgrade_total_data_df = pd.DataFrame()


# def process_weather(GIScode):
#     weather_file = f's3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{dataset_year}/{dataset_name}/weather/amy2018/{GIScode}_2018.csv'
#     weather_data = pd.read_csv(weather_file, storage_options={'anon': True}, usecols=['date_time', 'Dry Bulb Temperature [°C]'])
#     weather_data = weather_data.rename(columns={'date_time': 'timestamp'})
#     weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
#     #print(weather_data)
#     return weather_data


# df_list = []
# with concurrent.futures.ThreadPoolExecutor(max_workers=4) as weather_exec:  # Fewer workers
#     weather_futures = {weather_exec.submit(process_weather, code): code for code in GIScode}
#     for future in concurrent.futures.as_completed(weather_futures):
#         result = future.result()
#         df_list.append(result)
#         print(len(df_list))

# upgrade_total_data_df = upgrade_total_data_df.merge(df_list, on=['timestamp','bldg_id'])


# # Process non-HVAC loads
# non_hvac_loads = [
#                     'out.electricity.interior_equipment.energy_consumption',
#                     'out.electricity.interior_lighting.energy_consumption',
#                     'out.electricity.exterior_lighting.energy_consumption',
#                     'out.electricity.water_systems.energy_consumption',
#                     'out.electricity.refrigeration.energy_consumption',
#                 ]

# upgrade_total_data_df['augmented_baseload'] = upgrade_total_data_df['out.electricity.total.energy_consumption'] - upgrade_total_data_df[non_hvac_loads].sum(axis=1)
# df = upgrade_total_data_df.drop(columns=non_hvac_loads)

# 4. Prepare input and output
# X: [total load, dry bulb temp], y: HVAC load
upgrade_total_data_df.to_excel(os.path.join(data_dir,'HVAC_dissagregation_selected.xlsx'))
X = upgrade_total_data_df.drop(columns=['timestamp','out.electricity.cooling.energy_consumption.kwh','out.electricity.total.energy_consumption.kwh'])
y = upgrade_total_data_df['out.electricity.cooling.energy_consumption.kwh']
