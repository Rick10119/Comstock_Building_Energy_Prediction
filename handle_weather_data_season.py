# %% 1. 导入库
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import concurrent.futures


# 数据集路径 downloaded_data
data_dir = 'C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/downloaded_data'
upgrade_id = 1

#dataset informations
dataset_year = '2024'
dataset_name = 'comstock_amy2018_release_1'
state = 'NY'
#county = 'CO, Jefferson County'
dataset_path = f'nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{dataset_year}/{dataset_name}'



# 读取数据 upgrade{upgrade_id}_selected_data.parquet
total_upgrade_data_path = os.path.join(data_dir, 'upgrade01_metadata_and_annual_results.parquet')
#total_upgrade_data_path = os.path.join(data_dir, 'NY_baseline_metadata_and_annual_results.parquet')
#total_upgrade_data_path = f's3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{dataset_year}/{dataset_name}/metadata_and_annual_results/national/parquet/upgrade02_metadata_and_annual_results.parquet'

upgrade_total_data_df = pd.read_parquet(total_upgrade_data_path)
#upgrade_total_data_df = pd.read_parquet(total_upgrade_data_path, storage_options={'anon': True})
upgrade_total_data_df = upgrade_total_data_df.reset_index()



i= 0
def process_weather(GIScode):
    weather_data = pd.read_csv(f's3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{dataset_year}/{dataset_name}/weather/amy2018/{GIScode}_2018.csv')
    weather_data = weather_data.drop(columns=['date_time'])
    weather_data['season'] = pd.cut(weather_data['Dry Bulb Temperature [°C]'],
                                    labels=['winter','shoulder','summer'],
                                    bins=[-np.inf,12.777,21.111,np.inf])
    
    #print(len(weather_data[weather_data ['season'] == 'summer']))
    season_maxs = []
    for season in weather_data['season'].unique():
        season_df = weather_data[weather_data['season'] == season]
        if not season_df.empty:
            season_max = season_df[season_df['Dry Bulb Temperature [°C]'] == season_df['Dry Bulb Temperature [°C]'].max()].iloc[0]
            season_max['in.nhgis_county_gisjoin'] = GIScode
            season_maxs.append(season_max)
            #print(season_maxs)
    return season_maxs
        

weather_data_total = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    results = executor.map(process_weather,upgrade_total_data_df['in.nhgis_county_gisjoin'].unique())
    for res in results:
        weather_data_total.extend(res)
        print(len(weather_data_total))
        if len(weather_data_total) >= 8609:
            break


#weather_data_total_df = pd.concat(weather_data_total, ignore_index=True)
weather_data_total_df = pd.DataFrame(weather_data_total)
weather_data_total_df.to_excel('C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/weather_data_total_seasons.xlsx', index=False)
upgrade_total_data_df = upgrade_total_data_df.merge(weather_data_total_df, on='in.nhgis_county_gisjoin')
print(upgrade_total_data_df[['Dry Bulb Temperature [°C]','Relative Humidity [%]','Wind Speed [m/s]']])


# %%
