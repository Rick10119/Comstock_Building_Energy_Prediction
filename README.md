**DR Potential prediction task:**

DR\_Potential\_Prediction\_CNN.py : Demand response potential prediction CNN model

handle\_weather\_data\_season.py : Download and pre-process weather data based on each respective county data and stores it in weather\_data\_total\_seasons.xlsx.

upgrade01\_metadata\_and\_annual\_results.parquet : Metadata and annual energy consumption data from Comstock simulation output (all states) (Download from https://data.openei.org/s3\_viewer?bucket=oedi-data-lake\&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2024%2Fcomstock\_amy2018\_release\_1%2Fmetadata\_and\_annual\_results%2Fnational%2Fparquet%2F)

Weather\_data\_merger.py : Merging the excel weather data file with upgrade01\_metadata\_and\_annual\_results.parquet based on county code and preprocess the data to eliminate pollution

selected\_features.xlsx : Selected features name

selected\_output.xlsx : Selected output name





**HVAC load Dissagregation \& prediction task:**

Weather\_data\_merger.py : Merging the excel weather data file with upgrade01\_metadata\_and\_annual\_results.parquet based on county code and preprocess the data to eliminate pollution

HVAC\_Dissagregation\_CNN\_model.py: 2D CNN model for HVAC Dissagregation (using buildings total energy consumption and temperature as features)

HVAC\_Dissagregation.py: Preparing data for 2D CNN model for HVAC Dissagregation



HVAC\_Prediction\_Download.py : Download data for HVAC\_Prediction\_Data\_Preparing.py  (Comstock New York State individual buildings data)

HVAC\_Prediction\_Data\_Preparing.py : Preprocess data for data for  HVAC\_Prediction\_Model.py

HVAC\_Prediction\_Model.py : multi dimensional buildings HVAC prediction model (using metadata and climate/weather data as features)

selected\_features.xlsx : Selected features name

selected\_output.xlsx : Selected output name





**Note:**

Change all the file data directories to suit your saved folder

Metadata: Constant buildings characteristic data (area, built year, number of stories, etc.)

