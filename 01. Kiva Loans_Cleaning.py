# Linear algebra
import numpy as np

# Data processing
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Exploratory
from exploration_utils import dataset_exploration
pd.set_option('display.max_rows', None)

#*********************************************************************************************

############# Read dataset

df_loans = pd.read_csv("C:/Users/hp/Documents/KiVA/1. File Data/1. Data Raw/kiva_loans.csv")
df_locations = pd.read_csv("C:/Users/hp/Documents/KiVA/1. File Data/1. Data Raw/kiva_locations.csv")
df_theme_id = pd.read_csv("C:/Users/hp/Documents/KiVA/1. File Data/1. Data Raw/loan_theme_ids.csv")
df_theme_region = pd.read_csv("C:/Users/hp/Documents/KiVA/1. File Data/1. Data Raw/loan_themes_by_region.csv")



############# Preliminary observations

# 1. merging datasets
merged_dataset = df_loans.merge(df_theme_id, how='left', on='id')
merged_dataset2 = merged_dataset.merge(df_theme_region, how='left', on=['Partner ID', 'Loan Theme ID', 'country', 'region'])

# 2. dataset explorations
dataset_exploration(merged_dataset2)

# 3. Checking data timeframe

# Convert columns to datetime if they are not already
merged_dataset2['posted_time'] = pd.to_datetime(merged_dataset2['posted_time'])
merged_dataset2['disbursed_time'] = pd.to_datetime(merged_dataset2['disbursed_time'])
merged_dataset2['funded_time'] = pd.to_datetime(merged_dataset2['funded_time'])

# Get the earliest and latest dates for each timestamp
timeframe_summary = {
    'First Posted': merged_dataset2['posted_time'].min(),
    'Last Posted': merged_dataset2['posted_time'].max(),
    'First Disbursed': merged_dataset2['disbursed_time'].min(),
    'Last Disbursed': merged_dataset2['disbursed_time'].max(),
    'First Funded': merged_dataset2['funded_time'].min(),
    'Last Funded': merged_dataset2['funded_time'].max()
}

timeframe_summary


############# Feature Engineering

# 1. Drop duplicate columns - 'country_code' [we already have country], 'date'[date here is similar to posted_time], currency [since loan are disburse in USD]
# 2. Clean up missing values - certain variables have missing value together so we start with largest missing value, and see how we go from there
# 3. Create a proper variables from 'borrower_genders' and tags
# 4. Clean up some categorical variables - lower case, remove whitespace, high cardinality?
# 5. Other domain checking
# 6. Rearrange name for better look

#1
#remove unnecessary columns
df_cleaned = merged_dataset2.drop(columns=['country_code', 'date', 'currency','partner_id',
                                           'tags','Loan Theme ID','Partner ID','forkiva',
                                           'geocode_old','ISO','LocationName',
                                           'mpi_region','mpi_geo'])

#2
# Finding columns with missing values
columns_with_missing_values = df_cleaned.columns[df_cleaned.isnull().any()].tolist()
df_cleaned.isnull().sum()

#remove duplicate columns that have higher missing values
df_cleaned = df_cleaned.dropna(subset=['use',
                                         'region',
                                         'disbursed_time',
                                         'funded_time',
                                         'borrower_genders',
                                         'Loan Theme Type_x',
                                         'Field Partner Name',
                                         'sector_y',
                                         'Loan Theme Type_y',
                                         'number',
                                         'amount',
                                         'geocode',
                                         'names',
                                         'geo',
                                         'lat',
                                         'lon',
                                         'rural_pct'],
                               how='all')

df_cleaned = df_cleaned.dropna(subset=['Field Partner Name',
                                         'sector_y',
                                         'Loan Theme Type_y',
                                         'number',
                                         'amount',
                                         'geocode',
                                         'geo'],
                               how='all')

df_cleaned = df_cleaned.drop(columns=['geocode', 'names', 'lat','lon'])
df_cleaned = df_cleaned.dropna(subset=['funded_time',
                                       'rural_pct'],
                               how='all')
df_cleaned = df_cleaned.dropna(subset=['funded_time',
                                        'rural_pct',
                                        'use'])                                      


#3
df_cleaned['borrower_genders'].value_counts()
def count_gender(genders, gender_to_count):
    if pd.isna(genders):
        return 0
    return sum([1 for gender in genders.split(', ') if gender.strip() == gender_to_count])

df_cleaned['no_male_borrower'] = df_cleaned['borrower_genders'].apply(lambda x: count_gender(x, 'male'))
df_cleaned['no_female_borrower'] = df_cleaned['borrower_genders'].apply(lambda x: count_gender(x, 'female'))
df_cleaned['borrower_type'] = df_cleaned.apply(lambda row: 'single' if (row['no_male_borrower'] + row['no_female_borrower']) == 1 else 'group', axis=1)
df_cleaned = df_cleaned.drop(columns=['borrower_genders'])


#4
categorical_vars = df_cleaned.select_dtypes(include=['object'])
categorical_vars.columns

#lowercase and whitespace
df_cleaned[['activity', 'sector_x', 'use', 'country', 'region',
            'repayment_interval', 'Loan Theme Type_x', 'Field Partner Name',
            'sector_y', 'Loan Theme Type_y', 'geo', 'borrower_type']] = df_cleaned[['activity', 'sector_x', 'use', 'country', 'region',
                                                                                    'repayment_interval', 'Loan Theme Type_x', 'Field Partner Name',
                                                                                    'sector_y', 'Loan Theme Type_y', 'geo', 'borrower_type']].apply(lambda x: x.str.lower())
df_cleaned[['activity', 'sector_x', 'use', 'country', 'region',
            'repayment_interval', 'Loan Theme Type_x', 'Field Partner Name',
            'sector_y', 'Loan Theme Type_y', 'geo', 'borrower_type']] = df_cleaned[['activity', 'sector_x', 'use', 'country', 'region',
                                                                                    'repayment_interval', 'Loan Theme Type_x', 'Field Partner Name',
                                                                                    'sector_y', 'Loan Theme Type_y', 'geo', 'borrower_type']].apply(lambda x: x.str.strip())

#cardinality
categorical_vars = df_cleaned.select_dtypes(include=['object'])
categorical_vars.nunique()

region_counts = df_cleaned.groupby('country')['region'].apply(lambda x: x.value_counts()).reset_index(name='counts')
region_counts = region_counts.reset_index().rename(columns={'level_1': 'region'})
country_data = region_counts[region_counts['country'] == 'mali']
country_data_sorted = country_data.sort_values(by='region')

activity_counts = df_cleaned.groupby('sector_x')['activity'].apply(lambda x: x.value_counts()).reset_index(name='counts')
activityuse_counts = df_cleaned.groupby(['sector_x','activity'])['use'].apply(lambda x: x.value_counts()).reset_index(name='counts')


#5
inconsistent_rows = df_cleaned[df_cleaned['funded_amount'] > df_cleaned['loan_amount']]
df_cleaned['amount_difference'] = df_cleaned['loan_amount'] - df_cleaned['funded_amount']
df_cleaned['amount_difference'].describe()

# Check for invalid latitude and longitude values
invalid_lat = df_cleaned[(df_cleaned['lat'] < -90) | (df_cleaned['lat'] > 90)]
invalid_lng = df_cleaned[(df_cleaned['lng'] < -180) | (df_cleaned['lng'] > 180)]

#6
df_cleaned = df_cleaned.rename(columns={'sector_x': 'sector_borrowers',
                                        'Loan Theme Type_x':'loan_theme_borrowers',
                                        'sector_y': 'sector_loaners',
                                        'Loan Theme Type_y':'loan_theme_loaners'})


dataset_exploration(df_cleaned)


############# Save cleaned dataset

df_cleaned.to_parquet('C:/Users/hp/Documents/KiVA/1. File Data/2. Data Cleaned/cleaned_data.parquet')











