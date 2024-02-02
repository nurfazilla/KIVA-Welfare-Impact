# Linear algebra
import numpy as np

# Data processing
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Exploratory
from exploration_utils import dataset_exploration


#*********************************************************************************************

############# Read dataset

df = pd.read_parquet("C:/Users/hp/Documents/KiVA/1. File Data/2. Data Cleaned/cleaned_data.parquet")
dataset_exploration(df)
df.columns.to_list()

############# Exploratory Data Analysis

######## 1. Examine the distribution of borrower

################# BY COUNTRY AND REGION

# Calculate count of borrowers per country
borrower_count_by_country = df['country'].value_counts().reset_index()
borrower_count_by_country.columns = ['country', 'borrower_count']

# Get the top 4 countries with the most borrowers
top_countries = borrower_count_by_country.head(4)['country']

# Plot the overall count of borrowers by country
plt.figure(figsize=(12, 20))
sns.barplot(data=borrower_count_by_country, x='borrower_count', y='country', palette = 'Blues_d')
plt.title('Count of Borrowers by Country')
plt.xlabel('Number of Borrowers')
plt.ylabel('Country')
plt.show()

#Checking the unique region for each country
region_counts_by_country = df.groupby('country')['region'].nunique()
region_counts_by_country = region_counts_by_country.sort_values(ascending=False)

plt.figure(figsize=(12, 20))
sns.barplot(x=region_counts_by_country.values, y=region_counts_by_country.index, palette='Blues_d')
plt.xlabel('Number of Unique Regions')
plt.ylabel('Country')
plt.title('Number of Unique Regions by Each Country')
plt.show()

#Clean up the region a bit to reduce some cardinality
def clean_region(region):
    # Remove any commas or hyphens from the string
    cleaned_string = region.replace(',', '').replace('-', ' ')
    # Split the string into words
    words = cleaned_string.split()
    # Take the last two words if there are enough words, otherwise take what is available
    cleaned_region = ' '.join(words[-2:]) if len(words) >= 2 else ' '.join(words)
    return cleaned_region

# Apply the function to the 'region' column in dataframe
df['cleaned_region'] = df['region'].apply(clean_region)

#Checking the unique region again for each country
region_counts_by_country = df.groupby('country')['cleaned_region'].nunique()
region_counts_by_country = region_counts_by_country.sort_values(ascending=False)

plt.figure(figsize=(12, 20))
sns.barplot(x=region_counts_by_country.values, y=region_counts_by_country.index, palette='Blues_d')
plt.xlabel('Number of Unique Regions')
plt.ylabel('Country')
plt.title('Number of Unique Regions by Each Country')
plt.show()


# Now create a 2x2 grid for the top 4 countries showing their top 10 regions
fig, axes = plt.subplots(2, 2, figsize=(15, 15), constrained_layout=True)
top_countries = borrower_count_by_country['country'].head(4)

for i, country in enumerate(top_countries):
    # Filter the DataFrame for the current country and calculate the top 10 regions
    top_regions = df[df['country'] == country]['cleaned_region'].value_counts().head(10).reset_index()
    top_regions.columns = ['cleaned_region', 'borrower_count']
    ax = axes[i//2, i%2]  # Determine the correct subplot position in the 2x2 grid
    sns.barplot(data=top_regions, x='borrower_count', y='cleaned_region', ax=ax, palette = 'Blues_d')
    ax.set_title(f'Top 10 Regions in {country}')
    ax.set_xlabel('Number of Borrowers')
    ax.set_ylabel('Region')

plt.show()



################# BY SECTOR

borrower_count_by_sector = df['sector_borrowers'].value_counts().reset_index()
borrower_count_by_sector.columns = ['sector', 'borrower_count']
plt.figure(figsize=(12, 20))
ax = sns.barplot(data=borrower_count_by_sector, x='borrower_count', y='sector', palette='Blues_d')
# Rotate y-axis labels for better readability
plt.xticks(rotation=90)
plt.title('Count of Borrowers by Sector', fontsize=16)
plt.xlabel('Number of Borrowers', fontsize=12)
plt.ylabel('Sector', fontsize=12)


# Calculate the count of borrowers based on 'country' and 'sector_borrowers'
pivot_df = df.groupby(['country', 'sector_borrowers']).size().unstack(fill_value=0)
pivot_df.to_excel('C:/Users/hp/Documents/KiVA/1. File Data/3. Data Temporary/country_sector.xlsx')

plt.figure(figsize=(12, 20))
sns.heatmap(pivot_df, annot=False, cmap='crest')
plt.title('Sector Intensity Across Countries (Heatmap)')
plt.ylabel('Country')
plt.xlabel('Sector')
plt.xticks(rotation=90)
plt.tight_layout()



################# BY SECTOR AND ACTIVITY

# Calculate activity counts within each sector
activity_counts = df.groupby('sector_borrowers')['activity'].value_counts().reset_index(name='counts')

# Find out how many unique sectors we have
sectors = activity_counts['sector_borrowers'].unique()
num_sectors = len(sectors)

# Set up the matplotlib figure - we'll have a subplot for each sector
subplot_cols = 3  
subplot_rows = np.ceil(num_sectors / subplot_cols).astype(int)
fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=(20, 5*subplot_rows), constrained_layout=True)

# Flatten the axes array for easy iteration if we have more than one row
if subplot_rows > 1:
    axes_flat = axes.flatten()
else:
    axes_flat = axes

# Plot each sector's activity counts in a separate subplot
for i, sector in enumerate(sectors):
    # Select the current sector's data
    sector_data = activity_counts[activity_counts['sector_borrowers'] == sector]
    # Use a barplot for each sector's activity count
    sns.barplot(data=sector_data, x='counts', y='activity', ax=axes_flat[i], palette='Blues_d')
    axes_flat[i].set_title(f"Activities in {sector}")

# If there are more subplots than sectors, remove the extra plots
if num_sectors % subplot_cols != 0:
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

# Show the plot
plt.show()


################# BY TYPE OF BORROWER AND GENDER

# Create separate DataFrames for 'group' and 'single' borrower types
df_group = df[df['borrower_type'] == 'group']
df_single = df[df['borrower_type'] == 'single']

# Aggregate the data by country
grouped_group = df_group.groupby('country').agg({'no_male_borrower': 'sum', 'no_female_borrower': 'sum'}).reset_index()
grouped_single = df_single.groupby('country').agg({'no_male_borrower': 'sum', 'no_female_borrower': 'sum'}).reset_index()

# Sort the DataFrames by 'country' alphabetically for plotting
grouped_group = grouped_group.sort_values('country', ascending=True)
grouped_single = grouped_single.sort_values('country', ascending=True)

# Plot for 'group' borrower type
plt.figure(figsize=(10, 15))
sns.barplot(data=grouped_group, x='no_male_borrower', y='country', color='navy', label='Male')
sns.barplot(data=grouped_group, x='no_female_borrower', y='country', color='cornflowerblue', label='Female')
plt.title('Group Borrowers by Country')
plt.xlabel('Number of Borrowers')
plt.ylabel('Country')
plt.legend(loc='lower right')
plt.show()

# Plot for 'single' borrower type
plt.figure(figsize=(10, 15))
sns.barplot(data=grouped_single, x='no_male_borrower', y='country', color='navy', label='Male')
sns.barplot(data=grouped_single, x='no_female_borrower', y='country', color='cornflowerblue', label='Female')
plt.title('Single Borrowers by Country')
plt.xlabel('Number of Borrowers')
plt.ylabel('Country')
plt.legend(loc='lower right')
plt.show()


# Count the number of occurrences for each sector within group and single borrower types
group_sector_counts = df_group['sector_borrowers'].value_counts().reset_index()
group_sector_counts.columns = ['sector', 'count']
single_sector_counts = df_single['sector_borrowers'].value_counts().reset_index()
single_sector_counts.columns = ['sector', 'count']

# Set up the matplotlib figure - we'll have two subplots side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8), constrained_layout=True)

# Plot for 'group' borrower type
sns.barplot(data=group_sector_counts, x='count', y='sector', ax=axes[0], palette='Blues_d')
axes[0].set_title('Group Borrowers by Sector')
axes[0].set_xlabel('Number of Borrowers')
axes[0].set_ylabel('Sector')

# Plot for 'single' borrower type
sns.barplot(data=single_sector_counts, x='count', y='sector', ax=axes[1], palette='Blues_d')
axes[1].set_title('Single Borrowers by Sector')
axes[1].set_xlabel('Number of Borrowers')
axes[1].set_ylabel('Sector')

# Show the plot
plt.show()




######## 2. Examine the distribution of loans 

################# BY COUNTRY

pd.options.display.float_format = '{:.0f}'.format
loan_country_summary = df.groupby('country').agg({'loan_amount': ['sum', 'mean', 'median'],
                                                  'funded_amount': ['sum', 'mean', 'median']
                                                 }).reset_index()

# Flatten the MultiIndex in columns
loan_country_summary.columns = ['_'.join(col).strip() for col in loan_country_summary.columns.values]

# Rename columns for clarity
loan_country_summary.rename(columns={'country': 'country',
                                     'loan_amount_sum': 'total_loan_amount',
                                     'loan_amount_mean': 'average_loan_amount',
                                     'loan_amount_median': 'median_loan_amount',
                                     'funded_amount_sum': 'total_funded_amount',
                                     'funded_amount_mean': 'average_funded_amount',
                                     'funded_amount_median': 'median_funded_amount'
                                     }, inplace=True)


totalamount_countries = loan_country_summary.sort_values(by='total_loan_amount', ascending=False)
# Reset the index of the DataFrame to align with the new sorted order
totalamount_countries = totalamount_countries.reset_index(drop=True)
# Initialize the matplotlib figure
plt.figure(figsize=(12, 20))
# Plot the total loan amount by country
barplot = sns.barplot(x='total_loan_amount', y='country_', data=totalamount_countries, palette='Blues_d')
plt.xlabel('Total Loan Amount')  
plt.ylabel('Country')            
plt.title('Total Loan Amount by Country')  
# Iterate over the bars and annotate with the mean value
for index, row in totalamount_countries.iterrows():
    # Set the x-coordinate to be at the end of the bar plus a small offset
    x_coord = row['total_loan_amount']
    barplot.text(x_coord, index, f"Mean: {row['average_loan_amount']:.2f}", 
                 color='black', ha="left", va='center', fontsize=9)

plt.show()


# Create a function to add annotations
def add_annotations(ax, series):
    mean = series.mean()
    median = series.median()
    std_dev = series.std()
    min_val = series.min()
    max_val = series.max()

    ax.text(0.98, 0.9, f'Mean: {mean:.2f}', transform=ax.transAxes, horizontalalignment='right')
    ax.text(0.98, 0.85, f'Median: {median:.2f}', transform=ax.transAxes, horizontalalignment='right')
    ax.text(0.98, 0.8, f'Std Dev: {std_dev:.2f}', transform=ax.transAxes, horizontalalignment='right')
    ax.text(0.98, 0.75, f'Min: {min_val:.2f}', transform=ax.transAxes, horizontalalignment='right')
    ax.text(0.98, 0.7, f'Max: {max_val:.2f}', transform=ax.transAxes, horizontalalignment='right')

# Create the subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))  
# Total Loan Amount
sns.boxplot(data=loan_country_summary['total_loan_amount'], ax=axs[0, 0])
axs[0, 0].set_title('Total Loan Amount')
add_annotations(axs[0, 0], loan_country_summary['total_loan_amount'])

# Average Loan Amount
sns.boxplot(data=loan_country_summary['average_loan_amount'], ax=axs[0, 1])
axs[0, 1].set_title('Average Loan Amount')
add_annotations(axs[0, 1], loan_country_summary['average_loan_amount'])

# Median Loan Amount
sns.boxplot(data=loan_country_summary['median_loan_amount'], ax=axs[0, 2])
axs[0, 2].set_title('Median Loan Amount')
add_annotations(axs[0, 2], loan_country_summary['median_loan_amount'])

# Histograms
sns.histplot(loan_country_summary['total_loan_amount'], bins=10, kde=True, ax=axs[1, 0])
sns.histplot(loan_country_summary['average_loan_amount'], bins=10, kde=True, ax=axs[1, 1])
sns.histplot(loan_country_summary['median_loan_amount'], bins=10, kde=True, ax=axs[1, 2])

plt.tight_layout()  # Adjusts the plots to fit well
plt.show()






################# BY COUNTRY AND REGION

loan_summary = df.groupby(['country', 'cleaned_region']).agg({'loan_amount': ['sum', 'mean', 'median'],
                                                              'funded_amount': ['sum', 'mean', 'median']
                                                              }).reset_index()

# Flatten the MultiIndex for the columns
loan_summary.columns = ['_'.join(col).strip() for col in loan_summary.columns.values]

# Fix the column names for 'country' and 'region' since they don't need aggregation
loan_summary.rename(columns={'country_': 'country', 'cleaned_region_': 'region'}, inplace=True)


# Determine the top 4 countries by total loan amount
top_countries = loan_summary.groupby('country')['loan_amount_sum'].sum().nlargest(4).index

# Initialize a DataFrame to hold the top regions for the top countries
top_regions_summary = pd.DataFrame()

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

# Iterate through the top 4 countries and their axes to create a subplot for each
for i, country in enumerate(top_countries):
    # Select the ith subplot
    ax = axes[i//2, i%2]
    # Get the top 10 regions for the current country by total loan amount
    top_regions = loan_summary[loan_summary['country'] == country].nlargest(10, 'loan_amount_sum')
    # Create the barplot for the current top country's regions
    sns.barplot(x='loan_amount_sum', y='region', data=top_regions, ax=ax, palette='Blues_d')
    # Set the title for the subplot to the current country's name
    ax.set_title(f'Top 10 Regions by Total Loan Amount in {country}')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=360)

# Adjust the layout of the subplots
plt.tight_layout()

# Show the plot
plt.show()





################# BY SECTOR AND ACTIVITIES

# Calculate total loan amount within each sector and activity
loan_amount_totals = df.groupby(['sector_borrowers', 'activity'])['loan_amount'].sum().reset_index(name='total_loan_amount')

# Find out how many unique sectors we have
sectors = loan_amount_totals['sector_borrowers'].unique()
num_sectors = len(sectors)

subplot_cols = 3 
subplot_rows = np.ceil(num_sectors / subplot_cols).astype(int)
fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=(20, 5*subplot_rows), constrained_layout=True)

# Flatten the axes array for easy iteration if we have more than one row
if subplot_rows > 1:
    axes_flat = axes.flatten()
else:
    axes_flat = [axes]  # Ensure axes_flat is always iterable

# Plot each sector's total loan amount in a separate subplot
for i, sector in enumerate(sectors):
    # Select the current sector's data
    sector_data = loan_amount_totals[loan_amount_totals['sector_borrowers'] == sector]
    # Use a barplot for each sector's total loan amount
    sns.barplot(data=sector_data, x='total_loan_amount', y='activity', ax=axes_flat[i], palette='Blues_d')
    axes_flat[i].set_title(f"Loan Amounts in {sector}")

# If there are more subplots than sectors, remove the extra plots
if num_sectors % subplot_cols != 0:
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

# Show the plot
plt.show()


loan_amount_mean = df.groupby(['sector_borrowers', 'activity'])['loan_amount'].mean().reset_index(name='mean_loan_amount')
# Find out how many unique sectors we have
sectors = loan_amount_mean['sector_borrowers'].unique()
num_sectors = len(sectors)

subplot_cols = 3
subplot_rows = np.ceil(num_sectors / subplot_cols).astype(int)
fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=(20, 5*subplot_rows), constrained_layout=True)

# Flatten the axes array for easy iteration if we have more than one row
if subplot_rows > 1:
    axes_flat = axes.flatten()
else:
    axes_flat = [axes]  # Ensure axes_flat is always iterable

# Plot each sector's total loan amount in a separate subplot
for i, sector in enumerate(sectors):
    # Select the current sector's data
    sector_data = loan_amount_mean[loan_amount_mean['sector_borrowers'] == sector]
    # Use a barplot for each sector's total loan amount
    sns.barplot(data=sector_data, x='mean_loan_amount', y='activity', ax=axes_flat[i], palette='Blues_d')
    axes_flat[i].set_title(f"Loan Amounts in {sector}")

# If there are more subplots than sectors, remove the extra plots
if num_sectors % subplot_cols != 0:
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

# Show the plot
plt.show()



######## 3. Examine the distribution of loan repayment


# Analyze the distribution of repayment intervals
repayment_intervals_distribution = df['repayment_interval'].value_counts().reset_index()
repayment_intervals_distribution.columns = ['Repayment Interval', 'Frequency']

# Analyze the average term in months for each repayment interval
average_terms_by_interval = df.groupby('repayment_interval')['term_in_months'].mean().reset_index()
average_terms_by_interval.columns = ['Repayment Interval', 'Average Term']

# Analyze the repayment interval by country
repayment_by_country = df.groupby(['country', 'repayment_interval']).size().unstack().fillna(0)

# Plotting the distribution of repayment intervals
plt.figure(figsize=(8, 4))
sns.barplot(data=repayment_intervals_distribution, x='Repayment Interval', y='Frequency', palette='viridis')
plt.title('Distribution of Repayment Intervals')
plt.ylabel('Frequency of Repayment Interval')
plt.xlabel('Repayment Interval')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting the average term by repayment interval
plt.figure(figsize=(8, 4))
sns.barplot(data=average_terms_by_interval, x='Repayment Interval', y='Average Term', palette='viridis')
plt.title('Average Repayment Term by Interval')
plt.ylabel('Average Term (Months)')
plt.xlabel('Repayment Interval')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting the repayment interval distribution by country
plt.figure(figsize=(12, 20))
sns.heatmap(repayment_by_country, cmap='viridis', linewidths=.5)
plt.title('Repayment Interval Distribution by Country')
plt.ylabel('Country')
plt.xlabel('Repayment Interval')
plt.tight_layout()
plt.show()



