---
layout: post
title: DA - Hotel Reservation Prediction
date: 2023 Oct 15
categories: [DA, coding]
---

This post is a record of a final project of Data Analyst course.

# Table of contents
I. [Data processing](#Data_processing)
1. [About the data](#about_the_data)
2. [Data overview](#Data_overview)
3. [Data cleaning](#Data_cleaning)
   1. [**Duplicated**](#Duplicated)
   2. [**Null and Undefined**](#null_and_undefined_data)
   3. [**Outliers**](#Outliers)
   4. [**Merge columns**](#merge_columns)
   5. [**Create data for illustrate and calculate**](#create_data_for_illustrate_and_calculate)
   6. [**Drop unnescessary columns**](#Drop_unnescessary_columns)
   7. [**Brief EDA**](#brief_eda)
4. [**Machine Learning Process**](#machine_learning_process)
## Data processing <a name="Data_processing"></a>

### About the data <a name="about_the_data"></a>

Source: Kaggle - [Hotel Booking](https://www.kaggle.com/datasets/mathsian/hotel-bookings/data)

The dataset have 3 file:

- The original, raw data set is given in the hotel_bookings.csv file. 

- bookings_2023.csv is a slightly simplified version with 23 features. 

- The bookings.csv file has been further reduced to 10 columns and pre-processed to aid analysis.

Since this is hotel real data, all data elements pertaining hotel or costumer identification were deleted.

> The three datasets are essentially variations of one another, 'hotel booking' has been chosen as the primary dataset due to its raw data.

### Data overview <a name="Data_overview"></a>
Data has 119.390 rows and 32 columns. <br>

| **Number** | **Customers Source** | **Customer Character**         | **Reservation**             | **Recorded time**         |
|------------|----------------------|--------------------------------|-----------------------------|---------------------------|
| 1          | hotel                | adults                         | is_canceled                 | arrival_date_year         |
| 2          | market_segment       | children                       | lead_time                   | arrival_date_month        |
| 3          | distribution_channel | babies                         | days_in_waiting_list        | arrival_date_week_number  |
| 4          | agent                | country                        | stays_in_weekend_nights     | arrival_date_day_of_month |
| 5          | company              | is_repeated_guest              | stays_in_week_nights        | reservation_status        |
| 6          | customer_type        | previous_cancellations         | reserved_room_type          | reservation_status_date   |
| 7          |                      | previous_bookings_not_canceled | assigned_room_type          |                           |
| 8          |                      |                                | booking_changes             |                           |
| 9          |                      |                                | deposit_type                |                           |
| 10         |                      |                                | adr                         |                           |
| 11         |                      |                                | required_car_parking_spaces |                           |
| 12         |                      |                                | meal                        |                           |
| 13         |                      |                                | total_of_special_requests   |                           |

### Data cleaning <a name="Data_cleaning"></a>

#### **Duplicated** <a name="Duplicated"></a>
Input
```python
# checking total duplicated
hotel_booking.duplicated().sum()
```
```
31994
```

> As authors claims that:<br>
> - "Each observation represents a hotel booking." <br>
> and <br>
> - "Since this is hotel real data, all data elements pertaining hotel or costumer identification were deleted."


> Cause of 'Each observation represents a hotel booking' therefore data record is separate customers from separate hotels and coincidencelly have the same data. the action here is choose to **keep all the duplications**.

#### **Null and Undefined data** <a name="Null and Undefined"></a>

``` python 
# checking null
hotel_booking.isnull().sum().sort_values(ascending=False)
```
```
company                           112593
agent                              16340
country                              488
children                               4
```
```python
# check columns have that have 'Undefined' values
columns_with_value = (hotel_booking == 'Undefined')

columns_with_value.sum().sort_values(ascending=False)
```
```
meal                              1169
distribution_channel                 5
market_segment                       2
```

in short:
|*columns*|*problem*|
|---|---|
|agent| has nan|
|company| has nan|
|children| has nan|
|country| has nan|
|meal| has Undefined|
|market_segment| has Undefined|
|distribution_channel| has Undefined|

> author claim that:<br>
> In some categorical variables like Agent or Company, “NULL” is presented as one of the categories.<br>
> This should not be considered a missing value, but rather as “not applicable”.<br>
> For example, if a booking “Agent” is defined as “NULL” it means that the booking did not came from a travel agent.

> As said above:<br>
> for the 'agent' and 'company' columns can change 'nan values' into:
> - 0 as customer not come from agent/company
> - 1 as customer come from agent/company

```python
# create def to change data value 
def convert_num(i):
    if i > 0:
        return 1
    return 0

# apply def 
hotel_booking['agent_encode'] = hotel_booking.agent.apply(convert_num)
hotel_booking['company_encode'] = hotel_booking.company.apply(convert_num)

# 'children' columns has 4 nan values replace by median
hotel_booking['children'] = hotel_booking['children'].fillna(hotel_booking['children'].median())

# 'country' columns has 488 nulls values
# not using to run model but can use to illustrate customer character 
# leave as it is
hotel_booking.country.isnull().sum()
hotel_booking.country.value_counts()
```
for the 'Undefined' values:

```python
# as dictionary states that:
# - Undefined/SC : no meal package
# - BB : Bed & Breakfast
# - HB : Half board (breakfast and one other meal – usually dinner)
# - FB : Full board (breakfast, lunch and dinner)

# as said: encode 'meal' as Undefined/SC: 0 and others: 1
def convert_meal(j):
    if j == 'Undefined' or j == 'SC':
        return 0
    return 1

hotel_booking['meal_encode'] = hotel_booking.meal.apply(convert_meal)

# 'market_segment' and 'distribution_channel' have less than 6 'undefined' values 
# leave as it is
print(hotel_booking.market_segment.value_counts())
print('-----------')
print(hotel_booking.distribution_channel.value_counts())
```

#### **Outliers** <a name="Outliers"></a>

![Checking Outliers](/Phongs-Adventure/assets/material/hotel_reservation_pic/outliers.png)

> Showed that columns 18 which is 'adr' has 1 outliers -> choose to delete that row

```python
# drop outliers 
hotel_booking = hotel_booking[hotel_booking['adr'] < 1000] #.reset_index(drop=True, inplace=True)
```

#### **Merge columns** <a name="merge_columns"></a>

Merge columns that have similar meaning

```python
# create columns 'family_size' for illustation
hotel_booking['family_size'] = hotel_booking['adults'] + hotel_booking['children'] + hotel_booking['babies']

# booking_requests = booking_changes + required_car_parking_spaces + total_of_special_requests : cause of this is all request in booking process.
hotel_booking['booking_requests'] = hotel_booking.booking_changes + hotel_booking.required_car_parking_spaces + hotel_booking.total_of_special_requests

# stay_in_days = stays_in_weekend_nights + stays_in_week_nights
hotel_booking['stay_in_days'] = hotel_booking.stays_in_weekend_nights + hotel_booking.stays_in_week_nights

```

#### **Create data for illustrate and calculate** <a name="create_data_for_illustrate_and_calculate"></a>

```python 

# create columns 'source' for illustrate customer sources 
hotel_booking['source'] = np.where((hotel_booking['agent'] > 0) & (hotel_booking['company'] > 0), 'both',
                          np.where(hotel_booking['agent'] > 0, 'agent',
                          np.where(hotel_booking['company'] > 0, 'comapny', 
                          'not applicable')))

# create columns 'meal_request' 
meal_dictionary = {'BB': 'Meal', 'FB': 'Meal', 'HB': 'Meal', 'SC': 'No meal', 'Undefined': 'No meal'}
hotel_booking['meal_request'] = hotel_booking['meal'].map(meal_dictionary)

# create columns 'repeated_guest'
repeated_guest_dictionary = {0: 'New guest', 1: 'Old guest'}
hotel_booking['repeated_guest'] = hotel_booking['is_repeated_guest'].map(repeated_guest_dictionary)

---

# create columns 'cancelation_status'
cancel_dictionary = {0: 'Check_in', 1: 'Canceled'}
hotel_booking['cancelation_status'] = hotel_booking['is_canceled'].map(cancel_dictionary)

--- 

# create columns 'total_booking'
hotel_booking['total_booking'] = hotel_booking.previous_cancellations + hotel_booking.previous_bookings_not_canceled + 1

---

# create columns 'total_cancelation'
hotel_booking['total_cancelation'] = hotel_booking['previous_cancellations'] + hotel_booking['is_canceled']

---

# create 'cancelation_rate'
hotel_booking['cancelation_rate'] = (hotel_booking['total_cancelation'] / hotel_booking['total_booking'] )* 100

```

#### **Drop unnescessary columns** <a name="Drop_unnescessary_columns"></a>

drop columns below cause of:
- **reservation_status**: exactly as 'is_canceled' columns
- **reservation_status_date**: not see the use
- **arrival_date_year**: this analysis focus on customer behaviours in months and days
- **arrival_date_week_number**: this analysis focus on customer behaviours in months and days
- **company**: replace by 'company_encode'
- **agent**: replace by 'agent_encode'
- **market_segment**: similar with distribution_channel
- **adults**, **children**, **babies**: replace by 'family size'
- **stays_in_weekend_nights**, **stays_in_week_nights**: replace by 'stay_in_days'

```python
hotel_booking = hotel_booking.drop(['reservation_status', 'reservation_status_date', 'arrival_date_year', 'arrival_date_week_number', 
                                    'company', 'agent',
                                    'market_segment',
                                    'adults', 'children', 'babies',
                                    'booking_changes', 'required_car_parking_spaces', 'total_of_special_requests',
                                    'stays_in_weekend_nights', 'stays_in_week_nights'], axis=1)
```
#### **Brief EDA** <a name="brief_eda"></a>

For better EDA please check below.
```python
# basic graphs
hotel_booking.hist(figsize= (20,20))
plt.show()
```
![brief_eda](Phongs-Adventure/assets/material/hotel_reservation_pic/brief_eda.png)

## Machine learning process <a name="machine_learning_process"></a>
### Preparing <a name="Outliers"></a>
#### **Encode** <a name="Outliers"></a>

First, separate the numeric and categorical columns.

```python 
# seperate numeric columns
hb_numeric = hotel_booking.select_dtypes(include='number')
```

Then, encode the categorical columns, mostly using 'onehot'

```python
# encode the categories columns using onthot
hotel_dictionary = {'Resort Hotel': 1, 'City Hotel': 0}
hotel_booking['hotel_encode'] = hotel_booking.hotel.map(hotel_dictionary)

onehot_month            = pd.get_dummies(hotel_booking['arrival_date_month'], prefix = 'month').astype(int)
onehot_market           = pd.get_dummies(hotel_booking['distribution_channel'], prefix = 'market').astype(int)
onehot_reserved_room    = pd.get_dummies(hotel_booking['reserved_room_type'], prefix = 'reserved_room').astype(int)
onehot_assigned_room    = pd.get_dummies(hotel_booking['assigned_room_type'], prefix = 'assigned_room').astype(int)
onehot_deposit_type     = pd.get_dummies(hotel_booking['deposit_type'], prefix = 'deposit_type').astype(int)
onehot_customer_type    = pd.get_dummies(hotel_booking['customer_type'], prefix = 'customer_type').astype(int)
```
Finish by create new dataset.
```python
# create new dataset to concate 2 datasets

hb_encode = pd.concat([
    hb_numeric, # the dataset
    onehot_month,
    onehot_market,
    onehot_reserved_room,
    onehot_assigned_room,
    onehot_deposit_type,
    onehot_customer_type
], axis = 1)
```

#### **Defining X, y** <a name="Outliers"></a>
First, to calculate correlation and draw a chart for a better view.

```python
# calculate correlation of others columns to 'is_canceled'
hb_correlation = hb_encode.corr(numeric_only=True)[['is_canceled']]
hb_correlation.drop(index='is_canceled', inplace=True)
hb_correlation = hb_correlation.sort_values(by='is_canceled', ascending=False)

# plot chart
fig, ax = plt.subplots(figsize=(20, 20))

# draw a seperate linne at x = 0
ax.axvline(x=0, color='gray', linestyle='--')

# bar chart
colors = np.where(hb_correlation['is_canceled'] > 0, 'lightblue', 'lightcoral')
bars = ax.barh(hb_correlation.index, hb_correlation['is_canceled'], color=colors)

# rearrange value lable of bar chart
for bar in bars:
    width = bar.get_width()
    label_x_pos = width + 0.01 if width >= 0 else width - 0.05 # if value > 0 set in the right else set in the left
    ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center', fontsize=13)

ax.set_xlabel('Correlation')
ax.set_title('Correlation with Cancelation', size=16)
plt.grid()
plt.tight_layout()
plt.show()
```
![correlation](Phongs-Adventure/assets/material/hotel_reservation_pic/correlation.png)

As showed in the graph:

> *'cancelation_rate'* and *'total_cancelation'* have high correlation with 'is_canceled' cause 'is_canceled' constitutes for their formular. <br>
> --> not use this columns in model to avoid false calculation.

Other than 2 above columns, overal correlation numbers are decent, no more than |0,5|. Major of them are below |0,1|. Therefore, those columns have correlation higher than 0,1 and lower than -0.1 are chosen.

```python
# sort columns have value outside of this range (-0.1, 0.1)
sorted(hb_correlation[ ~hb_correlation['is_canceled'].between(-0.1, 0.1)].index)
```
```
['agent_encode',
 'assigned_room_A',
 'assigned_room_D',
 'booking_requests',
 'cancelation_rate',
 'customer_type_Transient',
 'customer_type_Transient-Party',
 'deposit_type_No Deposit',
 'deposit_type_Non Refund',
 'lead_time',
 'market_Direct',
 'market_TA/TO',
 'previous_cancellations',
 'total_cancelation']
```

Checking correlation of columns to each one another.

```python
# create new dataset
hb_encode_corr =  hb_encode[['is_canceled',
                             'agent_encode',
                            'assigned_room_A',
                            'assigned_room_D',
                            'booking_requests',
                            'customer_type_Transient',
                            'customer_type_Transient-Party',
                            'deposit_type_No Deposit',
                            'deposit_type_Non Refund',
                            'lead_time',
                            'market_Direct',
                            'market_TA/TO',
                            'previous_cancellations'
                            ]]

# draw correlation heatmap

# create dataset correlation and only select the correlation out of the range (-0.1, 0.1)
def high_corr(value):
  if -0.2 < value < 0.2:
    return 0
  return value
hb_high_correlation = hb_encode_corr.corr(numeric_only=True)
hb_high_correlation = hb_high_correlation.applymap(high_corr)

# draw heatmap of correlation
plt.figure(figsize=(8,8))
sns.heatmap(hb_high_correlation, annot=True, cmap='RdYlGn', vmin=-1, vmax=1)
```

![correlation_df](Phongs-Adventure/assets/material/hotel_reservation_pic/correlation.png)

There are couples of columns that have high correlation  with each other:
- customer_type_transient and customer_type_transient_party
- deposit_type_No_deposit and deposit_type_No_refund
- market_Direct and market_TA/TO

> The choice is to delete 1 of each columns to prevent Multicollinearity (Đa cộng tuyến)

This is the chart after drop some of the above columns: 

![correlation_clean](Phongs-Adventure/assets/material/hotel_reservation_pic/correlation_clean.png)

> The X here is: <br>
> 'agent_encode', 'assigned_room_A', 'assigned_room_D', 'booking_requests', 'customer_type_Transient', 'deposit_type_No Deposit', 'lead_time', 'market_TA/TO'.

>The y here is: <br>
> 'is_canceled'

#### **Data balancing** <a name="Outliers"></a>

```python
hb_encode_corr['is_canceled'].hist()
```
![balancing](Phongs-Adventure/assets/material/hotel_reservation_pic/balancing.png)

> As showed that, the 0 values is double the 1 values, the choice here is to undersampling.

```python
# dataset not cancel == 0
hbc_0 = hb_encode_corr[hb_encode_corr.is_canceled == 0]

# dataset is canceled == 1
hbc_1 = hb_encode_corr[hb_encode_corr.is_canceled == 1]

# dataset size
hbc_0.shape, hbc_1.shape
# random choose data from hbc_0
hbc_0_resapled = hbc_0.sample(44223, random_state=1)

# new dataset
hbc_0_resapled.shape
# connect hb_0 and hbc_1
hb_balance = pd.concat([hbc_0_resapled,hbc_1]) 

# y after rebalance
hb_balance['is_canceled'].value_counts()
```
```
is_canceled
0    44223
1    44223
Name: count, dtype: int64
```

Reset index for further use in normalization.
```python
hb_balance.reset_index(drop=True, inplace=True)
```

#### **Normalization: min-max scaler** <a name="Outliers"></a>

```python
# Scale X2
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(hb_balance.iloc[:, 1:11].values)

# create new dataframe
hb = pd.DataFrame(data = X_scaled, columns = hb_balance.iloc[:, 1:11].columns)

# add y2 column
hb['is_canceled'] = hb_balance['is_canceled']
```
#### **Split train-test dataset** <a name="Outliers"></a>

```python
# chose X, y

y = hb['is_canceled'].values
X = hb[['agent_encode', 'assigned_room_A', 'assigned_room_D', 'booking_requests', 'customer_type_Transient', 'deposit_type_No Deposit', 'lead_time', 'market_TA/TO']].values

# seting X_set for use
X_set = ['agent_encode', 'assigned_room_A', 'assigned_room_D', 'booking_requests', 'customer_type_Transient', 'deposit_type_No Deposit', 'lead_time', 'market_TA/TO']

# run split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

### Models <a name="Outliers"></a>

#### Logistic <a name="Outliers"></a>