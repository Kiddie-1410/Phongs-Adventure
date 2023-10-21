---
layout: post
title: DA - Hotel Reservation Prediction
date: 2023 Oct 15
categories: [DA, coding]
---

This post is a record of a final project of Data Analyst course.

## Data processing

### About the data

Source: Kaggle - [Hotel Booking](https://www.kaggle.com/datasets/mathsian/hotel-bookings/data)

The dataset have 3 file:

- The original, raw data set is given in the hotel_bookings.csv file. 

- bookings_2023.csv is a slightly simplified version with 23 features. 

- The bookings.csv file has been further reduced to 10 columns and pre-processed to aid analysis.

Since this is hotel real data, all data elements pertaining hotel or costumer identification were deleted.

> The three datasets are essentially variations of one another, 'hotel booking' has been chosen as the primary dataset due to its raw data.

### Data overview
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

### Data cleaning

#### Duplicated
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

#### Null and Undefined data

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

#### Outliers

![Checking Outliers](D:\Github\Phongs-Adventure\assets\material\hotel_reservation_pic\outliers.png)

Showed that columns 18 which is 'adr' has 1 outliers -> choose to delete that row

```python
# drop outliers 
hotel_booking = hotel_booking[hotel_booking['adr'] < 1000] #.reset_index(drop=True, inplace=True)
```

#### Merge columns

Merge columns that have similar meaning

```python
# create columns 'family_size' for illustation
hotel_booking['family_size'] = hotel_booking['adults'] + hotel_booking['children'] + hotel_booking['babies']

# booking_requests = booking_changes + required_car_parking_spaces + total_of_special_requests : cause of this is all request in booking process.
hotel_booking['booking_requests'] = hotel_booking.booking_changes + hotel_booking.required_car_parking_spaces + hotel_booking.total_of_special_requests

hotel_booking['stay_in_days'] = hotel_booking.stays_in_weekend_nights + hotel_booking.stays_in_week_nights

```

