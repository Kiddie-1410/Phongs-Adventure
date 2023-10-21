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

*The three datasets are essentially variations of one another, 'hotel booking' has been chosen as the primary dataset due to its raw data.*

### Data overview
1. Data has 119.390 rows and 32 columns. <br>

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

Input
```python
# checking total duplicated
hotel_booking.duplicated().sum()
```
```
31994
```

As authors claims that:<br>
"Each observation represents a hotel booking."<br>
"Since this is hotel real data, all data elements pertaining hotel or costumer identification were deleted."<br>

