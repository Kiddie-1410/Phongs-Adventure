---
layout: post
title: DA - Hotel Reservation Report
date: 2023 Oct 15
categories: [DA, coding]
---

This post is a record of the 3rd final project presentation of Data Analyst course.

# Table of contents
O. [Project materials](#Project_materials)

I. [Report](#Report)
1. [Overview](#Overview)
2. [Data Handling](#Data_handling)
3. [EDA](#EDA)

II. [Model running](#Model_running)
4. [Prepare for model](#Prepare_for_model)
5. [Models running](#Model_running)
6. [Results and Evaluation](#Results_and_evaluation)

## O. Project material <a name="Project_materials"></a>
This project has been process by 3 tools:
- Data processing and running model: [Python](/Phongs-Adventure/_posts/2023-10-15-hotel-reservation-prediction-code.markdown) (VScode)
- Data illustration: [PBI](https://github.com/Kiddie-1410/hotel-reservation-prediction/blob/main/hotel_illus.pbix) 
- Presentation: [Google slides](https://docs.google.com/presentation/d/1iKUgwfeVDMJJAeI2Pq8H9JvfraGytlFfb3YjiQh0K0Q/edit?usp=sharing)

For better understanding the project please check others file further information.

## I. Report <a name="Report"></a>

### 1. Overview <a name="Overview"></a>

- Dataset
  - Name of dataset: Hotel bookings in Portugal
  - Source: Kaggle
  - [Link](https://www.kaggle.com/datasets/mathsian/hotel-bookings/data)
  - [Original](https://www.researchgate.net/publication/329286343_Hotel_booking_demand_datasets)

- Objectives
  - Find out customer reservation behaviors via EDA
  - Conduct algorithms to predict/ classify customers reservation.

- About the data

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

Sample:

![data_sample](/Phongs-Adventure/assets/material/hotel_reservation_pic/Report/data_sample.png)

## 2. Data handling <a name="Data_handling"></a>

#### **Duplicated** <a name="Duplicated"></a>
Input
```python
# checking total duplicated
hotel_booking.duplicated().sum()
```
```
31994
```



![Channel_distribution]](image.png)
## II. Model running <a name="Model_running"></a>