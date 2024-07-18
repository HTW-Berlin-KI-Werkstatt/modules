---
title: "Visualizing Relationships"
layout: single
author_profile: true
author: Christina Kratsch
lecture_name: "Data Science"
lecture_desc: "Grundlegende Methoden für die Exploration und das Management von Daten."
licence: "CC-BY"
licence_desc: 2024 | HTW Berlin 
classes: wide
---

## Introduction

Python provides convenient support for handling time and dates. It is highly advisable to make use of these onboard functions when performing time-related data analysis. This module intends to explain the most relevant functions.

The `time` module encapsulates a range of functions dealing with the display and active management of time during programm execution:

| Function | Description |
| -- | -- |
| `time()` |Returns the time in floating point number in seconds                   |
|`ctime()`	|   Returns the current date and time|
|`sleep()`	|   Stops execution of a thread for the given duration  |     
|`localtime()` |  	   Returns the date and time in time.struct_time format      |
|`gmtime()`	|   Returns time.struct_time in UTC format|
|`mktime()`	 |  Returns the seconds passed since epochs are output|
|`asctime()	` |  Returns a string representing the same|


Python also has an in-built module named `DateTime` to deal with dates and times in numerous ways. In this module, we are going to see basic `DateTime` operations in Python.

There are **six main object classes** with their respective components in the datetime module mentioned below:

```Python
datetime.date
datetime.time
datetime.datetime
datetime.timedelta
datetime.tzinfo
datetime.timezone
```


## datetime.date():

We can generate date objects from the date class. A date object represents a date having a year, month, and day.

You can use `strftime` to print day, month, and year in various formats: 

```Python  
from datetime import date
 
# You can create a date object containing 
# the current date 
# by using a classmethod named today()
current = date.today() 
 
# print current year, month, and year individually
print("Current Day is :", current.day)
print("Current Month is :", current.month)
print("Current Year is :", current.year)
 
# strftime() creates string representing date in 
# various formats
print("\n")
print("Let's print date, month and year in different-different ways")
format1 = current.strftime("%m/%d/%y")
 
# prints in month/date/year format
print("format1 =", format1)
     
format2 =  current.strftime("%b-%d-%Y")
# prints in month(abbreviation)-date-year format
print("format2 =", format2)
 
format3 = current.strftime("%d/%m/%Y")
 
# prints in date/month/year format
print("format3 =", format3)
     
format4 =  current.strftime("%B %d, %Y")
 
# prints in month(words) date, year format
print("format4 =", format4)
```

## datetime.time():

A time object generated from the time class represents the local time, components: are
* hour
* minute
* second
* microsecond
* tzinfo

Code example:

```Python
from datetime import time
 
# time() takes hour, minutes, second,
# microsecond respectively in order 
# if no parameter is passed in time() by default
# it takes 0 
defaultTime = time()
 
print("default_hour =", defaultTime.hour)
print("default_minute =", defaultTime.minute)
print("default_second =", defaultTime.second)
print("default_microsecond =", defaultTime.microsecond)
 
# passing parameter in different-different ways
# hour, minute and second respectively is a default
# order
time1= time(10, 5, 25)
print("time_1 =", time1)
 
# assigning hour, minute and second to respective
# variables
time2= time(hour = 10, minute = 5, second = 25)
print("time_2 =", time2)
 
# assigning hour, minute, second and microsecond to
# respective variables
time3= time(hour=10, minute= 5, second=25, microsecond=55)
print("time_3 =", time3)
```

## datetime.datetime():

The `datetime.datetime()` module shows the combination of a date and a time. 


Again, using  `strftime()` allows printing different ways:

* strftime(“%d”) gives current day
* strftime(“%m”) gives current month
* strftime(“%Y”) gives current year
* strftime(“%H:%M:%S”) gives current time in an hour, minute, and second format
* strftime(“%m/%d/%Y, %H:%M:%S”) gives date and time together

```Python   
from datetime import datetime
 
# now() gives current date and time
current = datetime.now()
 
# print combinedly
print(current)
print("\n")
print("print each term individually")
 
day = current.strftime("%d")
 
# print day
print("day:", day)
 
month = current.strftime("%m")
 
# print month
print("month:", month)
 
year = current.strftime("%Y")
 
# print year
print("year:", year)
 
time = current.strftime("%H:%M:%S")
 
# time in hour, minute and second
print("time:", time)
 
print("\n")
print("printing date and time together")
date_time = current.strftime("%m/%d/%Y, %H:%M:%S")
print("date and time:", date_time)
print("\n")
 
# fetching details from timestamp
timestamp = 1615797322
date_time = datetime.fromtimestamp(timestamp)
 
# %c, %x and %X are used for locale's proper date and time representation
time_1 = date_time.strftime("%c")
print("first_output:", time_1)
 
time_2 = date_time.strftime("%x")
print("second_output:", time_2)
 
time_3 = date_time.strftime("%X")
print("third_output:", time_3)
 
print("\n")
 
# assigning each term manually
manual = datetime(2021, 3, 28, 23, 55, 59, 342380)
print("year =", manual.year)
print("month =", manual.month)
print("hour =", manual.hour)
print("minute =", manual.minute)
print("timestamp =", manual.timestamp())
```

## datetime.timedelta():
A duration that expresses the difference between two date, time, or datetime instances to microsecond resolution.

Here we implemented some basic functions and printed past and future days. Also, we will print some other attributes of timedelta max, min, and resolution that show maximum days and time, minimum date and time, and the smallest possible difference between non-equal timedelta objects respectively. Here we will also apply some arithmetic operations on two different dates and times.

```Python
from datetime import timedelta, datetime
 
present_date_with_time = datetime.now() 
 
print("Present Date :", present_date_with_time)
 
# coming date after 10 days
ten_days_after= present_date_with_time + timedelta(days = 10)
print('Date after 10 days :',ten_days_after)
 
# date before 10 days
ten_days_before= present_date_with_time - timedelta(days = 10)
print('Date before 10 days :',ten_days_before)
 
# date before one year ago
one_year_before_today= present_date_with_time + timedelta(days = 365)
print('One year before present Date :', one_year_before_today)
 
#date before one year ago
one_year_after_today= present_date_with_time - timedelta(days = 365)
print('One year before present Date :', one_year_after_today)
 
print("\n")
print("print some other attributes of timedelta\n")
 
# maximum days and time
print("Max : ",timedelta.max)
 
# minimum days and time
print("Min : ",timedelta.min)
 
# The smallest possible difference between non-equal
# timedelta objects, timedelta(microseconds=1)
print("Resolution: ",timedelta.resolution)
 
print('Total number of seconds in an year :', 
      timedelta(days = 365).total_seconds())
 
print("\nApply some operations on timedelta function\n")
time_after_one_min = present_date_with_time + timedelta(seconds=10) * 6
print('Time after one minute :', time_after_one_min)
 
print('Timedelta absolute value :', abs(timedelta(days = +20)))
 
print('Timedelta string representation :', str(timedelta(days = 5,
                       seconds = 40, hours = 20, milliseconds = 355)))
 
print('Timedelta object representation :', repr(timedelta(days = 5, 
                       seconds = 40, hours = 20, milliseconds = 355)))
```


## datetime.tzinfo()

`datetime.tzinfo()` is an abstract base class for time zone information objects used by the datetime and time classes to provide a customizable notion of time adjustment. 

* `utcoffset(self, dt)` returns the offset of the datetime instance passed as an argument
* `dst(self, dt)` dst stands for Daylight Saving Time. dst denotes advancing the clock 1 hour in summer so that darkness falls later according to the clock.  It is set to on or off. 
* `tzname(self, dt)` returns a Python String object. It is used to find the time zone name of the datetime object passed.
* `fromutc(self, dt)` returns the equivalent local time and takes up the date and time of the object in UTC. It is mostly used to adjust the date and time. It is called from default datetime.astimezone() implementation. The dt.tzinfo will be passed as self, dst date and time data will be returned as an equivalent local time.


```python
from datetime import datetime, timedelta
from pytz import timezone
import pytz

time_zone = timezone('Asia/Calcutta')

normal = datetime(2021, 3, 16)
ambiguous = datetime(2021, 4, 16, 23, 30)

# is_dst parameter is ignored for most of the
# timstamps.It is only used during DST
# transition ambiguous periods to resolve that
# ambiguity
print("Operations on normal datetime")
print(time_zone.utcoffset(normal, is_dst=True))
print(time_zone.dst(normal, is_dst=True))
print(time_zone.tzname(normal, is_dst=True))

# put is_dst=False
print(time_zone.utcoffset(normal, is_dst=False))
print(time_zone.dst(normal, is_dst=False))
print(time_zone.tzname(normal, is_dst=False))

print("\n")
print("Operations on ambiguous datetime")
print(time_zone.utcoffset(ambiguous, is_dst=True))
print(time_zone.dst(ambiguous, is_dst=True))
print(time_zone.tzname(ambiguous, is_dst=True))

# is_dst=False
print(time_zone.utcoffset(ambiguous, is_dst=False))
print(time_zone.dst(ambiguous, is_dst=False))
print(time_zone.tzname(ambiguous, is_dst=False))
```

## datetime.timezone()

`datetime.timezone()` is a class that implements the tzinfo abstract base class as a fixed offset from the UTC.

```python   
from datetime import datetime, timedelta
from pytz import timezone
import pytz
 
utc = pytz.utc
print(utc.zone)
 
india = timezone('Asia/Calcutta')
print(india.zone)
 
eastern = timezone('US/Eastern')
print(eastern.zone)
 
time_format = '%Y-%m-%d %H:%M:%S %Z%z'
 
# localize() is used to localize
# datetime with no timezone information
loc_dt = india.localize(datetime(2021, 3, 16, 6, 0, 0))
loc_dt = india.localize(datetime(2021, 3, 16, 6, 0, 0))
print(loc_dt.strftime(time_format))
 
# another way of building a localized time is by converting
# an existing localized time 
# using the standard astimezone() method
eastern_dt = loc_dt.astimezone(eastern)
print(eastern_dt.strftime(time_format))
 
print(datetime(2021, 3, 16, 12, 0, 0, tzinfo=pytz.utc).strftime(time_format))
 
# 10 minutes before
before_dt = loc_dt - timedelta(minutes=10)
print(before_dt.strftime(time_format))
print(india.normalize(before_dt).strftime(time_format))
 
# 20 mins later
after_dt = india.normalize(before_dt + timedelta(minutes=20))
print(after_dt.strftime(time_format))
```
