---
title: "Things to consider before planning your trip to Boston."
date: 2021-08-11
last_modified_at: 2021-08-11 08:48:05 -0400
excerpt: "Data analysis of Airbnb listings in Boston using the CRISP-DM process."
categories:
  - Data Analysis
tags:
  - Visualisation
  - Data Analysis
  - Crisp Dm
header:
  teaser: /assets/images/Things-to-consider-before-planning-your-trip-to-Boston/header.jpg
  overlay_image: /assets/images/Things-to-consider-before-planning-your-trip-to-Boston/header.jpg
  overlay_filter: 0.5
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
  actions:
    - label: "Code"
      url: "https://github.com/kumarsantosh04/data-science-blog-post"
---


# Introduction
When it comes to travel Airbnb is one of the popular choices for a comfortable stay. Since 2008, guests and hosts have used Airbnb to travel in a more unique, personalized way. Airbnb or Air Bed and Breakfast was started back in 2007 by roommates Joe Gebbia and Brian Chesky. It started as a website AirBedandBreakfast.com where guests were charged $80 per night to sleep on the air mattresses. Since then it has evolved to more diverse offerings like private rooms, houses, apartments, and more.
{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/world-data.png" alt="Vaccinated world population source: ourworldindata.org" caption="Vaccinated world population source: ourworldindata.org" %}


Today, in the post-covid era, as vaccinations are becoming more widely available, more and more people want to travel. According to ourworldindata.org, currently, 30.44% of the world population are vaccinated with at least one dose. In the United States alone this number is 59%.

When planning your trip, would you consider Airbnb as an option?

In this article, I have used Boston, one of the popular cities for Airbnb. I have analyzed the listings offered, the price trend, and the amenities provided. The dataset is available here. For more ideas, visualizations of all Boston datasets you can look here.

# Data Analysis Process
I have used <b>CRISP-DM (CRoss Industry Standard Process for Data Mining)</b> in this process. It is a process model with six phases that naturally describes the data science life cycle. The phases are:

  1. Business Understanding
  2. Data Understanding
  3. Data Preparation
  4. Modeling
  5. Evaluation
  6. Deploy

However, the 6 phases aren’t necessary for all the projects; a lot of questions can be answered without building models. In this post, <b>I will apply CRISP-DM steps 1,2,3,5, and 6 to the Airbnb Boston datasets.
Business Understanding </b>

# Business Understanding
Boston is one of the most visited cities in the world. I will discuss some questions that those who plan to travel to Boston and use Airbnb would find interesting.

  1. Which neighborhood of Boston has more listings? What type of properties are provided there?
  2. What is the availability of the listings across each neighborhood and their review/responsiveness?
  3. What is the price trend among the neighborhood throughout the year? what is the weekend price trend?
  4. Which amenities, in general, can you expect and what are some rare amenities, and how much extra do you have to pay to get those?
  5. Which neighborhood is recently seeing a surge in listings?

# Data Understanding

In the dataset, you can find three files. The following Airbnb activity is included in these files:

  1. Listings, including full descriptions and average review score
  2. Reviews, including unique id for each reviewer and detailed comments
  3. Calendar, including listing id and the price and availability for that day

# Analysis

## 1. Which neighborhood of Boston has more listings? What type of properties are provided there?

Boston has <b>25 neighborhoods</b>, and as expected listings are not distributed uniformly in all these neighborhoods. If we plot the listings’ location on Boston Map, we can clearly see that these are in clusters. I have used folium (a python library) to draw out the points. Here blue points depicts an entire room/apartment, green depicts the private room and red depict the shared room.

{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/1.properties-.png" alt="Listings in Boston’s neighborhood" caption="Listings in Boston’s neighborhood" %}


We can clearly see that there is a concentration of listings in Back Bay and South End. The color of the neighborhood is encoded with the listings count in that neighborhood. Using that we can see that <b>Jamaica Plain(343), South End(326), Back Bay(302)</b> are among the top three in listings count.

{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/1.property-type.png" alt="No. of listings in each property type" caption=" No. of listings in each property type" %}


Coming to the type of the room offered, from map we can see a lot of blue dots, i.e. <b>entire room/apartment is one of the prominent offerings (2127)</b>, the second is Private room(1378) and lastly shared room(80). With respect to property type Apartment(over 2500 listings) is at first, then house(over 500) and then condo, town house, bed & breakfast and so on.

## 2. What is the availability of the listings across each neighborhood and their review and host responsiveness?

When it comes to staying in a hotel, we want all things to be perfect. We don’t want any bad event to ruin our stay. It is obvious to look for review and responsiveness of the hotels.

{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/2.availability_index.png" alt="" caption="" %}


During peak season, hotels book very fast and availability is a concern. But availability is not constant among all neighborhood. Mission Hill, Allston, Leather District offers average availability of ~150 days in a year while <b>Dorchester, Bay Village, Roslindale and Mattapan offer ~220 days in a year.</b>

Response time is also an important factor. If you want your queries to be resolved within the hour, Bay Village, Mattapan and East Boston is your best bet. In these areas, more than 55% of the listing’s hosts have responded within an hour. If you don’t mind few hours of wait, you can go with <b>West End and China Town with more than 60% of the listing’s hosts have responded within a few hours.</b>

{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/2.response-index.png" alt="" caption="" %}


On the review side, we can see that all the neighborhood has good ratings. If we look for check-in and cleanliness only, the least rating is about 9 belonging to West End’s listings. Leather District is at the top, with an average rating of 10 in terms of check-in, cleanliness and communication. One thing to note here is that Leather District also had the least listings count. Apart from it, <b>Jamaica Plain, West Roxbury and Roslindale is among the high rated neighborhoods.</b>

{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/2.review-index.png" alt="" caption="" %}

## 3. What is the price trend among the neighborhood throughout the year? What is the weekend price trend?

Price is also one of the main factors for your travel planning. Looking at the Boston listings datasets, we can in general see a trend in price among neighborhoods. <b>West End, Chinatown, South Boston Water Front, Bay Village and Leather District are among highly-priced neighborhoods</b> with average listing price going for over 400$ . Looking at the trend we can see that these neighborhood are <b>costlier in September, October and November</b> with an average increase of 100$ from other months. Hyde Park, Mattapan and Dorchester are among cheaper options with average listings price of under 100$.

{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/3.yearly-priceindex.png" alt="" caption="" %}


In general, listings are costlier in a different months for each neighborhood, but overall from graph, we can see September and October are costlier months for Boston.

On weekly basis, we can see there is not much variation in the price in some neighborhood, while in some like Bay Village, West End, China Town, Down Town, Back Bay and Leather District we can see some variations based on weekdays.

{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/3.weekly-price-index.png" alt="" caption="" %}

{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/3.weekly-price-var-index.png" alt="" caption="" %}

On plotting the bar chart we can see that Friday and Saturdays are priced <b>10$–15$</b> more on average. From the previous chart, we also saw that this variation is not the same. For instance, Leather District has this difference of about <b>75$</b>.

<b><i>How are different types of rooms priced in Boston?</i></b>

If we see the pricing on the basis of room type, Entire home/apartment is almost always priced high. Private rooms are second-costliest room type followed by shared rooms.

{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/3.roomtype-price-index.png" alt="" caption="" %}

## 4. Which amenities, in general, can you expect and what are the rare amenities and how much extra do you have to pay to get those?

Amenities are important for your pleasant stay. In Boston dataset, we can see that there are about <b>45 different amenities</b> provided. So, there is a good chance you will find what you need here.
{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/4-amenities-normal-df.png" alt="Top 10 common amenities in Boston listings" caption="Top 10 common amenities in Boston listings" %}


If we look at the top common amenities; kitchen, Heating, Internet, Air conditioning, TV and so on are provided in more than 50%as we expect. We can also see the percentage of listings offering it in each room type from the left table. If we set up 60% as cut-off, we can see from the table that <b>we can always get all these amenities from an apartment or Private room; but shared room only offers Kitchen, Heating and Wifi only in their 60% listings.</b>

{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/4.Avg-amenities-price.PNG" alt="average price w/wo top 10 common amenities" caption="average price w/wo top 10 common amenities" %}

{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/4-amenities-normal-df.png" alt="" caption="" %}

<b><i>How much extra do you need to pay for getting all the top 10 amenities?</i><b>

For the listings in Boston, we can see that there is a slight difference in prices with and without all common amenities. The listings with all common amenities cost <b>64$</b> more on average than others
average price w/wo top 10 common amenities


{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/4-amenities-normal-df.png" alt="" caption="" %}

On average listings with all common amenities costs around <b>220$</b> while others only cost around <b>156$</b>.

Coming to the rare amenities, <b>Washer/Dryer, Hot tub, Smoking Allowed, Pets are among a few are amenities</b> allowed in Boston Airbnb listings. Also, Washer/Dryer is not available in a Private or shared Room.
## 5. Which neighborhood is seeing a surge in listings?

As we all know that the popularity of Airbnb is increasing day by day. Looking at the recent listing would give a brief idea about the expansion in Boston.

{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/5.recent-listings-index.png" alt="Last 90 days listings" caption="Last 90 days listings" %}

Looking at the above chart we can see that, <b>Brighton(21), Fenway (18) and Dorchester(14)</b> are among the top 3 neighborhood with new listings in the past 90 days. Roslindale, West End and Bay Village have the least new listings added.

Percentage-wise, <b>Longwood Medical Area has surged in new listings as the number of listings</b> doubled in the last 90 days.

{% include figure image_path="/assets/images/Things-to-consider-before-planning-your-trip-to-Boston/5.newly-listed-roomtype-index.png" alt="" caption="" %}

If we look at the room type, there is a surge in shared rooms offering which increased by <b>16%</b> followed by private room at <b>7%</b> and apartments at <b>3%</b> in the past <b>90 days</b>.

# Conclusion

In this article, we have cover five questions about the Airbnb listings in Boston; here are some takeaways:

  1. Among the 25 neighborhoods listed, Jamaica Plain(343), South End(326), Back Bay(302) are among the top 3 neighborhood with more listings, with Back Bay and South End among the densest. Also In these areas Apartment or Entire house is a common type of listings.
  2. In terms of availability Mission Hill and Allston offers the least availability throughout the year while Roslindale and Mattapan offer higher availability. East Boston and Mattapan has over 60% listings with host response time within an hour while West End and Mission Hill has only 20%–30% listings with host response time within an hour. Also, West Roxbury and Roslindale have high rated listings among others.
  3. In general September and October months are costlier in Boston with some neighborhood listing prices go up by ~ 100$. Hyde Park and Mattapan is some of budget-friendly neighborhoods to stay while Bay Village and Leather District are among the costliest neighborhoods. The price of listings also changes with weekday, with Friday and Saturday priced Higher.
  4. Kitchen, Heating, Internet, Air conditioning, TV are some common amenities offered in Boston listings and on average you have to pay 64$ more for getting all the top 10 common amenities. Washer/Dryer, pet allowance, hot tub are among rare amenities.
  5. Brighton, Fenway and Dorchester are witnessing a surge in new listings in the past 90 days. While Longwood Medical Area has doubled its listing count. Among these increases, shared rooms are most common.

