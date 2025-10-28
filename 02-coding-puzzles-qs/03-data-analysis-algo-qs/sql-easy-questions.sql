/* Q 
You're tasked with analyzing a Spotify-like dataset that captures user listening habits.
For each user, calculate the total listening time and the count of unique songs they've listened to.
 Round the total listening duration to the nearest whole minute.

The output should contain three columns: 'user_id', 'total_listen_duration', and 'unique_song_count'.
*/

select 
    user_id,
    round(sum(ifnull(listen_duration, 0)) / 60.0) as total_listen_duration,
    count(distinct(song_id)) as unique_song_count
from listening_habits
group by user_id

/* Q1. Find the highest number of parking spots across all properties */

select max(parking_spots) as max_parking_spots 
    from airbnb_dim_property;

/* Q2. Among videos with duration > 5 min, how many unique users watched per video? 
Output the video_id and count. Order by number of distinct viewers desc. */

SELECT video_id, COUNT(DISTINCT user_id) AS number_of_distinct_users
FROM youtube_reactions
    WHERE video_id IN (SELECT video_id FROM youtube_videos WHERE duration_sec > 300)
GROUP BY video_id
ORDER BY 2 desc;



/* Q3. What is the average time in days it takes for a user to enroll in prime after creating an account?
Ignore users who never joined prime. Round to 2 decimal places. */

select round(avg(datediff(prime_joined_dt, joined_dt)), 2) as avg_days_to_join
from amazon_users
where prime_member = 1

SELECT avg(timestampdiff(day, joined_dt, prime_joined_dt)) avg_days_to_join
FROM amazon_users
WHERE prime_member = 1;

/* Q4. What is the average video duration in minutes of monetized videos and non-monetized videos
among users in UK, Australia, and US? Output the monetization type and average duration in minutes.
 Round duration to the 2 decimal points.

Expected Output:
Your answer should match this structure exactly. Keep in mind SQL sorting, grouping, & rounding when answering.
is_monetized	avg_video_duration
0	5.28
1	5.75

*/
. 

SELECT a.is_monetized, round(avg(a.duration_sec) / 60, 2)
FROM youtube_videos a
WHERE user_id IN (
  SELECT DISTINCT(user_id)
  FROM youtube_users
  WHERE country in (
  'UK', 'Australia', 'US'
  )
)
GROUP BY is_monetized
--------

SELECT 
  is_monetized, 
  ROUND(CAST(AVG(duration_sec) as double) / 60), 2) avg_video_duration
FROM youtube_videos a
LEFT JOIN youtube_users b
ON a.user_id = b.user_id
WHERE b.country in ('US', 'UK', 'Australia')
GROUP BY a.is_monetized

--------

SELECT is_monetized, round(cast(avg(duration_sec) as double) /60, 2) avg_video_duration
FROM youtube_videos
JOIN (SELECT user_id FROM youtube_users WHERE country IN ('UK', 'Australia', 'US')) u
USING (user_id)
GROUP BY is_monetized;

/*Q5. Find the top 5 largest properties at AirBnB (3000+ sqft). Ignore ties, return exactly 5 order by property id asc.

Expected Output:
Your answer should match this structure exactly. Keep in mind SQL sorting, grouping, & rounding when answering.
property_id
2
3
4
15
25
*/
 
select property_id 
from airbnb_dim_property 
where total_sqft = '3000+' 
order by property_id ASC 
limit 5

/* Q6. In each country, what is the percentage of users who enrolled on prime within 5 days 
(inclusive) after joining Amazon? Round to 2 decimal places and order the output by country asc.

Expected Output:

Your answer should match this structure exactly. Keep in mind SQL sorting, grouping, & rounding when answering.
country	percentage
Australia	25
Brazil	7.69
China	14.29
France	20
UK	14.29
US	7.14
*/

SELECT country, 
    round(avg(joined_prime_within_5_days) * 100, 2) as percentage
FROM (
    SELECT country,
        CASE 
            WHEN timestampdiff(day, joined_dt, prime_joined_dt) <= 5
            THEN 1 
            ELSE 0
        END joined_prime_within_5_days,
        joined_dt,
        prime_joined_dt
    FROM amazon_users
) prime_users
GROUP BY country
ORDER BY country;

/* Q7. How many transactions have missing payment method?

Expected Output:

Your answer should match this structure exactly. Keep in mind SQL sorting, grouping, & rounding when answering.
num_missing
19
*/ 

SELECT count(transaction_id) as num_missing
FROM appstore_transactions	
WHERE payment_method is NULL

-----

SELECT count(*) num_missing
FROM appstore_transactions
WHERE payment_method IS NULL;

/* Q8. Find the top poster(s) on Facebook and count how many of each post type they had. 
Please consider ties and return all users if tie exists. Order answer by creator asc and post type asc.

Expected Output:

Your answer should match this structure exactly. Keep in mind SQL sorting, grouping, & rounding when answering.
creator	post_type	total_posts
2	ad	1
2	poll	1
2	post	1
2	video	2
8	ad	1
8	poll	2
8	post	2

*/

with posts_rpt as ( 
    select 
     user_id 
    ,rank() over(order by count(*) desc) rnk 
    from fb_posts 
    group by 1 
) 
Select 
 user_id creator
,post_type
,count(*) total_posts 
from fb_posts 
where user_id in (select user_id from posts_rpt where rnk = 1) 
group by 1, 2 
order by 1, 2

/* Q9. What is the global click through rate (CTR) of advertisement and non-advertisement link types?
CTR = # of Clicked / # of Viewed in this scenario. Round conversion to the 2 decimal points and order by advertisement asc.

Expected Output:

Your answer should match this structure exactly. Keep in mind SQL sorting, grouping, & rounding when answering.
advertisement	conversion
0	0.41
1	0.5

*/

SELECT advertisement, 
    round(SUM(CASE WHEN event_type = 'clicked' THEN 1 ELSE 0 END) 
    / SUM(CASE WHEN event_type = 'viewed' THEN 1 ELSE 0 END), 2) conversion
FROM google_search_activity
GROUP BY advertisement
order by 1 asc;

/* Q10. Your manager wants a cleaned up report with only Low, Med, High as 
growth types broken down by total count. Help him clean up the data with 
just those values and when you return your answer make sure its sorted by total desc.

Expected Output:

Your answer should match this structure exactly. Keep in mind SQL sorting, grouping, & rounding when answering.
growth_typecleaned	total
Low	31
High	13
Med	6


*/ 

SELECT
CASE
  WHEN growth_type = 'Loww' THEN 'Low'
  WHEN growth_type = 'low' THEN 'Low'
  WHEN growth_type = 'Hgh' THEN 'High'
  ELSE growth_type
END as growth_typecleaned, count(*) as total
FROM messy_data
GROUP BY growth_typecleaned

/* Q11.  Find the total number of luxury properties that AirBnb. */

SELECT COUNT(property_type) as total_luxury_properties
FROM airbnb_dim_property
WHERE property_type = 'Luxury'

------------

select count(*) as total_luxury_properties 
from airbnb_dim_property 
where property_type = 'Luxury';

/* Q12. Find the total number of impressions each ad received. Sort answer by impressions desc and ad asc.
*/

SELECT ad_name, COUNT(ad_exp_id) as ad_impressions
FROM ads_actions
GROUP BY ad_name
ORDER BY 2 DESC, 1 ASC;

------------

select
 ad_name
,count(*) ad_impressions
from ads_actions
group by 1
order by 2 desc, 1 asc

/* Q13.   */

WITH top_property as (
SELECT property_id
,RANK() OVER(ORDER BY SUM(num_of_nights) DESC) rnk
FROM airbnb_fct_rentals
GROUP BY 1
  )
SELECT property_id from top_property WHERE rnk = 1

-------------

with property_rpt as (
    SELECT property_id
    ,count(*) rentals
    ,rank() over(order by count(*) desc) rnk
    from airbnb_fct_rentals r
    group by 1
)
select property_id
from property_rpt
where rnk = 1
order by 1 asc

/*Q14. Among all the events after April 1, 2022, what is the total count as well as 
the most recent datetime of each event type? Order the output in the alphabetical order of event_type.
*/

SELECT event_type, COUNT(*) as ride_counts
,date(max(creation_dt)) event_datetime
FROM google_maps_actions
WHERE creation_dt > '2022-04-01'
GROUP BY 1
ORDER BY 1 ASC;


/* Q15. Find the percent of users who are permanently banned. Round your answer to 2 decimals.
*/

SELECT
ROUND(COUNT(CASE WHEN is_temporary = 0 THEN 1 ELSE NULL END) / 
  COUNT(*) * 100, 2) permanent_banned_prcnt
FROM banned_users


/* Q16. You are on the Digital Marketing team at Google. You are asked to find 
what is the click-through-rate, (CTR), per each website type? CTR = # of Clicked / # of Viewed. 
Be sure to use the 'event_type' column for this one and not the 'conversion' column, 
our data engineer said theres some bugs in this one. Round to 2 decimal places. 
Output answer sorted by type asc.
*/

SELECT A.type, 
ROUND(SUM(CASE WHEN A.event_type = 'clicked' THEN 1 ELSE 0 END) /
  SUM(CASE WHEN A.event_type = 'viewed' THEN 1 ELSE 0 END), 2 )conversion
FROM
(SELECT a.*, b.type
FROM google_search_activity a
JOIN google_search_websites b
ON a.website_id = b.website_id) A
GROUP BY A.type
ORDER BY 1 ASC;

-----------------

WITH website_type AS (
    SELECT type, event_type
    FROM google_search_websites
    JOIN google_search_activity
    USING (website_id)
)
SELECT type, 
    round(sum(case when event_type = 'clicked' then 1 else 0 end) 
    / sum(case when event_type = 'viewed' then 1 else 0 end), 2) conversion
FROM website_type
GROUP BY type
ORDER BY type;

/* Q17. Which users who joined between Jan 1, 2022 and Jan 8, 2022 never made 
a single purchase on App Store between ? Output the email and joined date of users sorted by email ascending
*/

SELECT email, joined_dt
FROM apple_users 
LEFT JOIN (SELECT distinct user_id FROM appstore_transactions) appstore_users
USING (user_id)
WHERE appstore_users.user_id is null
and joined_dt between '2022-01-01' and '2022-01-08'
ORDER BY email asc;

/* Q18. Find the ride refund rate of all Uber trips, that is out of all rides how many are refunded?.
 Be careful how you count... there could be more than 1 reason a rider cancels. Round your answer to 2 decimals.
 */

SELECT  
ROUND(COUNT(DISTINCT trip_id) / 
  (SELECT COUNT(DISTINCT ride_id) FROM uber_fct_trips) * 100, 2) ride_refund_rate
FROM uber_refunds

-------
select round(count(distinct r.trip_id) / count(distinct t.ride_id) * 100, 2) ride_refund_rate
from uber_fct_trips t, uber_refunds r

/* Q19. Among all users other than the UK, how many worked in Stripe, Uber, or Meta? 
Be careful of users who might have worked in a few of these places should you double count?
*/

WITH user_counting as (
SELECT a.*, b.country
FROM linkedin_emp_history as a
JOIN (SELECT user_id, country FROM linkedin_users) as b
ON a.user_id = b.user_id
)
SELECT COUNT(DISTINCT user_id) as user_count
FROM user_counting
WHERE country <> 'UK' AND
employment in ('Stripe', 'Uber', 'Meta')

--------------------

SELECT count(distinct user_id) user_count
FROM linkedin_emp_history 
JOIN linkedin_users
USING (user_id)
WHERE employment IN ('Stripe', 'Uber', 'Meta')
AND country != 'UK';

/*Q20.  Your boss is trying to put together some numbers for a leadership presentation. 
They ask which 3 employers have the longest average tenure in months? 
Your boss doesn't care about ties and ONLY wants exactly 3 rows returned 
rounded to 2 decimals and sorted by tenure desc. Are you up to the task?
*/

WITH rank_employer as (
SELECT employment
,AVG(timestampdiff(MONTH, start_date, end_date)) avg_tenure,
RANK() OVER(ORDER BY AVG(timestampdiff(MONTH, start_date, end_date)) DESC) rnk
FROM linkedin_emp_history
GROUP BY employment
ORDER BY avg_tenure DESC
)
SELECT employment, avg_tenure
FROM rank_employer
WHERE rnk <= 3

--------------

SELECT employment, avg_tenure 
FROM (
    SELECT employment, round(avg(timestampdiff(month, start_date, end_date)),2) avg_tenure
    FROM linkedin_emp_history
    GROUP BY employment
) emp_sq
ORDER BY avg_tenure DESC
LIMIT 3;

/*Q21 For each product category, which product is the most expensive? 
Output the category, product name and price in the alphabetical order of category. Only return one product per category.
*/

SELECT a.category, a.product_name, a.price
  FROM (
SELECT category, product_name, max(price) as price,
RANK() OVER(PARTITION BY category ORDER BY max(price) DESC) rnk
FROM amazon_products
GROUP BY category, product_name
  ) a
WHERE a.rnk = 1


-----------

SELECT 
 category
,product_name
,price 
FROM (
    SELECT category, product_name, price,
        row_number() over(partition by category order by price desc) price_rank
    FROM amazon_products
) rank_table
WHERE price_rank = 1
ORDER BY category;

/* Q22 What is the average tenure per company in months? 
Output the company and average tenure, rounded to 1 decimal point and sorted by company ascending.
*/
SELECT employment as company, 
  ROUND(AVG(timestampdiff(MONTH, start_date, end_date)), 1) as avg_tenure
FROM linkedin_emp_history
GROUP BY employment
ORDER BY employment;

/*Q23 Find the total number of poor quality properties that AirBnb has. Please return a single number.
*/

SELECT COUNT(*) as total_properties
FROM airbnb_dim_property
WHERE property_quality = 'Poor'

/*Q24 Get the like and comment counts per day. Output the date, 
reaction type, and count sorted by date asc and reaction type asc. It's okay if some of the dates are missing.
*/

WITH reaction_type as (SELECT DATE(reaction_dt) as reaction_date ,reactions as reaction_type,
  SUM(CASE WHEN reactions = 'like' THEN 1 ELSE 0 END) as total_likes,
  SUM(CASE WHEN reactions = 'comment' THEN 1 ELSE 0 END) as total_comments
  FROM youtube_reactions
WHERE reactions in ('like', 'comment')
GROUP BY reaction_dt, reactions
ORDER BY reaction_dt
)
SELECT reaction_date, reaction_type
  ,sum(total_likes) as total_likes
  ,SUM(total_comments) as total_comments
FROM reaction_type
GROUP BY reaction_date, reaction_type

---------------

SELECT 
    DATE(reaction_dt) reaction_date,
    reactions reaction_type,
    count(case when reactions = 'like' then 1 else null end) total_likes,
    count(case when reactions = 'comment' then 1 else null end) total_comments
    FROM youtube_reactions
    where reactions in ('like', 'comment')
    group by 1, 2
    order by 1 asc, 2 asc

/*Q.25 Find the full view rate of all viewers. That is what is the percent 
of views that are watched to completion? Round answer to 2 decimals. */

select round(count(case when viewed_to_completion = 1 then 1 else null end) 
/ count(*) * 100, 2) full_view_rate 
from tiktok_fct_views

/*Q.26 Get the average rating and price per app. Round to 2 decimal places.
 Output the table in the alphabetical order of app name.
*/


SELECT app_name
,ROUND(AVG(rating), 2) as avg_rating
,ROUND(AVG(price), 2) as avg_price
FROM appstore_transactions
GROUP BY app_name
ORDER BY app_name

/*Q.27 How many job posts on Linkedin have been posted just once? Return the total number of single posts.
*/

SELECT COUNT(*) num_unique_post
  FROM
(
  SELECT post_id
  FROM linkedin_job_posts
  GROUP BY post_id
  HAVING COUNT(*) = 1
  )sq_post;

/* Q.28 Get the like and comment counts per video per day.
 Output the date, video_title, and count in the ascending order of the date. 
 It's okay if some of the dates are missing.
*/

SELECT event_date, video_title, COUNT(*) action_count
FROM (
    SELECT
        DATE(reaction_dt) event_date,
        video_id
    FROM youtube_reactions
    WHERE reactions IN ('like', 'comment') 
) reaction_sq
JOIN youtube_videos USING (video_id)
GROUP BY event_date, video_title
ORDER BY event_date;

/* Q29. Find the usernames of all users who are permanently banned. Order by username asc */

SELECT username
FROM fb_users_all
  WHERE user_id in (SELECT user_id
FROM banned_users
WHERE is_temporary = '0')

------------

select
 u.username
from fb_users_all u
join banned_users b
on u.user_id = b.user_id
where b.is_temporary = 0
order by 1

/* Q.30 Find the total number of properties at AirBnb located in Bon Air. */

SELECT COUNT(*) total_properties
FROM airbnb_dim_property
WHERE location_town in ('Bon Air')

/*Q.31 Return the email address of creators who posted monetized videos with at least 5 minute duration more than once. 
Order in the ascending order of email. */

SELECT email 
  FROM youtube_users
  WHERE user_id in (SELECT user_id
  FROM youtube_videos
WHERE is_monetized = '1'
and duration_sec > 300
GROUP BY user_id
HAVING COUNT(video_id) > 1)
ORDER BY email

------------
WITH creators_with_lengthy_monetized_vids AS (
    SELECT user_id
    FROM youtube_videos
    WHERE duration_sec >= (5*60) and is_monetized = 1
    GROUP BY user_id
    HAVING count(*) > 1
)
SELECT email
FROM youtube_users JOIN creators_with_lengthy_monetized_vids
USING (user_id)
ORDER BY email;

/* Q.32 Well executives heard the AI/ML buzz and want a ML model for their 
store data to help predict transaction amount and total spend. 
They give you the data and ask if its possible, the key variable they think is store.
 You can only build a ML model in the next step if the total number of nulls 
 in the field are less than 5% of total rows. 
 
 Calculate the % of total store rows that are null to see if you can proceed. 
 Round your final answer to 2 decimals, fingers crossed for you!

*/ 


SELECT 
ROUND(SUM(CASE WHEN store is NULL THEN 1 else 0 END) / COUNT(*) * 100, 2) prcnt_store_null
FROM ml_inputate

/*Q.33 You are on the community polls DS team at twitter and 
asked to calculate the percent of response types for each poll. 
Please calculate the percent of yes repsonses and percent of no responses for each poll. 
Round final answer to 2 decimals and sort by poll name asc.
*/


SELECT poll_name,
  ROUND(SUM(CASE WHEN poll_answer = 'YES' THEN 1 ELSE 0 END) / 
  COUNT(*) * 100, 2) yes_prcnt,
  ROUND(SUM(CASE WHEN poll_answer = 'NO' THEN 1 ELSE 0 END) / 
  COUNT(*) * 100, 2) no_prcnt
FROM twitter_polls
GROUP BY poll_name
ORDER BY poll_name;

------------

with polls_rpt as (
    SELECT  
    poll_name, 
    sum(case when poll_answer = 'Yes' then 1 else 0 end) total_yes,
    sum(case when poll_answer = 'No' then 1 else 0 end) total_no,
    count(*) total_responses
    FROM twitter_polls
    group by 1
)
select 
poll_name,
round(100.00 * total_yes / total_responses, 2) yes_prcnt,
round(100.00 * total_no / total_responses, 2) no_prcnt
from polls_rpt
order by 1 asc


 /* Q.34 Given a table of bank transactions with columns id, transaction_value, 
 and created_at representing the date and time for each transaction, write a query to get the last transaction for each day.
*/
WITH timed_transactions AS 
       (
	SELECT * , ROW_NUMBER() OVER (PARTITION BY DATE(created_at) ORDER BY created_at DESC) as ordered_time
	FROM bank_transactions
	)
SELECT  created_at,transaction_value, id  FROM timed_transactions
WHERE ordered_time = 1;

--
SELECT * FROM bank_transactions 
WHERE created_at IN ( SELECT MAX(created_at) 
AS maxdate 
FROM bank_transactions 
GROUP BY DATE(created_at) )

/* Q35. 
Given a table of transactions and a table of users, write a query to determine if users tend to order more to their primary address versus other addresses.

Note: Return the percentage of transactions ordered to their home address as home_address_percent.

*/
SELECT
ROUND( 
SUM(CASE WHEN u.address = t.shipping_address THEN 1 END)
/ COUNT(t.id)
,2)  as home_address_percent
FROM transactions as t
JOIN users as u
ON t.user_id = u.id

/* Q36. We’re given two tables, a users table with demographic information and the neighborhood they live in and a neighborhoods table.

Write a query that returns all neighborhoods that have 0 users. 

*/
/* 
Whenever the question asks about finding values with 0 something (users, employees, posts, etc..)
 immediately think of the concept of LEFT JOIN! An inner join finds any values
  that are in both tables, a left join keeps only the values in the left table.

*/

SELECT name FROM neighborhoods 
WHERE id not in (
    SELECT distinct neighborhood_id
    FROM users)

--------------

SELECT n.name   
FROM neighborhoods AS n 
LEFT JOIN users AS u
    ON n.id = u.neighborhood_id
WHERE u.id IS NULL

