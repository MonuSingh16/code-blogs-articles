/* Which creators published a monetized video 5 mins or longer 
followed by a non-monetized video that was also 5 mins or longer? 
Round intermediary calculations to 2 decimals. Output the email of creators sorted the alphabetical order.
*/


/* Write a query to identify customers who placed more than three transactions each in both 2019 and 2020.
*/

WITH transaction_count AS (
    SELECT u.id,
    u.name,
    SUM(CASE WHEN YEAR(t.created_at)='2019' THEN 1 ELSE 0 END) AS t_2019,
    SUM(CASE WHEN YEAR(t.created_at)='2020' THEN 1 ELSE 0 END) AS t_2020
    FROM transactions t
    JOIN users u
    on u.id = t.user_id
    GROUP BY 1
    HAVING t_2019 > 3 AND t_2020 > 3
)

SELECT tc.name as customer_name
FROM transaction_count tc