-- Initial filtering and binning

SELECT plat, plon, dlat, dlon,
       px, py, dx, dy,
       CONCAT(STRING(px),'-' , STRING(py)) as pickup,
       CONCAT(STRING(dx),'-' , STRING(dy)) as dropoff,
       USEC_TO_TIMESTAMP(ptime) as ptime,
       USEC_TO_TIMESTAMP(dtime) as dtime,
       trip_distance, fare_amount, total_amount
FROM
  TO_TRANSFORMED_GRID(
    SELECT pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude,
           pickup_datetime, dropoff_datetime,
           trip_distance, fare_amount, total_amount
    FROM [nyc-tlc:yellow.trips]
    WHERE TIMESTAMP_TO_SEC(dropoff_datetime) - TIMESTAMP_TO_SEC(pickup_datetime)
      BETWEEN 30 AND 3600
    AND DAYOFWEEK(pickup_datetime) BETWEEN 2 AND 6 -- Monday to Friday
    AND HOUR(pickup_datetime) BETWEEN 17 AND 19    -- 5pm to 7pm
    AND trip_distance < 15                         -- Less than 15 miles
  )
WHERE px BETWEEN 0 and 23
  AND dx BETWEEN 0 and 23
  AND py BETWEEN 0 and 35
  AND dy BETWEEN 0 and 35

-- lambda_i
SELECT
  pickup as station,
  count(*) as pickups
FROM [nyc_taxi_data.yellow_filtered]
GROUP BY station
ORDER by pickups DESC

-- T_ij
SELECT
  pickup, dropoff,
  count(*) as counts,
  AVG(TIMESTAMP_TO_SEC(dtime) - TIMESTAMP_TO_SEC(ptime)) as trip_time_mean
FROM [nyc_taxi_data.yellow_filtered]
GROUP BY pickup, dropoff
ORDER BY counts DESC

-- number of passengers traveling between a pair of stations
-- (can be used to compute p_ij)
SELECT
  pickup, dropoff, count(*) as counts
FROM [nyc_taxi_data.yellow_filtered]
GROUP BY pickup, dropoff
ORDER BY pickup, dropoff

-- p_ij
SELECT
  trip.pickup, trip.dropoff,
  trip.counts / FLOAT(arrival.counts) as p_ij
FROM
  (SELECT
     pickup, dropoff, count(*) as counts
   FROM [nyc_taxi_data.yellow_filtered]
   GROUP BY pickup, dropoff
  ) as trip
JOIN
  (SELECT
     pickup, count(*) as counts
   FROM [nyc_taxi_data.yellow_filtered]
   GROUP BY pickup
  ) as arrival
ON trip.pickup = arrival.pickup
ORDER BY trip.pickup, trip.dropoff
