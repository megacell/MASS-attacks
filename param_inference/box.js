function toTransformedGrid(row, emit) {
    // Constants for transformation
    var x0 = -74.038658,
        y0 = 40.711516,
        a = 0.80901699,
        b = -0.58778525,
        c = 0.39766626,
        d = 0.94714653,
        gridsize = 0.0022;

    var x = row.lon - x0,
        y = row.lat - y0;

    var transform = function (x, y) {
        var xp, yp;
        x = x - x0;
        y = y - y0;
        xp = ((a * x + b * y) / gridsize) | 0;
        yp = ((c * x + d * y) / gridsize) | 0;
        return [xp, yp];
    };

    var pickup  = transform(row.pickup_longitude, row.pickup_latitude);
    var dropoff = transform(row.dropoff_longitude, row.dropoff_latitude);

    emit({ plat : row.pickup_latitude,
           plon : row.pickup_longitude,
           dlat : row.dropoff_latitude,
           dlon : row.dropoff_longitude,
           px : pickup[0], py : pickup[1], dx : dropoff[0], dy : dropoff[1],
           ptime : row.pickup_datetime,
           dtime : row.dropoff_datetime,
           trip_distance : row.trip_distance,
           fare_amount : row.fare_amount,
           total_amount : row.total_amount });
}

bigquery.defineFunction(
    'TO_TRANSFORMED_GRID',                  // Name of the function exported to SQL
    ['pickup_latitude', 'pickup_longitude',
     'dropoff_latitude', 'dropoff_longitude',        // Names of input columns
     'pickup_datetime', 'dropoff_datetime',
     'trip_distance', 'fare_amount', 'total_amount'],
    [{'name': 'plat', 'type': 'float'},     // Output schema
     {'name': 'plon', 'type': 'float'},
     {'name': 'dlat', 'type': 'float'},
     {'name': 'dlon', 'type': 'float'},
     {'name': 'px', 'type': 'integer'},
     {'name': 'py', 'type': 'integer'},
     {'name': 'dx', 'type': 'integer'},
     {'name': 'dy', 'type': 'integer'},
     {'name': 'ptime', 'type': 'timestamp'},
     {'name': 'dtime', 'type': 'timestamp'},
     {'name': 'trip_distance', 'type': 'float'},
     {'name': 'fare_amount', 'type': 'float'},
     {'name': 'total_amount', 'type': 'float'}
    ],
    toTransformedGrid                        // Reference to JavaScript UDF
);
