### Grid generation

ROT       = 36                      # Degree of rotation to fit Manhattan grid
LON, LAT  = -74.038658, 40.711516   # Starting point
DIAF      = -0.235                  # Dialation factor, to make sure squares
                                    # look square after rotation
GRID_SIZE = 0.0022                  # Size of each grid square in lat/lon

XSIZE = 23  # Number of grid boxes in x-direction
YSIZE = 35  # Number of grid boxes in y-direction

### Data processing
LAMBDA_FILE = 'data/lambda.csv'
T_FILE = 'data/T.csv'

# Grid boxes with rides below this number are cut off, they are mostly in the
# water
LAMBDA_CUTOFF = 700

# Stations with less than this number of destinations will be removed
DEST_CUTOFF = 400

# Total time sampled over in hours, when the number of arrivals is divided by
# this number, we get the arrival rate:

# (weeks from 1/1/2009 to 6/30/2015) * (days per week) * (hours per day)
TOTAL_TIME = 337.3 * 5 * 2

# Convert from seconds in results to hours
TRIP_TIME_SCALE = 3600

# The alpha value used in Laplacian Smoothing
SMOOTHING_FACTOR = 1

# The delta coordinates used in smoothing of T
DELTAS = [(i, j) for i in [-1,0,1] for j in [-1,0,1]]

# Value to use for trip time if there are no trips between two stations
INF_TRIP_TIME = 100000000
