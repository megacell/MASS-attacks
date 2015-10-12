### Grid generation

ROT       = 36                      # Degree of rotation to fit Manhattan grid
LON, LAT  = -74.038658, 40.711516   # Starting point
DIAF      = -0.235                  # Dialation factor, to make sure squares
                                    # look square after rotation
GRID_SIZE = 0.0022                  # Size of each grid square in lat/lon

XSIZE = 24  # Number of grid boxes in x-direction
YSIZE = 36  # Number of grid boxes in y-direction

### Data processing
LAMBDA_FILE = 'data/lambda.csv'
T_FILE = 'data/T.csv'

# Grid boxes with rides below this number are cut off, they are mostly in the
# water
LAMBDA_CUTOFF = 700

# Stations with less than this number of destinations will be removed
DEST_CUTOFF = 400

# Hard-code stations to be included/excluded
BLACKLIST = ['20-29', '21-29', '22-29', '22-28', '23-28', '23-34', '20-5',
             '21-5', '18-4', '19-4', '20-4', '21-4', '20-3', '23-3', '19-2',
             '20-2', '21-2', '22-2', '23-2', '20-1', '21-1', '22-1', '23-1',
             '20-0', '21-0', '22-0', '23-0']
WHITELIST = ['19-19', '19-18', '19-13', '20-13', '21-13', '19-13', '21-13',
             '19-12','18-10', '19-10', '18-9', '17-8', '13-7', '15-7', '16-7',
             '15-6']

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
