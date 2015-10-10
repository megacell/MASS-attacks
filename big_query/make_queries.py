import pickle

from pdb import set_trace as T
from query import BigQuery, format_response

lambda_sql = '''
SELECT
  pickup as station,
  count(*) as pickups
FROM [nyc_taxi_data.yellow_filtered]
GROUP BY station
ORDER by pickups DESC
'''
bq = BigQuery('nyc-taxi-trips-analysis')

res = format_response(bq.query(lambda_sql))
pickle.dumps(res, open('lambda.pkl'))
