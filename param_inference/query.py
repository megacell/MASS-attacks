from pdb import set_trace as T
from apiclient.discovery import build
from apiclient.errors import HttpError
from oauth2client.client import GoogleCredentials

class BigQuery:
    def __init__(self, project_id):
        self.project_id = project_id
        self.credentials = GoogleCredentials.get_application_default()
        self.bigquery_service = build('bigquery', 'v2',
                                      credentials=self.credentials)

    def query(self, query_str):
        try:
            query_request = self.bigquery_service.jobs()

            query_data = {'query': query_str, 'timeoutMs': 60000}

            query_response = query_request.query(
                projectId=self.project_id,
                body=query_data).execute()

            return query_response

        except HttpError as err:
            print('Error: {}'.format(err.content))
            raise err

def format_response(response):
    fields = [f['name'] for f in response['schema']['fields']]
    rows = [[i['v'] for i in row['f']] for row in response['rows']]
    return dict(fields=fields, rows=rows)

if __name__ == '__main__':
    bq = BigQuery('nyc-taxi-trips-analysis')
    query_string = 'SELECT TOP(corpus, 10) as title, COUNT(*) as unique_words FROM [publicdata:samples.shakespeare];'
    print format_response(bq.query(query_string))
