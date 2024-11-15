import pandas as pd
import yaml 
import psycopg2 # biblioteca para conexão com o postgresql

def fetch_data_from_db(sql_query):
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        con = psycopg2.connect(
            dbname=config['database_config']['dbname'], 
            user=config['database_config']['user'], 
            password=config['database_config']['password'], 
            host=config['database_config']['host']
        )

        cursor = con.cursor()
        cursor.execute(sql_query)

        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'con' in locals():
            con.close()

    return df
                                                            



                