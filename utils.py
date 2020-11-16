import cv2
import numpy as np
import json
import os
from sqlalchemy import create_engine, text

def point_perspective_transform(points, perspective_transform_matrix):
    """Function that makes a perspective transform to the given points using the given transformation matrix
        returns the transformed points"""

    if len(points)>0:
        #Computing the new coordinates
        points_to_transform = np.float32(points).reshape(-1,1,2)
        transformed_points = cv2.perspectiveTransform(points_to_transform, perspective_transform_matrix)   

        #Adding the transformed points to a list
        transformed_points_list = []
        for i in range(0,transformed_points.shape[0]):
            transformed_points_list.append([transformed_points[i][0][0], transformed_points[i][0][1]])

        return transformed_points_list
    else:
        return []

def save_json_file(data, json_output_path):
    """Function to save a JSON file to a filename path"""
    with open(json_output_path, 'w') as fp:
        json.dump(data, fp, indent= 4)
        print('JSON file successfully saved')

def load_json_file(json_path):
    """Function to load a JSON file from the filename path"""
    with open(json_path) as fp:
        data = json.load(fp)
        print('JSON file successfully loaded')
    return data

def get_db():
    #parameters
    host = "team-cv.cfsx82z4jthl.us-east-2.rds.amazonaws.com"
    user = "ds4a_69"
    port = "5432"
    password = <password>
    database = "postgres"
    
    # Create the engine with db credentials
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}', max_overflow = 20)
    return engine

def upload_to_db(df, table_name, if_exist_param='append'):
    """Function to Upload a dataframe to a database table"""

    # Create the engine with the db credentials
    engine = get_db()

    # uploading the data to the database
    df.to_sql(table_name, engine, if_exists = if_exist_param, index=False, method = 'multi')
    print('Data succesfully uploaded to the database')