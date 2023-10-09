from flask import Flask, jsonify, request, send_file
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import time
import os
from typing import List
from rdp import rdp
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import ttest_ind
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

executor = ThreadPoolExecutor(4)

global_epsilon = 1


def find_optimal_chunk_size(data):
    """
    Returns the number of chunks that the points will be divided to be processed in a parallel way
    """
    len_data = len(data)
    if len_data <= 1000:
        return 4
    elif len_data > 1000 and len_data <= 10000:
        return 16
    elif len_data > 10000 and len_data <= 100000:
        return 32
    elif len_data > 100000:
        return 64


def calculate_epsilon(data, global_mad):
    """
    Find an epsilon value for Ramer-Douglas-Peucker line simplification
    based on the median absolute deviation (MAD) of the data.
    """
    time_interval = 2  # determines the intensity of the change (1 = 100% maximum value for the best epsilon, 0.5 = 50%, 0.1 = 10%)
    data_range = np.max(data) - np.min(data)  # range of the data
    # multiplying the average of the MAD and range to the intensity of change to get the epsilon
    epsilon = (global_mad + data_range) / 2 * time_interval
    return epsilon


def calculate_chunk_epsilon(chunk_data):
    """
    Calculate a chunk-specific epsilon value based on the global epsilon value
    and the median absolute deviation (MAD) of the chunk.
    """
    time_interval = 1
    data_range = np.max(chunk_data) - np.min(chunk_data)

    chunk_mad = np.median(np.abs(chunk_data - np.median(chunk_data)))

    chunk_epsilon = (chunk_mad + data_range) / 2 * time_interval
    return chunk_epsilon


def classic_rdp(points, eps):
    """
    Returns the classic rdp result
    """
    res = rdp(points, epsilon=eps)
    return res


def parallel_rdp(points):
    """
    Returns the rdp result for every chunk
    """
    chunk_epsilon = calculate_chunk_epsilon([p[1] for p in points])
    future = executor.submit(rdp, points, epsilon=chunk_epsilon)
    result = future.result()
    return result


def parallel_rdp_algorithm(data: List[List[float]], epsilon: float, chunk_size: int = None) -> List[List[float]]:
    """
    This is the function where the process of running all the chunks of the original line will happen in a parallel way through the use of multiprocessing's threadpoolexecutor
    """

    # Create a thread pool with four threads
    executor = ThreadPoolExecutor(4)

    # Divide the data into chunks of size chunk_size (if specified)
    data_chunks = [data[i:i+chunk_size]
                   for i in range(0, len(data), chunk_size)]

    # Submit each chunk to the thread pool
    futures = [executor.submit(parallel_rdp, chunk)
               for chunk in data_chunks]

    # Wait for all threads to finish and collect the results
    results = [future.result() for future in futures]

    # Concatenate the results into a single list
    return [point for sublist in results for point in sublist]


def get_file_size(directory):
    """
    Gets the file size of the csv files in the /originals and /simplified folders
    """
    # return the new file size in KB
    file_size = os.path.getsize(directory) / 1024
    return file_size


def save_points_to_csv(points, filename, columns):
    """
    Turns the new parallelized rdp points into a csv file and saves it in the /simplified directory
    """
    try:
        col_1 = [point[0] for point in points]
        col_2 = [point[1] for point in points]

        # save the parallelized points into a new csv file
        directory = 'simplified/' + filename.split('.')[0] + '(simplified).csv'
        df = pd.DataFrame({columns[0]: col_1, columns[1]: col_2})
        df.to_csv(directory, index=False)
        return 1
    except:
        print("\nAn error occured during file saving.\n")
        return 0

# this function is for creating new CSV file for the simplified original CSV file


def createNewCSV(file, file_size, paralValue, df, return_val):
    global new_filename
    filename = file.filename
    df_simplified = pd.DataFrame(
        paralValue, columns=[df.columns[0], df.columns[1]])
    new_filename = filename.split('.')[0]+'(simplified).csv'
    df_simplified.to_csv(new_filename, index=False)

    new_file_size = os.path.getsize(new_filename)
    new_file_type = convert_bytes(new_file_size)

    diff_file_size = file_size - new_file_size
    diff_file_type = convert_bytes(diff_file_size)

    return_val.update({
        "new_file_name": new_filename,
        "new_file_size": new_file_size,
        "new_file_type": new_file_type,
        "diff_file_size": diff_file_size,
        "diff_file_type": diff_file_type,
    })


# this function is for getting file size label
def convert_bytes(num):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%s" % (x)
        num /= 1024.0


# simplify
@app.route('/api/simplify', methods=['POST'])
def trigger():
    return_val = {}

    try:
        # get file from api call
        file = request.files['file']

        file_size = os.fstat(file.fileno()).st_size
        file_type = convert_bytes(file_size)

        # read file using pandas
        df = pd.read_csv(file.stream, delimiter=',')

        # take the columns and rows
        cols = df.columns.values.tolist()
        first_row = df.iloc[:, 0]
        second_row = df.iloc[:, 1].astype(float)

        # list rows
        list_row_1 = first_row.values.tolist()
        list_row_2 = second_row.values.tolist()

        points = np.column_stack([range(len(first_row)), second_row])

        # get automatic epsilon value
        points_values = [p[1] for p in points]

        # global mad
        global_mad = np.median(
            np.abs(points_values - np.median(points_values)))  # MAD

        # global epsilon
        global_epsilon = calculate_epsilon(points_values, global_mad)

        # chunk size
        chunk = find_optimal_chunk_size(points)

        # parallel results
        parallelized_start_time = time.time()
        parallelized_points = parallel_rdp_algorithm(
            points, epsilon=global_epsilon, chunk_size=chunk)
        parallelized_end_time = time.time()
        parallelized_runtime = parallelized_end_time - parallelized_start_time

        # calculate mean
        original_mean = np.mean(np.mean(points, axis=0)[1])
        parallelized_mean = np.mean(np.mean(parallelized_points, axis=0)[1])

        # calculate standard deviation
        original_standard_deviation = np.std([p[1] for p in points])
        parallelized_standard_deviation = np.std(
            [p[1] for p in parallelized_points])

        # calculate the t statistic
        t_statistic, p_value = ttest_ind([point[1] for point in points], [
            point[1] for point in parallelized_points])

        first_row_rdp = [list_row_1[int(item[0])]
                         for item in parallelized_points]
        second_row_rdp = [list_row_2[int(item[0])]
                          for item in parallelized_points]

        # stacks the new first and second row for the simplified data
        paralValue = np.column_stack([first_row_rdp, second_row_rdp])

        list_row_1_rdp = []
        list_row_2_rdp = []

        counter = 0
        for item in first_row:
            if item in first_row_rdp:
                list_row_1_rdp.append(item)
                list_row_2_rdp.append(second_row[counter])
            else:
                list_row_1_rdp.append(None)
                list_row_2_rdp.append(None)
            counter += 1

        # return values
        return_val.update({
            "columns": cols,
            "row_1": list_row_1,
            "row_2": list_row_2,
            "row_1_rdp": list_row_1_rdp,
            "row_2_rdp": list_row_2_rdp,
            "file_type": file_type,
            "file_size": file_size,
            "t_statistic": t_statistic,
            "p_value": p_value,
            "std_original": original_standard_deviation,
            "std_rdp": parallelized_standard_deviation,
            "mean_original": original_mean,
            "mean_rdp": parallelized_mean,
            "runtime_parallelized": parallelized_runtime,
        })

        # write the simplified dataframe to a new csv file
        createNewCSV(file, file_size, paralValue, df, return_val)

        return return_val

    # Catch error if the CSV file is not a valid time series dataset
    except TypeError as e:
        return jsonify({'message': str(e)}), 400

    # Catch error if the uploaded CSV file is empty
    except pd.errors.EmptyDataError:
        return jsonify({'message': 'CSV file is empty!'}), 400

    # Catch any type of errors
    except Exception as e:
        return jsonify({'message': str(e)}), 400


@app.route('/download')
def download():
    # send the newly created csv file to the react client
    return send_file(new_filename, as_attachment=True)

# members api route


@app.route("/members")
def members():
    return {"members": ["Member1", "Member2", "Member3"]}


if __name__ == "__main__":
    app.run(debug=True)
