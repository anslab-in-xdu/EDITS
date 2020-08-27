import numpy as np

def preprocessing(column):
    column = list(filter(lambda x: (x != '0' and x != ''), column))
    column = np.array(list(map(float, column)))

    n = math.ceil(len(column) ** 0.5)

    data_lower = np.min(column)
    if data_lower < 0:
        data_lower = 0
    column = np.append(column, np.random.randint(data_lower, data_lower + 1, n * n - len(column)))

    # form
    column = column.reshape(n, int(len(column) / n))

    return column, n

if __name__ == '__main__':

	# input : dataset and column_y
	# ouput : data and its shape 
	DATA = {
	        # Example:
	        #   'dataset_name':'dataset_result'
	        # Description:
	        #   if the dataset result is specific, the dataset_result needs to be filled with the column name.
	        #   if the result column is at the end, just fill in 0.
	    }

	    for datasets in DATA.keys():
	        print(datasets)
	        if datasets == 'dataset_name':
	            column = cv2.imread('dataset_path', cv2.IMREAD_GRAYSCALE)
	            n = column.shape
	        else:
	            path = '../' + datasets + '.csv'
	            if datasets in ['dataset_result_1', 'dataset_result_2', 'dataset_result_n']:
	                with open(path, 'r') as csvfile:
	                    reader = csv.reader(csvfile)
	                    L = list(reader)
	                column = [x for j in L for x in j]
	            else:
	                with open(path, 'r') as csvfile:
	                    reader = csv.DictReader(csvfile)
	                    column = [row[DATA[datasets]] for row in reader]
	            column, n = preprocessing(column)

	        print(n)