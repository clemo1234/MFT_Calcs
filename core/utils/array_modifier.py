from utils import np

def dense_array(format, num_points, tol = 1e-3):
    pre_array = np.empty(len(format), dtype=type(np.array([])))
    try:
        format = np.array(format)
    except:
        TypeError("Input param not of valid type. Must be list or np.array")
    for idx, interval in enumerate(format):
        if len(interval) == 2:
            if idx == 0:
                pre_array[idx] = np.linspace(*interval, num_points[idx])
            else:
                pre_array[idx] = np.linspace(interval[0]+tol, interval[1], num_points[idx])
        else:
            raise ValueError("Incorrect param shape")
    array_temp = []
    for i in pre_array:
        for j in i:
            array_temp.append(j)
    
    return np.array(array_temp)
            
            
