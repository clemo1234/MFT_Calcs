import numpy as np

from decimal import Decimal


def triple_error(index_t, betas, gammas):
    if index_t >= 1:
        neighbourhood = [index_t - 1, index_t + 1]
    else:
        neighbourhood = [index_t + 1]

    vecs = []

    search_window = [index_t, *neighbourhood]


    for s in search_window:
        try:
            if np.shape(gammas[s])[0] > 0:
                for i in range(2):
                    vecs.append(np.array([betas[s], gammas[s][i]]))
            else:
                vecs.append(np.array([betas[s], gammas[s]]))
        except:
            vecs.append(np.array([betas[s], gammas[s]]))

    dists = []
    try:
        if np.shape(gammas[index_t])[0] > 0:
            central_vec = (vecs[0] + vecs[1])/2 
            new_vec = vecs[2:]
            for v in new_vec:
                dists.append(np.dot(central_vec, v))

            idx = np.argmin(dists)
            
            error_t = np.abs(central_vec - new_vec[idx])
            print("good")
        else:
            new_vec = vecs[1:]
            for v in new_vec:
                dists.append(np.dot(vecs[0], v))

            idx = np.argmin(dists)
            
            error_t = np.abs(vecs[0] - new_vec[idx])
    except:
        new_vec = vecs[1:]
        for v in new_vec:
            dists.append(np.dot(vecs[0], v))

        idx = np.argmin(dists)
        
        error_t = np.abs(vecs[0] - new_vec[idx])
            
    

    return error_t

def float_to_int_digits(x):
    d = Decimal(str(x))      # preserve written decimal form
    places = -d.as_tuple().exponent
    return int(d * (10 ** places)), places


        

    

    