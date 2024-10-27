import math
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
# returns Euclidean distance between vectors and b
def euclidean(a,b):
    if len(a) != len(b):
        raise ValueError("Both vectors must have the same dimension")
    dist = 0
    for dimension in range(len(a)):
       dist += (b[dimension] - a[dimension]) ** 2
    dist = math.sqrt(dist)

    return(dist)
        
# returns Cosine Similarity between vectors and b
def cosim(a,b):
    if len(a) != len(b):
        raise ValueError("Both vectors must have the same dimension")
    dot_product = 0
    magnitude_a = 0
    magnitude_b = 0
    for dimension in range(len(a)):
        dot_product += a[dimension] * b[dimension]
        magnitude_a += a[dimension] ** 2
        magnitude_b += b[dimension] ** 2
    magnitude_a = math.sqrt(magnitude_a)
    magnitude_b = math.sqrt(magnitude_b)
    dist = dot_product / magnitude_b / magnitude_a
    return(dist)
# return pearson correlations of  a and b.
def pearson_correlation(a, b):
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    numerator = 0
    denominator_x = 0
    denominator_y = 0
    for i in range(len(a)):
        numerator += (a[i] - mean_a) * (b[i] - mean_b)
        denominator_x += (a[i] - mean_a) ** 2
        denominator_y += (b[i] - mean_b) ** 2
    denominator_y = math.sqrt(denominator_y)
    denominator_x = math.sqrt(denominator_x)
    denominator = denominator_y * denominator_x
    dist = numerator / denominator
    return dist
#return hammering distance of two binary vectors.
def hammering_distance(a, b):
    return (sum(math.abs(a[i] - b[i])) for i in range(len(a)))

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metri, K = 5, n_comp = 50):
    if metri.lower() == "euclidean":
        dist_func = euclidean
    else:
        dist_func = cosim
    labels = []
    for idx in range(len(train)):
        train[idx] = [int(train[idx][0]), [int(item) for item in train[idx][1]]]
    # convert all string type data to integer
    train_data = np.array([list(map(int, t[1])) for t in train])
    train_label = np.array([int(t[0]) for t in train])
    query_data = np.array([list(map(int, q[1])) for q in query])

     # Apply PCA to reduce dimensionality
    pca = PCA(n_components = n_comp)
    train_data_reduced = pca.fit_transform(train_data)
    query_data_reduced = pca.transform(query_data)

    for q in query_data_reduced:
        # for each data in query, compute distance from each train dataset.
        distances_from_train =[dist_func(t, q) for t in train_data_reduced]
        # sorted the indices of train data by the distance.
        k_nearest_index = sorted(range(len(distances_from_train)), key = lambda i : distances_from_train[i])[:K]
        # find the nearest label based on the index found.
        k_nearest_labels = [train_label[index] for index in k_nearest_index]
        # use Counter function from collection package to predict the most possible label by majority vote.
        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
        labels.append(most_common_label)

    return(labels)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    return(labels)

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    # show('mnist_train.csv','pixels')
    print(knn(read_data("mnist_train.csv"), read_data("mnist_valid.csv"), "euclidean"))
    # x = [1, 5,9, 4, 7]
    # y = [2,10,25,28, 20]
    # print(pearson_correlation(x, y ))
    
if __name__ == "__main__":
    main()
    