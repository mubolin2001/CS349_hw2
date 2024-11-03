import math
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr

'''
returns Euclidean distance between vectors a and b
'''
def euclidean(a,b):
    if len(a) != len(b):
        raise ValueError("Both vectors must have the same dimension")
    dist = 0
    for dimension in range(len(a)):
       dist += (b[dimension] - a[dimension]) ** 2
    dist = math.sqrt(dist)

    return(dist)
        
'''
returns Cosine Similarity between vectors and b
'''
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

'''
return pearson correlations of  a and b.
'''
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

'''
return hammering distance of two binary vectors.
'''
def hammering_distance(a, b):
    hammer_dist = 0
    for i in range(len(a)):
        hammer_dist += abs(a[i] - b[i])
    return hammer_dist

'''
returns a list of labels for the query dataset based upon labeled observations in the train dataset.
metric is a string specifying either "euclidean" or "cosim".  
All hyper-parameters should be hard-coded in the algorithm.
'''
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

'''
returns a list of labels for the query dataset based upon observations in the train dataset. 
labels should be ignored in the training set
metric is a string specifying either "euclidean" or "cosim".  
All hyper-parameters should be hard-coded in the algorithm.
'''
def kmeans(train, query, metric, num_clusters=5, max_iterations=100):
    # Select distance function based on metric
    if metric.lower() == "euclidean":
        dist_func = euclidean
    else:
        dist_func = cosim

    # Initialize centroids randomly from the training data
    train_data = np.array([list(map(int, t[1])) for t in train])
    centroids = train_data[np.random.choice(len(train_data), num_clusters, replace=False)]
    
    for _ in range(max_iterations):
        # Step 1: Assign each query point to the nearest centroid
        clusters = [[] for _ in range(num_clusters)]
        for point in train_data:
            distances = [dist_func(point, centroid) for centroid in centroids]
            nearest_centroid_idx = np.argmin(distances)
            clusters[nearest_centroid_idx].append(point)
        
        # Step 2: Update centroids by calculating the mean of assigned points
        new_centroids = []
        for cluster in clusters:
            if cluster:  # Avoid empty clusters
                new_centroid = np.mean(cluster, axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[np.random.choice(len(centroids))])
        
        # Check for convergence (if centroids do not change)
        if np.all([np.allclose(new, old) for new, old in zip(new_centroids, centroids)]):
            break
        centroids = new_centroids

    # Assign labels to query points based on nearest centroid
    labels = []
    for q in query:
        distances = [dist_func(list(map(int, q[1])), centroid) for centroid in centroids]
        nearest_centroid_idx = np.argmin(distances)
        labels.append(nearest_centroid_idx)

    return labels


'''
helper function: read file
'''
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

'''
helper function: read movielens.txt and other txts"
'''
def read_movie_data(file_name):
    data_set = []
    with open(file_name, 'rt') as f:
        # Skip the header line
        header = f.readline().strip().split('\t')
        
        for line in f:
            line = line.strip()
            tokens = line.split('\t')
            
            # Map tokens to column names for each row
            row_data = {header[i]: tokens[i] for i in range(len(header))}
            data_set.append(row_data)
    
    return data_set

'''
helper function: show files
'''
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
          
'''
helper function: load data using pd
'''
# Load dataset function
def load_data(file_path):
    return pd.read_csv(file_path, delimiter="\t")

# Example usage for loading datasets
train_a = load_data('train_a.txt')
train_b = load_data('train_b.txt')
train_c = load_data('train_c.txt')
movieLens = load_data('movielens.txt')
# Add other datasets as needed

def get_top_k_similar_users(target_user, other_users, k, metric='cosine'):
    similarities = []
    target_user_ratings_dict = dict(zip(target_user['movie_id'], target_user['rating']))
    
    for user_id, other_user in other_users.groupby('user_id'):
        other_user_ratings_dict = dict(zip(other_user['movie_id'], other_user['rating']))
        
        # Find common movies
        common_movies = set(target_user_ratings_dict.keys()).intersection(other_user_ratings_dict.keys())
        
        if common_movies:
            # Extract ratings for common movies
            target_ratings = [target_user_ratings_dict[movie] for movie in common_movies]
            other_ratings = [other_user_ratings_dict[movie] for movie in common_movies]
            
            # Calculate similarity based on the selected metric
            if metric == 'cosine':
                sim = cosim(target_ratings, other_ratings)
            elif metric == 'euclidean':
                sim = euclidean(target_ratings, other_ratings)
            elif metric == 'pearson':
                sim = pearson_correlation(target_ratings, other_ratings)
            similarities.append((user_id, sim))
    
    # Sort by similarity and select top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

def recommend_movies(target_user_id, other_users, top_k_users, threshold=4):
    recommendations = {}
    
    # Get ratings for similar users
    for user_id, _ in top_k_users:
        similar_user_ratings = other_users[other_users['user_id'] == user_id]
        for _, row in similar_user_ratings.iterrows():
            movie_id, movie_name, rating = row['movie_id'], row['title'], row['rating']
            
            # Only recommend highly-rated movies
            if rating >= threshold:
                if movie_id not in recommendations:
                    recommendations[movie_id] = {'name': movie_name, 'score': 0}
                
                # Accumulate the score based on the ratings from similar users
                recommendations[movie_id]['score'] += rating
    
    # Sort recommendations by the accumulated score from top_k similar users
    recommended_movies = sorted(recommendations.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Return the list of movie names and scores
    return [(movie_data['name'], movie_data['score']) for _, movie_data in recommended_movies]


def main():


    # show('mnist_train.csv','pixels')
    # print(knn(read_data("mnist_train.csv"), read_data("mnist_valid.csv"), "euclidean"))
    # labels=kmeans(read_data("mnist_train.csv"), read_data("mnist_test.csv"), "euclidean")
    # print("K-means cluster assignments:", labels)

    # x = [1, 5,9, 4, 7]
    # y = [2,10,25,28, 20]
    # print(hammering_distance(x, y))
    # print(read_movie_data("train_a.txt"))
    # print(user_similarity(read_data("train_a.txt"), 405 ))
    # Load training, validation, and test data
    
    # Load target user and other users
    target_user = train_b  # Replace 405 with desired user_id
    other_users = train_c
    # Get top K similar users
    top_k_users = get_top_k_similar_users(target_user, other_users, k=10, metric='cosine')
    
    # Get movie recommendations
    recommendations = recommend_movies(405, other_users, top_k_users)
    
    # Print recommendations
    print("Recommended movies for user 405:")
    for movie_name, score in recommendations:
        print(f"Movie name: {movie_name}, Score: {score}")
if __name__ == "__main__":
    main()
    