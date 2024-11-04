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
    try:
        dist = numerator / denominator
    except ZeroDivisionError:
        dist = 999
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
def preprocess_data(data, n_components=50):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    normalized_data = normalize(reduced_data, norm='l2')
    return normalized_data

def kmeans(train, query, metric='euclidean', num_clusters=10, max_iterations=100, tol=1e-4):
    if metric.lower() == "euclidean":
        dist_func = euclidean
    else:
        dist_func = cosim

    train_data = np.array([list(map(int, t[1])) for t in train])
    train_data = preprocess_data(train_data)  # Apply PCA and normalization

    centroids = train_data[np.random.choice(len(train_data), num_clusters, replace=False)]
    
    for _ in range(max_iterations):
        clusters = [[] for _ in range(num_clusters)]
        for point in train_data:
            distances = [dist_func(point, centroid) for centroid in centroids]
            nearest_centroid_idx = np.argmin(distances)
            clusters[nearest_centroid_idx].append(point)
        
        new_centroids = []
        for cluster in clusters:
            if cluster:  
                new_centroid = np.mean(cluster, axis=0)
                new_centroids.append(new_centroid)
            else:
                all_distances = np.array([np.min([dist_func(point, c) for c in centroids]) for point in train_data])
                farthest_point = train_data[np.argmax(all_distances)]
                new_centroids.append(farthest_point)
        
        centroid_shifts = [np.linalg.norm(new - old) for new, old in zip(new_centroids, centroids)]
        if np.all(np.array(centroid_shifts) < tol):
            break
        centroids = new_centroids

    query_data = np.array([list(map(int, q[1])) for q in query])
    query_data = preprocess_data(query_data) 

    labels = []
    for q in query_data:
        distances = [dist_func(q, centroid) for centroid in centroids]
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
test_a = load_data('test_a.txt')
test_b = load_data('test_b.txt')
test_c = load_data('test_c.txt')
valid_a = load_data('valid_a.txt')
valid_b = load_data('valid_b.txt')
valid_c = load_data('valid_c.txt')
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
            try:
                similarities.append((user_id, sim))
            except UnboundLocalError:
                similarities.append((user_id,0))
    
    # Sort by similarity and select top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

def d_get_top_k_similar_users(target_user, other_users, k, metric='cosine'):
    target_user = target_user.copy()
    other_users = other_users.copy()
    similarities = []
    all_occupations = list(other_users.groupby('occupation').groups.keys())

    #Normalize dataset
    other_users['occupation'] = other_users['occupation'].apply(lambda x: all_occupations.index(x)/len(other_users))
    other_users['age'] = other_users['age'].apply(lambda x: x/len(other_users))
    other_users['rating'] = other_users['rating'].apply(lambda x: x/len(other_users))
    other_users['gender'] = other_users['gender'].apply(lambda x: 1/len(other_users) if x == "M" else 2/len(other_users))

    target_user['occupation'] = target_user['occupation'].apply(lambda x: all_occupations.index(x)/len(other_users))
    target_user['age'] = target_user['age'].apply(lambda x: x/len(other_users))
    target_user['rating'] = target_user['rating'].apply(lambda x: x/len(other_users))
    target_user['gender'] = target_user['gender'].apply(lambda x: 1/len(other_users) if x == "M" else 2/len(other_users))

    target_user_ratings_dict = dict(zip(target_user['movie_id'], np.array(target_user['rating'])))
    target_occupation = target_user.loc[0]['occupation']
    target_age = target_user.loc[0]['age']
    target_gender = target_user.loc[0]['gender']
    


    for user_id, other_user in other_users.groupby(['user_id', 'occupation', 'gender', 'age']):
        _, other_occupation, other_gender, other_age = user_id
        other_vector = [other_occupation,other_age,other_gender]
        target_vector = [target_occupation, target_age, target_gender]
        other_user_ratings_dict = dict(zip(other_user['movie_id'], other_user['rating']))
        
        # Find common movies
        common_movies = set(target_user_ratings_dict.keys()).intersection(other_user_ratings_dict.keys())
        
        if common_movies:
            # Extract ratings for common movies
            target_vector += [target_user_ratings_dict[movie] for movie in common_movies]
            other_vector += [other_user_ratings_dict[movie] for movie in common_movies]
            
            # Calculate similarity based on the selected metric
            if metric == 'cosine':
                sim = cosim(target_vector, other_vector)
            elif metric == 'euclidean':
                sim = euclidean(target_vector, other_vector)
            elif metric == 'pearson':
                sim = pearson_correlation(target_vector, other_vector)

            try:
                similarities.append((user_id[0], sim))
            except UnboundLocalError:
                similarities.append((user_id[0], 0))

        
    
    # Sort by similarity and select top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]
   

def recommend_movies(other_users, top_k_users, threshold = 4):
    recommendations = {}
    
    # Get ratings for similar users
    for user_id, _ in top_k_users:
        similar_user_ratings = other_users[other_users['user_id'] == user_id]
        for _, row in similar_user_ratings.iterrows():
            movie_id, movie_name, rating = row['movie_id'], row['title'], row['rating']
            
            # Only recommend highly-rated movies
            if rating >= threshold:
                if movie_id not in recommendations:
                    recommendations[movie_id] = {'name': movie_name, 'score': []}
                        
                            # Accumulate the score based on the ratings from similar users
                            # ZW: Changed this to average rating of the similar users
                recommendations[movie_id]['score'] += [rating]
            
    
    # Sort recommendations by the accumulated score from top_k similar users
    recommended_movies = dict(sorted(recommendations.items(), key=lambda x: x[1]['score'], reverse=True))

    # Return the list of movie names and scores
    #return [(movie_data['name'], int(np.rint(np.array(movie_data['score']).mean()))) for _, movie_data in recommended_movies]
    return recommended_movies



def evaluate(train_set, test_set, other_users, k, mode = "mean", metric = "cosine", demographic = True, threshold = 4):
    if demographic == True:
        top_k_users = d_get_top_k_similar_users(train_set, other_users, k=k, metric=metric)
    else:
        top_k_users = get_top_k_similar_users(train_set, other_users, k=k, metric=metric)
    recommendations = recommend_movies(other_users, top_k_users, threshold)
    test_set_movie_ids = set(test_set['movie_id'])
    ratings = []

    for _ in test_set_movie_ids:
        if _ in recommendations:
            if mode == "mean":
                rating = np.rint(np.array(recommendations[_]['score']).mean()).astype(int)
            if mode == "max":
                rating = np.rint(np.array(recommendations[_]['score']).max()).astype(int)
            if mode == "min":
                rating = np.rint(np.array(recommendations[_]['score']).min()).astype(int)
            target_rating = test_set[test_set['movie_id'] == _]['rating'].values[0]
            ratings += [(recommendations[_]['name'], rating, target_rating)]

    ratings.sort(key = lambda x:x[1], reverse=True)
    TP = 0
    FP = 0
    FN = 0
    for _ in ratings:
        if _[1]>=4 & _[2]>=4:
            TP += 1
        if _[1]>=4 & _[2]<4:
            FP += 1
        if _[1]<4 & _[2]>=4:
            FN += 1
  
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)  
    f1 = (2*precision*recall)/(precision+recall)
    return (precision, recall, f1, ratings)

    


def main():


    # show('mnist_train.csv','pixels')
    # print(knn(read_data("mnist_train.csv"), read_data("mnist_valid.csv"), "euclidean"))
    labels=kmeans(read_data("mnist_train.csv"), read_data("mnist_test.csv"), "euclidean")
    print("K-means clsuster assignments:", labels)

    # x = [1, 5,9, 4, 7]
    # y = [2,10,25,28, 20]
    # print(hammering_distance(x, y))
    # print(read_movie_data("train_a.txt"))
    # print(user_similarity(read_data("train_a.txt"), 405 ))
    # Load training, validation, and test data
    
    # Load target user and other users
    target_user = train_a  # Replace 405 with desired user_id
    other_users = movieLens
    # Get top K similar users
    #top_k_users = d_get_top_k_similar_users(target_user, other_users, k=5, metric='cosine')

    # Get movie recommendations
    #recommendations = recommend_movies(other_users, top_k_users, threshold=1)
    
    # Print recommendations
    #print("Recommended movies for user 405:")
    #for movie_name, score in recommendations:
        #print(f"Movie name: {movie_name}, Score: {score}")
    # precision, recall, f1, ratings = evaluate(train_a, test_a, movieLens, k = 5,  mode = "max", demographic= True, metric="pearson", threshold = 4)
    # print("Precision = ", precision, "Recall = ", recall, "F1 = ", f1)
    # print(ratings)

if __name__ == "__main__":
    main()
    