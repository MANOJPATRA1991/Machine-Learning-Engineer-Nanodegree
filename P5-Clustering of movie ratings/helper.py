import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.metrics import silhouette_samples, silhouette_score

def draw_scatterplot(x_data, x_label, y_data, y_label):
    # Figure size is 8 inches X 8 inches
    fig = plt.figure(figsize=(8,8))
    # The subplot will take the first position in a grid of 1 row and 1 column
    ax = fig.add_subplot(111)

    # Set the x-limit for the plot
    plt.xlim(0, 5)
    # Set the y-limit for the plot
    plt.ylim(0, 5)
    # Set the label for the x-axis
    ax.set_xlabel(x_label)
    # Set the label for the y-axis
    ax.set_ylabel(y_label)
    # A scatter plot of y_data vs. x_data with marker size of 30
    ax.scatter(x_data, y_data, s=30)


def draw_clusters(biased_dataset, predictions, cmap='viridis'):
    # Figure size is 8 inches X 8 inches
    fig = plt.figure(figsize=(8,8))
    # The subplot will take the first position in a grid of 1 row and 1 column
    ax = fig.add_subplot(111)
    # Set x-limit for plot
    plt.xlim(0, 5)
    # Set y-limit for plot
    plt.ylim(0, 5)
    # Set label for x-axis
    ax.set_xlabel('Avg scifi rating')
    # Set label for y-axis
    ax.set_ylabel('Avg romance rating')
    
    # 1. Add indices for biased_dataset as a column to the dataset
    # 2. Concatenate with data frame 'group' set to predictions along the row axis (or x-axis)
    clustered = pd.concat([biased_dataset.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
    
    # A scatter plot of clustered['avg_romance_rating'] vs. clustered['avg_scifi_rating'] with 
    # a) marker size of 20
    # b) c: A sequence of n numbers to be mapped to colors using cmap and norm
    # c) cmap: cmap is only used if c is an array of floats. If None, defaults to rc image.cmap.
    plt.scatter(clustered['avg_scifi_rating'], clustered['avg_romance_rating'], c=clustered['group'], s=20, cmap=cmap)

        
def clustering_errors(k, data):
    # fit(data) and then predict(data) is the same as fit_predict(da
    kmeans = KMeans(n_clusters=k).fit(data)
    predictions = kmeans.predict(data)
    #cluster_centers = kmeans.cluster_centers_
    # errors = [mean_squared_error(row, cluster_centers[cluster]) for row, cluster in zip(data.values, predictions)]
    # return sum(errors)
    
    # The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of. Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.
    
    # The best value is 1 and the worst value is -1. 
    # Values near 0 indicate overlapping clusters. 
    # Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
    silhouette_avg = silhouette_score(data, predictions)
    return silhouette_avg

def sparse_clustering_errors(k, data):
    kmeans = KMeans(n_clusters=k).fit(data)
    predictions = kmeans.predict(data)
    cluster_centers = kmeans.cluster_centers_
    errors = [mean_squared_error(row, cluster_centers[cluster]) for row, cluster in zip(data, predictions)]
    return sum(errors)


def get_genre_ratings(ratings, movies, genres, column_names):
    # Two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns)
    genre_ratings = pd.DataFrame()
    for genre in genres:        
        # Get all movies with 'genres' column containing the given genre
        genre_movies = movies[movies['genres'].str.contains(genre)]
        # 1. Filter movies from ratings collection based on whether the 'movieId' is present in the genre_movies collection
        # 2. Get columns 'userId' and 'rating'
        # 3. Group the above data by 'userId' column
        # 4. Find the mean of the 'ratings' column and round to two decimal digits
        avg_genre_votes_per_user = ratings[ratings['movieId'].isin(genre_movies['movieId'])].loc[:, ['userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)
        # Concatenate the data along axis=1 with the genre_ratings dataframe
        genre_ratings = pd.concat([genre_ratings, avg_genre_votes_per_user], axis=1)
        
    print(genre_ratings)
    # Assign column names to the genre_ratings dataframe
    genre_ratings.columns = column_names
    return genre_ratings
    
def get_dataset_3(movies, ratings, genre_ratings):    
    # Extract action ratings from dataset
    action_movies = movies[movies['genres'].str.contains('Action') ]
    # Get average vote on action movies per user
    avg_action_votes_per_user = ratings[ratings['movieId'].isin(action_movies['movieId'])].loc[:, ['userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)
    # Add action ratings to romance and scifi in dataframe
    genre_ratings_3 = pd.concat([genre_ratings, avg_action_votes_per_user], axis=1)
    genre_ratings_3.columns = ['avg_romance_rating', 'avg_scifi_rating', 'avg_action_rating']
    
    # Let's bias the dataset a little so our clusters can separate scifi vs romance more easily
    b1 = 3.2
    b2 = 2.5
    biased_dataset_3 = genre_ratings_3[((genre_ratings_3['avg_romance_rating'] < b1 - 0.2) & (genre_ratings_3['avg_scifi_rating'] > b2)) | ((genre_ratings_3['avg_scifi_rating'] < b1) & (genre_ratings_3['avg_romance_rating'] > b2))]
    biased_dataset_3 = pd.concat([biased_dataset_3[:300], genre_ratings_3[:2]])
    biased_dataset_3 = pd.DataFrame(biased_dataset_3.to_records())
    
    return biased_dataset_3

def draw_clusters_3d(biased_dataset_3, predictions):
    # Figure size is 8 inches X 8 inches
    fig = plt.figure(figsize=(8,8))
    # The subplot will take the first position in a grid of 1 row and 1 column
    ax = fig.add_subplot(111)
    # Set x-limit for plot
    plt.xlim(0, 5)
    # Set y-limit for plot
    plt.ylim(0, 5)
    # Set label for x-axis
    ax.set_xlabel('Avg scifi rating')
    # Set label for y-axis
    ax.set_ylabel('Avg romance rating')

    # 1. Add indices for biased_dataset_3 as a column to the dataset
    # 2. Concatenate with data frame 'group' set to predictions along the row axis (or x-axis)
    clustered = pd.concat([biased_dataset_3.reset_index(), pd.DataFrame({'group':predictions})], axis=1)

    # itertools.cylce: Make an iterator returning color codes from the iterable and save a copy of each.
    # Runs indefinitely
    
    # print(plt.rcParams["axes.prop_cycle"])
    # cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    
    # print(plt.rcParams["axes.prop_cycle"].by_key())
    # {'color': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']}
    
    # print(plt.rcParams["axes.prop_cycle"].by_key()['color'])
    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    # Iterate through sorted unique elements of group column
    for g in clustered.group.unique():
        color = next(colors)
        for index, point in clustered[clustered.group == g].iterrows():
            # Identify avg_actio_rating by size of markers
            if point['avg_action_rating'].astype(float) > 3: 
                size = 50
            else:
                size = 15
            plt.scatter(point['avg_scifi_rating'], 
                        point['avg_romance_rating'], 
                        s=size, 
                        color=color)
    
def draw_movie_clusters(clustered, max_users, max_movies):
    c=1
    for cluster_id in clustered.group.unique():
        # To improve visibility, we're showing at most max_users users and max_movies movies per cluster.
        # You can change these values to see more users & movies per cluster
        d = clustered[clustered.group == cluster_id].drop(['index', 'group'], axis=1)
        n_users_in_cluster = d.shape[0]
        
        d = sort_by_rating_density(d, max_movies, max_users)
        
        d = d.reindex_axis(d.mean().sort_values(ascending=False).index, axis=1)
        d = d.reindex_axis(d.count(axis=1).sort_values(ascending=False).index)
        d = d.iloc[:max_users, :max_movies]
        n_users_in_plot = d.shape[0]
        
        # We're only selecting to show clusters that have more than 9 users, otherwise, they're less interesting
        if len(d) > 9:
            print('cluster # {}'.format(cluster_id))
            print('# of users in cluster: {}.'.format(n_users_in_cluster), '# of users in plot: {}'.format(n_users_in_plot))
            fig = plt.figure(figsize=(15,4))
            ax = plt.gca()

            ax.invert_yaxis()
            ax.xaxis.tick_top()
            labels = d.columns.str[:40]

            ax.set_yticks(np.arange(d.shape[0]) , minor=False)
            ax.set_xticks(np.arange(d.shape[1]) , minor=False)

            ax.set_xticklabels(labels, minor=False)
                        
            ax.get_yaxis().set_visible(False)

            # Heatmap
            heatmap = plt.imshow(d, vmin=0, vmax=5, aspect='auto')

            ax.set_xlabel('movies')
            ax.set_ylabel('User id')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            # Color bar
            cbar = fig.colorbar(heatmap, ticks=[5, 4, 3, 2, 1, 0], cax=cax)
            cbar.ax.set_yticklabels(['5 stars', '4 stars','3 stars','2 stars','1 stars','0 stars'])

            plt.setp(ax.get_xticklabels(), rotation=90, fontsize=9)
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', labelbottom='off', labelleft='off') 
            #print('cluster # {} \n(Showing at most {} users and {} movies)'.format(cluster_id, max_users, max_movies))

            plt.show()


            # Let's only show 5 clusters
            # Remove the next three lines if you want to see all the clusters
            # Contribution welcomed: Pythonic way of achieving this
            # c = c+1
            # if c > 6:
            #    break
                
def get_most_rated_movies(user_movie_ratings, max_number_of_movies):
    # 1- Count
    # Append count of user ratings for each movie to the end of this frame and ignore index labels
    user_movie_ratings = user_movie_ratings.append(user_movie_ratings.count(), ignore_index=True)
    
    # 2- sort
    # Sort based on length of user_movie_ratings along the columns in descending order
    user_movie_ratings_sorted = user_movie_ratings.sort_values(len(user_movie_ratings)-1, axis=1, ascending=False)
    
    # Delete the last row
    user_movie_ratings_sorted = user_movie_ratings_sorted.drop(user_movie_ratings_sorted.tail(1).index)
    
    # 3- slice
    # Restrict number of columns to max_number_of_movies
    most_rated_movies = user_movie_ratings_sorted.iloc[:, :max_number_of_movies]
    return most_rated_movies

def get_users_who_rate_the_most(most_rated_movies, max_number_of_users):
    # Get most voting users
    # 1- Count
    # Count non-NA cells for each user(axis=1: count along columns)
    most_rated_movies['counts'] = pd.Series(most_rated_movies.count(axis=1))
    
    # 2- Sort
    # Sort users in descending order of counts as calculated above
    most_rated_movies_users = most_rated_movies.sort_values('counts', ascending=False)
    
    # 3- Slice
    # Restrict number of rows to max_number-of_users
    most_rated_movies_users_selection = most_rated_movies_users.iloc[:max_number_of_users, :]
    most_rated_movies_users_selection = most_rated_movies_users_selection.drop(['counts'], axis=1)
    
    return most_rated_movies_users_selection

def sort_by_rating_density(user_movie_ratings, n_movies, n_users):
    most_rated_movies = get_most_rated_movies(user_movie_ratings, n_movies)
    most_rated_movies = get_users_who_rate_the_most(most_rated_movies, n_users)
    return most_rated_movies
    
def draw_movies_heatmap(most_rated_movies_users_selection, axis_labels=True):
    
    # Reverse to match the order of the printed dataframe
    #most_rated_movies_users_selection = most_rated_movies_users_selection.iloc[::-1]
    
    # Create a figure of width 15 inches and height 4 inches
    fig = plt.figure(figsize=(15,4))
    
    # Gets the current axes, creating one if needed. It is only equivalent in the simplest 1 axes case.
    ax = plt.gca()
    
    # Draw heatmap
    # imshow --> Display an image, i.e, data on a 2D regular raster
    #
    # vmin and vmax cover the data range that colormap covers
    #
    #
    # interpolation='nearest'--> simply displays an image without trying to interpolate between pixels if the display resolution is not the same as the image resolution (which is most often the case). It will result an image in which pixels are displayed as a square of multiple pixels.
    #
    # aspect --> controls the aspect ratio of the axes
    ### Two options available for aspect:
    ### 1. equal: Ensures an aspect ratio of 1. Pixels will be square (unless pixel sizes are explicitly made non-square in data coordinates using extent).
    ### 2. auto: The axes is kept fixed and the aspect is adjusted so that the data fit in the axes. In general, this will result in non-square pixels.
    heatmap = ax.imshow(most_rated_movies_users_selection,  interpolation='nearest', vmin=0, vmax=5, aspect='auto')

    if axis_labels:
        # Set required number of y-ticks
        ax.set_yticks(np.arange(most_rated_movies_users_selection.shape[0]) , minor=False)
        # Set required number of x-ticks
        ax.set_xticks(np.arange(most_rated_movies_users_selection.shape[1]) , minor=False)
        # Invert the y-axis
        ax.invert_yaxis()
        # Place x-axis tick marks at the top of the image
        ax.xaxis.tick_top()
        labels = most_rated_movies_users_selection.columns.str[:40]
        # Set x-tick labels
        ax.set_xticklabels(labels, minor=False)
        # Set y-tick labels
        ax.set_yticklabels(most_rated_movies_users_selection.index, minor=False)
        # Rotate x-tick labels by 90 degrees
        plt.setp(ax.get_xticklabels(), rotation=90)
    else:
        # Hide the axes
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    ax.grid(False)
    ax.set_ylabel('User id')

    # Separate heatmap from color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Color bar
    cbar = fig.colorbar(heatmap, ticks=[5, 4, 3, 2, 1, 0], cax=cax)
    cbar.ax.set_yticklabels(['5 stars', '4 stars','3 stars','2 stars','1 stars','0 stars'])



    plt.show()
    
def bias_genre_rating_dataset(genre_ratings, score_limit_1, score_limit_2):
    '''
    This function filters out all users who game same ratings to both genres: romance and sci-fi
    '''
    biased_dataset = genre_ratings[((genre_ratings['avg_romance_rating'] < score_limit_1 - 0.2) & (genre_ratings['avg_scifi_rating'] > score_limit_2)) | ((genre_ratings['avg_scifi_rating'] < score_limit_1) & (genre_ratings['avg_romance_rating'] > score_limit_2))]
    biased_dataset = pd.concat([biased_dataset[:300], genre_ratings[:2]])
    biased_dataset = pd.DataFrame(biased_dataset.to_records())
    return biased_dataset