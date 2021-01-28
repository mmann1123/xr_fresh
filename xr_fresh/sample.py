#%% minimum distance groups
# from scipy.spatial import distance

# data = XYna
# group = "rk_code"
# coords = ["x", "y"]
# #%%

# loc_groups = data.reset_index()[[group] + coords].groupby(["rk_code"]).mean()
# #%%


# def shuffle_array(data=None, random_state=12):
#     """
#     Random shuffle an array by rows

#     :param data: array of data, defaults to None
#     :type data: pd.DataFrame, optional
#     :param random_state: random seed, defaults to 12
#     :type random_state: int, optional
#     :return: shuffled data array
#     :rtype: pd.DataFrame
#     """
#     data = data.sample(frac=1, random_state=random_state)
#     return data[data.index.values]


# def create_rand_dist_array(
#     data=None, group=None, coords=["x", "y"], shuffle=True, random_state=12
# ):
#     """
#     Generate array of distances between coords aggregated by groups.

#     :param data: array of training data, defaults to None
#     :type data: pd.DataFrame, optional
#     :param group: column name to aggregate coords by, defaults to None
#     :type group: string, optional
#     :param coords: list of coordinates, defaults to ["x", "y"]
#     :type coords: list, optional
#     :param shuffle: randomize order of rows, defaults to True
#     :type shuffle: bool, optional
#     :param random_state: random seed, defaults to 12
#     :type random_state: int, optional
#     :return: array of distances between coords aggregated by groups
#     :rtype: [type]
#     """

#     loc_groups = data.reset_index()[[group] + coords].groupby([group]).mean()
#     loc_dist = pd.DataFrame(distance.cdist(loc_groups[coords], loc_groups[coords]),)

#     if shuffle:
#         loc_dist = shuffle_array(loc_dist, random_state=random_state)

#     return loc_dist


# def get_lower_diag(data):
#     """
#     Retrieve lower diagonal of array, replace upper and diagonal with NaN

#     :param data: NxN array
#     :type data: pd.DataFrame, np.Array
#     :return: lower diagonal of array
#     :rtype: np.Array
#     """
#     data[:] = np.tril(data)
#     data.replace(0.0, np.NaN, inplace=True)
#     return data


# loc_dist = create_rand_dist_array(data=XYna, group="rk_code", coords=["x", "y"])
# dist_data = get_lower_diag(loc_dist)
# dist_data

# #%%
# def drop_close_dist(dist_data, col_idx=0, min_dist=1):
#     """
#     Drop rows and columns of NxN distance.cdist array with distances less than min_dist

#     :param dist_data: [description]
#     :type dist_data: [type]
#     :param col_idx: [description], defaults to 0
#     :type col_idx: int, optional
#     :param min_dist: [description], defaults to 1
#     :type min_dist: int, optional
#     :return: NxN distance.cdist array with distances less than min_dist
#     :rtype: pd.DataTypes
#     """
#     drop_idx = dist_data.index[dist_data.iloc[:, col_idx] < min_dist]
#     dist_data = dist_data.drop(drop_idx, axis=0)
#     dist_data = dist_data.drop(drop_idx, axis=1)
#     return dist_data


# col = 0
# while True:
#     dist_data = drop_close_dist(dist_data, col_idx=col, min_dist=0.1)
#     col += 1
#     if col == dist_data.shape[1]:
#         break


#%% final min distance subset
# loc_group_subset = loc_groups.iloc[dist_data.index.values].reset_index()
# XYna = XYna[XYna["rk_code"].isin(loc_group_subset.rk_code)]
