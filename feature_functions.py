# -------------------------------------------------- #
# File: feature_functions
# -----------------------
# contains all feature extracting functions
# each one takes in a chat_dataframe and a user_dataframe
# -------------------------------------------------- #
#--- Standard ---
import math
from collections import Counter
from collections import defaultdict
from copy import copy

#--- Pandas ---
import pandas as pd

#-- Numpy/Scipy ---
import numpy as np
import scipy

#-- Networkx --
import networkx as nx

#--- My Files ---
from util import *

from matplotlib import pyplot

########################### CONSTANTS #######################################
PATH_LENGTH_CUTOFF = 5


####################################################################################################
#######################[ --- UTILS --- ]############################################################
####################################################################################################

# Function: get_user_ids
# ----------------------
# returns user1, user2 given a chat series
def get_user_ids (chat_series):
	
	return (chat_series['first_user_id'], chat_series['second_user_id'])


# Function: get_user_series
# -------------------------
# returns first_user_series, second_user_series given the chat_dataframe
def get_user_series (chat_series, user_dataframe):
	
	### Step 1: get their ids ###
	f_id, s_id = get_user_ids (chat_series)

	### Step 2: get the series out of the user dataframe ###
	return user_dataframe.loc[f_id], user_dataframe.loc[s_id]


####################################################################################################
#######################[ --- INDIVIDUAL USER FEATURES --- ]#########################################
####################################################################################################

# Function: get_weighted_topic_vectors_as_features
# ------------------------------------------------
# given a user dataframe, returns a num_users*num_topics mat, representing
# each user's interest in the topics
def get_weighted_topic_vectors_as_features (user_dataframe):

	features = []
	def add_topic_vector (x):
		features.append (x['weighted_topic_vector'])
	user_dataframe.apply (add_topic_vector, axis=1)
	return np.matrix (features)






####################################################################################################
#######################[ --- BASELINE --- ]#########################################################
####################################################################################################

# Function: get_baseline_features
# -------------------------------
# returns a list of features calculated objectively from
# the chats you are presented with
def get_baseline_features (chat_series, user_dataframe):

	features = []
	### Step 1: get the user series ###
	f_series, s_series = get_user_series (chat_series, user_dataframe)

	### Feature: age diff ###
	if f_series['age'] != 'None' and s_series['age'] != 'None':
		features.append (abs(int(f_series['age']) - int(s_series['age'])))
	else:
		features.append(-1)

	### Feature: Fraction of Disconnects ###
	f_disconnect_fraction = float(sum(f_series['disconnected_vector'])) / float(len(f_series['disconnected_vector']))
	s_disconnect_fraction = float(sum(s_series['disconnected_vector'])) / float(len(s_series['disconnected_vector']))
	features.append (abs(f_disconnect_fraction - s_disconnect_fraction))

	### Feature: fraction of lines in conversation ###
	f_participation_fraction = float(sum(f_series['lines_sent_vector'])) / float(sum(f_series['lines_sent_vector']) + sum(f_series['lines_received_vector']))
	s_participation_fraction = float(sum(s_series['lines_sent_vector'])) / float(sum(s_series['lines_sent_vector']) + sum(s_series['lines_received_vector']))
	features.append (abs(f_participation_fraction - s_participation_fraction))

	### Feature: boolean gender equivalence ###
	features.append (f_series['gender'] == s_series['gender'])

	return features


# Function: baseline_feature_extractor
# ------------------------------------
# returns a vector composed of the differnce between two users.
# see the guo et al paper on this for more details.
def baseline_feature_extractor (chat_dataframe, user_dataframe, removed_edge_graphs):

	feature_vectors = []

	def baseline_chat_features (chat_series):

		feature_vectors.append (np.array(get_baseline_features(chat_series, user_dataframe)))


	### Step 1: apply baseline_chat_features to each series (row) in the chat dataframe ###
	print_inner_status ('getting feature_vectors')
	chat_dataframe.apply (baseline_chat_features, axis=1)

	### Step 2: convert the list of feature vectors to a numpy matrix, return ###
	print_inner_status ('converting to numpy matrix')
	X = np.matrix (feature_vectors)
	return X






####################################################################################################
#######################[ --- TOPICS --- ]###########################################################
####################################################################################################

# Function: get_topic_features
# ----------------------------
# given a chat series, returns a list of features representing the 
# topics
def get_topic_features (chat_series, user_dataframe):
	
	### Step 1: get the user series, topic vectors ###
	f_series, s_series = get_user_series (chat_series, user_dataframe)
	f_vec = copy(f_series['weighted_topic_vector'])
	s_vec = s_series['weighted_topic_vector']

	### Step 2: add and return their absolute difference ###
	return abs(f_vec - s_vec)


# Function: topic_feature_extractor
# ---------------------------------
# returns a vector composed of the difference between the two users' 
# weighted topic vectors
def topic_feature_extractor (chat_dataframe, user_dataframe, removed_edge_graphs):

	feature_vectors = []

	def topic_features (chat_series):

		feature_vectors.append (get_topic_features(chat_series, user_dataframe))


	### Step 2: get features ###
	print_inner_status ("Extracting features for each chat")
	chat_dataframe.apply (topic_features, axis=1)

	### Step 3: convert to matrix, return ###
	print_inner_status ("Converting to matrix, returning")
	return np.matrix(feature_vectors)




####################################################################################################
#######################[ --- PERSONALITIES --- ]###########################################################
####################################################################################################

# Function: p_combo_to_index
# --------------------------
# given two personalities, returns the index of their attraction
def p_combo_to_index (f_cluster, s_cluster):
	if f_cluster >= s_cluster:
		return 10*f_cluster + s_cluster
	else:
		return 10*s_cluster + f_cluster


# Function: index_to_p_combo
# --------------------------
# given an index, returns the combo of the two personalities
def index_to_p_combo (index):

	return (index % 10, (index - (index % 10))/10)


# Function: get_personality_features
# ----------------------------------
# given a chat series, returns a list of personality features
def get_personality_features (chat_series, user_dataframe):

	### Step 1: get the user series, topic vectors ###
	f_series, s_series = get_user_series (chat_series, user_dataframe)
	f_cluster = f_series['personality_kmeans']
	s_cluster = s_series['personality_kmeans']

	### Step 2: add their absolute difference ###
	features = np.zeros (100)
	features[p_combo_to_index(f_cluster, s_cluster)] = 1
	feature_vectors.append (features)


# Function: personality_combo_feature_extractor
# -----------------------------------------------
# returns a one-hot vector, where the on element is the 
# product of the two cluster indeces
# NOTE: assumes that the number of personalities is 100....
def personality_combo_feature_extractor (chat_dataframe, user_dataframe):


	feature_vectors = scipy.sparse.csr_matrix ((len(chat_dataframe), 100))
	counter = 0

	def personality_combo_features (chat_series):

		### Step 1: get the user series, topic vectors ###
		f_series, s_series = get_user_series (chat_series, user_dataframe)
		f_cluster = f_series['personality_kmeans']
		s_cluster = s_series['personality_kmeans']

		### Step 2: add their absolute difference ###
		features = np.zeros (100)
		features[p_combo_to_index(f_cluster, s_cluster)] = 1
		feature_vectors.append (features)

	### Step 2: get features ###
	print_inner_status ("Extracting features for each chat")
	chat_dataframe.apply (personality_combo_features, axis=1)

	### Step 3: convert to matrix, return ###
	print_inner_status ("Converting to matrix, returning")
	return np.matrix(feature_vectors)






####################################################################################################
#######################[ --- NETWORKS --- ]#########################################################
####################################################################################################

# Function: network_feature_extractor
# -----------------------------------
# the baseline feature extractor with network-related features
# added on top
def network_feature_extractor (chat_dataframe, user_dataframe, removed_edge_graphs):

	feature_vectors = []

	#hist = defaultdict(lambda: 0)

	count = [0]
	# Function: baseline_chat_features
	# --------------------------------
	# given a chat series, this adds a numpy array to feature_vectors
	def network_features (chat_series):
		print count [0]
		feature_vectors.append (get_network_features(chat_series, removed_edge_graphs))
		count[0] += 1
		#hist[val] += 1

	### Step 1: apply baseline_chat_features to each series (row) in the chat dataframe ###
	chat_dataframe.apply (network_features, axis=1)

	#items = sorted(hist.items(), key=lambda l: l[1])
	#x = [cur[0] for cur in items]
	#y = [cur[1] for cur in items]

	#pyplot.plot(x, y, 'x')
	#pyplot.show()		

	### Step 2: convert the list of feature vectors to a numpy matrix, return ###
	return np.matrix (feature_vectors)

def get_network_features (chat_series, removed_edge_graphs):
		chat_features = []

		node1, node2 = get_user_ids(chat_series)
		### Feature: networkx feature - highest degree with path between the two people ###
		val = get_max_line_count_removed_path(node1, node2, removed_edge_graphs)
		chat_features.append (val)

		### Feature: networkx feature - long path score ###
		# score = get_long_walk_score(node1, node2, removed_edge_graphs[2])
		# chat_features.append (score)

		return np.array(chat_features)

# Function: get_max_line_count_removed_path
# --------------------------------
# Gets the max line count such that removing all edges with smaller line count still preserves
# a path between the two specified nodes. This is used as a feature for classification.
def get_max_line_count_removed_path(node1, node2, removed_edge_graphs):
	max_line_count = -1
	for index, graph in enumerate(removed_edge_graphs):
		edge_removed = False
		if graph.has_edge(node1, node2): 
			edge_removed = True # Remembering for later
			edge_data = graph[node1][node2]
			graph.remove_edge(node1, node2)

		if nx.has_path(graph, node1, node2):
			max_line_count = index
		else:
			if edge_removed: # Need to add it back
				graph.add_edge(node1, node2, attr_dict=edge_data)	
			break

	if max_line_count == -1: return max_line_count
	else:
		actual_sizes = [0,1,2,4,8,16,32,64,128] #TODO:move
		return actual_sizes[max_line_count]


# Function: get_long_walk_score
# -----------------------------------
# Does a long-walk between two nodes to attribute a score from 0-1 
# where 1 indicates that they share a path of users who enjoyed 
# speaking with each other and 0 indicates they did not (no info)
def get_long_walk_score(node1, node2, graph):
	
	edge_removed = False
	if graph.has_edge(node1, node2):
		edge_removed = True
		edge_data = graph[node1][node2]
		graph.remove_edge(node1, node2)

	all_simple_paths = nx.all_simple_paths(graph, source=node1, target=node2, cutoff=PATH_LENGTH_CUTOFF)
	
	output = 0
	num_paths = 0
	for path in all_simple_paths:
		path_score = 1
		for loc1,loc2 in zip(*(path[i:] for i in [0,1])): # for every connected pair
			dat = graph[loc1][loc2]
			num_lines = dat["first_user_num_lines"]+dat["second_user_num_lines"]
			path_score *= (1 - 1/float(num_lines+1) )
 		output = max(output,path_score)
		num_paths += 1
    
	if edge_removed:                                   
		graph.add_edge(node1, node2, attr_dict=edge_data)

	return output	








####################################################################################################
#######################[ --- BASELINE + TOPICS --- ]################################################
####################################################################################################

# Function: baseline_and_topics_extractor
# ---------------------------------------
# includes baseline features, topic features, network features
def all_features_extractor (chat_dataframe, user_dataframe, removed_edge_graphs):

	baseline_features   = baseline_feature_extractor 	(chat_dataframe, user_dataframe, removed_edge_graphs)
	topic_features      = topic_feature_extractor 		(chat_dataframe, user_dataframe, removed_edge_graphs)
	network_features 	= network_feature_extractor 	(chat_dataframe, user_dataframe, removed_edge_graphs)
	return np.concatenate ([baseline_features, topic_features, network_features], axis=1)


