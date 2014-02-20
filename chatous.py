#!/usr/bin/python
# -------------------------------------------------- #
# File: chatous
# -------------
# main class, all ML
# -------------------------------------------------- #
#--- Standard ---
import os
import sys
from collections import defaultdict
from collections import Counter
import csv
import argparse
import math
import pickle
from itertools import chain
from random import sample

#--- json ---
import json
import math

#--- pandas ---
import pandas

#--- sklearn ---
import numpy as np 
import sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model
from sklearn import cross_validation

#--- networkx ----
import networkx as nx

#--- gensim ---
from gensim import corpora, models

#--- plotting ---
# import matplotlib.pyplot as pyplot

#--- Our Files ---
from util import *
from feature_functions import *
from label_functions import *



# Class: Chatous
# --------------
# contains all data and procedures relevant to cs224w project on
# chatous
class Chatous:

	####################################################################################################
	#######################[ --- DATA --- ]#############################################################
	####################################################################################################

	#==========[ FILENAMES ]==========
	filenames = {}	
	data_directory 					= os.path.join(os.getcwd(), '../data/')
	filenames ['profiles'] 			= os.path.join(data_directory, 'profiles.csv')
	filenames ['chats'] 			= os.path.join(data_directory, 'chats.csv')
	filenames ['saved_states'] 		= os.path.join (data_directory, 'saved_states')
	filenames ['stopwords_list'] 	= os.path.join (data_directory, 'stopwords_list.obj')

	#==========[ HYPERPARAMETERS ]==========
	num_chats 					= -1
	num_profiles 				= -1
	num_users 					= -1
	num_personality_clusters 	= 10

	#==========[ DATAFRAMES ]==========
	chat_dataframe 		= None
	profile_dataframe 	= None
	user_dataframe 		= None

	#==========[ NETWORKS ]==========
	graph 				= None
	REMOVED_EDGE_RANGE 	= [0, 1, 2, 4, 8, 16, 32, 64, 128]
	removed_edge_graphs = []	

	#==========[ CLASSIFICATION ]==========
	kmeans 				= None
	personality_GMM 	= None
	text_collection 	= None

	#==========[ EXPLORATION ]==========
	histo 	= Counter()
	vals 	= []







	####################################################################################################
	#######################[ --- CONSTRUCTOR/INITIALIZATION --- ]#######################################
	####################################################################################################

	# Function: constructor
	# ---------------------
	# loads in the data 
	def __init__ (self, state_name=None, num_chats=10000, num_profiles=332888, num_personalities=10, num_topics=100, preloaded_dataframe=None):

		#============[ --- INTERFACE --- ]=============
		print_welcome ()


		#============[ --- LOAD STATE --- ]=============
		print_header ("Loading State")
		if state_name:
			self.load_state (state_name)
			return

		#============[ --- PARAMETERS --- ]=============
		self.num_chats 			= num_chats
		self.num_profiles 		= num_profiles
		self.num_personalities 	= num_personalities
		self.num_topics			= num_topics
		print_parameters (self)


		#============[ --- NLP SETUP --- ]=========
		print_status ("NLP Initialization", "loading stopwords")
		self.load_stopwords ()


		#============[ --- GET PROFILE DATA --- ]=============
		print_header ("Loading/Initializing Profile Data")
		self.get_profile_data ()


		#============[ --- GET CHAT DATA --- ]=============
		print_header ("Loading/Initializing Chat Data")
		self.get_chat_data ()


		#============[ --- NETWORK CREATION --- ]============
		print_header ("Creating Networkx Model")
		self.create_network()


		#============[ --- GET USER DATA --- ]=============
		print_header ("Loading/Initializing User Data")
		self.get_user_data ()


		#============[ --- Topic Modelling --- ]=========
		# print_header ("Performing Topic Modeling")
		# self.perform_topic_modelling ()


		

	####################################################################################################
	#######################[ --- LOAD/SAVE STATE --- ]##################################################
	####################################################################################################

	# Function: state_name_to_directory
	# ---------------------------------
	# given a state name (i.e. all_chats_all_profiles), returns the name
	# of a directory to save it in (after making sure the directory exists)
	def state_name_to_directory (self, state_name):

		return os.path.join (self.filenames['saved_states'], state_name)


	# Function: save_state
	# --------------------
	# saves the state of the current program - includes all dataframes
	def save_state (self, state_name):

		### Step 1: get the name, make sure it exists ###
		dir_name = self.state_name_to_directory (state_name)
		if not os.path.isdir (dir_name):
			os.path.mkdir (dir_name)

		### Step 2: save dataframes ###
		print_status ("Save State", "Saving chat dataframe")
		self.chat_dataframe.to_pickle (os.path.join (dir_name, 'chat.dataframe'))
		print_status ("Save State", "Saving user dataframe")
		self.user_dataframe.to_pickle (os.path.join (dir_name, 'user.dataframe'))


	# Function: load_state 
	# --------------------
	# given a state name (see save_state, above), 
	# this will load in all of the relevant dataframes
	def load_state (self, state_name):

		### Step 1: get directory name, make sure it exists ###
		dir_name = self.state_name_to_directory (state_name)
		if not os.path.isdir (dir_name):
			print_error ("The state name you entered doesn't exist", "try again")

		### Step 2: get dataframes ###
		print_status ("Load State", "Loading chat dataframe")
		self.chat_dataframe = pandas.read_pickle (os.path.join (dir_name, 'chat.dataframe'))
		print_status ("Load State", "Loading user dataframe")
		self.user_dataframe = pandas.read_pickle (os.path.join (dir_name, 'user.dataframe'))

		






	####################################################################################################
	#######################[ --- LOADING/INITIALIZING CHATS --- ]#######################################
	####################################################################################################

	# Function: load_chats_from_csv
	# -----------------------------
	# gets all of the chats in from the csv file, stores it in
	# self.chat_dataframe
	def load_chats_from_csv (self):
		self.chat_dataframe = pandas.read_csv(	self.filenames['chats'], 
												sep=";", 
												header=0, 
												nrows=self.num_chats
		)


	# Function: remove_small_chats 
	# ----------------------------
	# will remove all chats where the total length is less than 5 lines
	def remove_small_chats(self):

		speaking_indeces = (self.chat_dataframe.first_user_number_lines + self.chat_dataframe.second_user_number_lines > 4)
		self.chat_dataframe = self.chat_dataframe[speaking_indeces]		


	# Function: truncate_ids
	# ----------------------
	# converts from 'user:xxx' and 'profile:xxx' to just the integers
	# reduces amount of storage necessary!
	def truncate_chat_ids (self):

		### Step 1: chat ids ###
		self.chat_dataframe['chat_id'] = self.chat_dataframe['chat_id'].map (strip_5)

		### Step 2: user ids ###
		self.chat_dataframe['first_user_id'] = self.chat_dataframe['first_user_id'].map(strip_5)
		self.chat_dataframe['second_user_id'] = self.chat_dataframe['second_user_id'].map(strip_5)

		### Step 3: profile ids ###
		self.chat_dataframe['first_user_profile_id'] = self.chat_dataframe['first_user_profile_id'].map(strip_8)
		self.chat_dataframe['second_user_profile_id'] = self.chat_dataframe['second_user_profile_id'].map(strip_8)		

		### Step 4: disconnections ###
		self.chat_dataframe['who_disconnected'] = self.chat_dataframe['who_disconnected'].map (strip_5)


	# Function: drop_superfluous_chat_data
	# ------------------------------------
	# drops columns from chat dataframe, including reported columns
	def drop_superfluous_chat_data (self):

		self.chat_dataframe = self.chat_dataframe.drop (['who_reported', 'reason_for_report'], 1)


	# Function: get_chat_bow_from_json
	# -------------------------------
	# converts all of the json strings representing user bow
	# to dicts in the chat_dataframe
	def get_chat_bow_from_json (self):

		self.chat_dataframe['first_user_word_vector'] = self.chat_dataframe['first_user_word_vector'].map (lambda s: json.loads(s))
		self.chat_dataframe['second_user_word_vector'] = self.chat_dataframe['second_user_word_vector'].map (lambda s: json.loads(s))


	# Function: remove_stopwords_from_chats
	# -------------------------------------
	# iterates through each chat and removes stopwords
	def remove_stopwords_from_chats (self):

		self.chat_dataframe['first_user_word_vector'] 	= self.chat_dataframe['first_user_word_vector'].map (self.remove_stopwords)
		self.chat_dataframe['second_user_word_vector'] 	= self.chat_dataframe['second_user_word_vector'].map (self.remove_stopwords)


	# Function: add_chatlength_stats
	# ------------------------------
	# adds columns to the chat dataframe summarizing chat length, including:
	# - number of words for either user (*_user_num_words)
	# - number of total words exchanged
	# - number of total lines
	# - log number of total lines
	def add_chatlength_stats (self):

		### Step 1: individual user number of words ###
		self.chat_dataframe['first_user_num_words'] = self.chat_dataframe['first_user_word_vector'].map(lambda v: len(v))
		self.chat_dataframe['second_user_num_words'] = self.chat_dataframe['second_user_word_vector'].map(lambda v: len(v))

		### Step 2: total number of words ###
		self.chat_dataframe['num_words'] 		= self.chat_dataframe['first_user_num_words'] + self.chat_dataframe['second_user_num_words']
		self.chat_dataframe['log_num_words']	= self.chat_dataframe['num_words'].map (lambda x: math.log(x, 2) if x > 0 else 0)

		### Step 3: number of total lines ###
		self.chat_dataframe['num_lines'] 		= self.chat_dataframe['first_user_number_lines'] + self.chat_dataframe['second_user_number_lines']
		self.chat_dataframe['log_num_lines'] 	= self.chat_dataframe['num_lines'].map (lambda x: math.log(x, 2) if x > 0 else 0)


	# Function: get_chat_data
	# -----------------------
	# does all steps necessary to set up self.chat_dataframe
	def get_chat_data (self):

		print_status ("Chat Initialization", "loading in chats from csv")
		self.load_chats_from_csv ()

		print_status ("Chat Initialization", "removing small chats (< 4 lines)")
		#self.remove_small_chats () # TODO: we commented this out for now because our network analysis involves these chats

		print_status ("Chat Initialization", "truncating ids")
		self.truncate_chat_ids ()

		print_status ("Chat Initialization", "dropping superfluous chat data")
		self.drop_superfluous_chat_data ()

		print_status ("Chat Initialization", "getting chat bow from json")
		self.get_chat_bow_from_json ()

		print_status ("Chat Initialization", "removing stopwords")
		self.remove_stopwords_from_chats ()

		print_status ("Chat Initialization", "adding chatlength stats")
		self.add_chatlength_stats ()

		self.chat_dataframe = self.chat_dataframe.set_index ('chat_id')






	####################################################################################################
	#######################[ --- LOADING/INITIALIZING PROFILES --- ]####################################
	####################################################################################################
	
	# Function: load_profiles_from_csv
	# --------------------------------
	# gets all of the profiles from the csv file, stores it 
	# in self.profile_dataframe.
	def load_profiles_from_csv (self):
		
		self.profile_dataframe = pandas.read_csv(	self.filenames['profiles'], 
													sep=";", 
													header=0, 
		)


	# Function: truncate_profile_ids
	# ------------------------------
	# compresses profiles s.t. they have integer indeces where appropriate
	def truncate_profile_ids (self):

		self.profile_dataframe['id'] = self.profile_dataframe['id'].map(strip_8)


	# Function: drop_superfluous_profile_data
	# ---------------------------------------
	# drops superfluous columns from profile dataframe
	def drop_superfluous_profile_data (self):

		self.profile_dataframe = self.profile_dataframe.drop (['location_flag', 'created_at', 'screenname'], 1)


	# Function: get_profile_bow_from_json
	# -----------------------------------
	# converts all of the json strings representing profile bow
	# to dicts in the profile_dataframe
	def get_profile_bow_from_json (self):

		self.profile_dataframe['about'] = self.profile_dataframe['about'].map (lambda x: json.loads(x))


	# Function: remove_stopwords_from_profiles
	# ----------------------------------------
	# iterates through each profile and removes stopwords
	def remove_stopwords_from_profiles (self):

		self.profile_dataframe['about'] = self.profile_dataframe['about'].map (self.remove_stopwords)


	# Function: get_profile_data
	# --------------------------
	# does all steps necessary to set up self.chat_dataframe
	def get_profile_data (self):

		print_status ("Profile Initialization", "loading profiles from csv")
		self.load_profiles_from_csv ()

		print_status ("Profile Initialization", "truncating profile ids")
		self.truncate_profile_ids ()

		print_status ("Profile Initialization", "dropping superfluous profile data")
		self.drop_superfluous_profile_data ()

		print_status ("Profile Initialization", "getting profile bow from json")
		self.get_profile_bow_from_json ()

		print_status ("Profile Initialization", "removing stopwords from profiles")
		self.remove_stopwords_from_profiles ()









	####################################################################################################
	#######################[ --- LOADING/INITIALIZING USERS --- ]#######################################
	####################################################################################################

	# Function: get_all_users
	# -----------------------
	# gets a set of all the users in current chat dataframe
	def get_all_users (self):

		self.all_users = list(set(self.chat_dataframe['first_user_id']).union(set(self.chat_dataframe['second_user_id'])))


	# Function: create_user_dataframe
	# -----------------------------
	# create a dataframe with all empty fields,
	# user_id as index
	def create_user_dataframe (self):

		self.user_dataframe = pandas.DataFrame([	{	

													#=====[ Low-Level ]=====
													'user_id':user_id, 				# this user id
													'chat_id_vector':[],			# ids of all chats this user has been in
													'bow_vector':[],				# list of bow dicts
													'lines_sent_vector':[],			# number of lines sent for each chat
													'lines_received_vector':[],		# number of lines received for each chat
													'num_lines_vector':[],			# number of total lines per chat
													'num_words_vector':[],			# number of total words per chat
													'disconnected_vector':[],		# 1/0 for user disconnected on each chat
													'profile_id_vector':[],			# all user profile ids														
													'partner_id_vector':[],			# ids of all users seen
													'age':None,
													'gender':None,
													'location':None,
													'profile_bow':None,

													#=====[ Computed ]=====
													'profile_lda':None,				# Counter: topic -> weight in profile 
													'lda_vector':[], 				# list of Counter: topic -> weight in chat
													'weighted_topic_vector':None, 	# Counter: topic -> weighted avg. relevance to user
													'personality_cluster':None 		# index of cluster that user fits in?
												} 
											for user_id in self.all_users])

		self.user_dataframe = self.user_dataframe.set_index ('user_id')


	# Function: add_chats_to_users
	# ----------------------------
	# loops once through the chats, adding all the relevant user data to 
	# first_user and second_user
	def add_chats_to_users (self):

		def update_users (x):

			### Step 1: get user series ###
			f = self.user_dataframe.loc[x['first_user_id']]
			s = self.user_dataframe.loc[x['second_user_id']]

			### Step 2: add to list of chat_ids ###
			chat_id = x.name
			f['chat_id_vector'].append (chat_id)
			s['chat_id_vector'].append (chat_id)

			### Step 3: word vectors ###
			f['bow_vector'].append (x['first_user_word_vector'])
			s['bow_vector'].append (x['second_user_word_vector'])		

			### --- number of lines sent/received --- ###
			f_sent = x['first_user_number_lines']
			s_sent = x['second_user_number_lines']
			f['lines_sent_vector'].append (f_sent)
			f['lines_received_vector'].append (s_sent)
			s['lines_sent_vector'].append (s_sent)
			s['lines_received_vector'].append (f_sent)

			### Step 5: number of words per chat ###
			f['num_words_vector'].append (x['num_words'])
			s['num_words_vector'].append (x['num_words'])

			### Step 5: disconnected vectors ###
			if x['who_disconnected'] == x['first_user_id']:
				f['disconnected_vector'].append (1)
			else:
				f['disconnected_vector'].append (0)
			if x['who_disconnected'] == x['second_user_id']:
				s['disconnected_vector'].append (1)
			else:
				s['disconnected_vector'].append (0)

			### Step 7: profile ids ###
			f['profile_id_vector'].append (x['first_user_profile_id'])
			s['profile_id_vector'].append (x['second_user_profile_id'])

			### Step 8: users seen ###
			f['partner_id_vector'].append (x['second_user_profile_id'])
			s['partner_id_vector'].append (x['first_user_profile_id'])

			### Step 9: update the user dataframe ###
			self.user_dataframe.loc[x['first_user_id']] = f
			self.user_dataframe.loc[x['second_user_id']] = s


		### Step 1: get baseline statistics ###
		self.chat_dataframe.apply (update_users, axis=1)


	# Function: add_profiles_to_users
	# -------------------------------
	# loops once through the users, adding profile data as it goes
	def add_profiles_to_users (self):

		def update_user (x):

			profile_id = x['profile_id_vector'][-1]
			profile = self.profile_dataframe.loc[profile_id]

			x['age'] = profile['age']
			x['gender'] = profile['gender']
			x['location'] = profile['location']
			x['profile_bow'] = profile['about']

			return x

		self.user_dataframe.apply (update_user, axis=1)


	# Function: get_user_data
	# -----------------------
	# does all steps necessary to get self.user_dataframe up
	def get_user_data (self):

		print_status ("User Initialization", "getting list of all users")
		self.get_all_users ()

		print_status ("User Initialization", "creating user dataframe")
		self.create_user_dataframe ()

		print_status ("User Initialization", "adding chats to users")
		self.add_chats_to_users ()

		print_status ("User Initialization", "adding profile data to users")
		self.add_profiles_to_users ()







	####################################################################################################
	#######################[ --- NATURAL LANGUAGE (TOPIC MODELING) --- ]################################
	####################################################################################################

	# Function: load_stopwords 
	# ------------------------
	# fills self.stopwords with a set of stopwords
	def load_stopwords (self):

		self.stopwords = set(pickle.load (open(self.filenames['stopwords_list'], 'r')))


	# Function: remove_stopwords
	# --------------------------
	# given a bow dict, this removes all words in self.stopwords
	def remove_stopwords (self, bow):
		
		return {key:value for key, value in bow.iteritems() if not key in self.stopwords}


	# Function: get_gensim_corpus 
	# ---------------------------
	# gets a gensim corpus from all chats
	# puts it in self.dictionary
	def get_gensim_corpus (self):

		self.dictionary = corpora.Dictionary (	chain (	
														self.chat_dataframe['first_user_word_vector'], 
														self.chat_dataframe['second_user_word_vector'] 
												)
											)


	# Function: get_gensim_bow_vector
	# -------------------------------
	# adds a bow_vector to self.user_dataframe
	def convert_to_gensim_bow (self):

		print_inner_status ("first user word vectors to gensim bow")
		self.chat_dataframe['first_user_word_vector'] = self.chat_dataframe['first_user_word_vector'].map(lambda x: self.dictionary.doc2bow (x))
		print_inner_status ("second user word vectors to gensim bow")
		self.chat_dataframe['second_user_word_vector'] = self.chat_dataframe['second_user_word_vector'].map(lambda x: self.dictionary.doc2bow (x))		
		print_inner_status ("profile word vectors to gensim bow")
		self.profile_dataframe['about'] = self.profile_dataframe['about'].map (lambda x: self.dictionary.doc2bow(x))


	# Function: get_lda_vectors
	# -------------------------
	# for each chat in chat_dataframe, adds in an lda vector describing it
	# (represented as a counter)
	def get_chat_lda_vectors (self):

		### Step 1: build the lda model ###
		print_inner_status ("building lda model")
		self.lda_model = models.LdaModel (	chain ( self.chat_dataframe['first_user_word_vector'],
													self.chat_dataframe['second_user_word_vector'],
											), 
											num_topics=self.num_topics
										)

		### Step 2: get actual lda vectors ###
		print_inner_status ("getting lda vectors: first_user")
		self.chat_dataframe['first_user_lda_vector'] = self.chat_dataframe['first_user_word_vector'].map(lambda vec: Counter(dict(self.lda_model[vec])))
		print_inner_status ("getting lda vectors: second_user")
		self.chat_dataframe['second_user_lda_vector'] = self.chat_dataframe['second_user_word_vector'].map(lambda vec: Counter(dict(self.lda_model[vec])))


	# Function: get_weighted_topic_vector
	# -----------------------------------
	# adds to the user dataframe the weighted_topic_vector
	def add_weighted_topic_vectors_to_users (self):

		### Step 1: get lda vectors as Counter ###
		self.user_dataframe['lda_counter_vector'] 			= self.user_dataframe['lda_vector'].map(lambda vec: [Counter(dict(x)) for x in vec])
		self.user_dataframe['profile_counter_lda'] 			= self.user_dataframe['profile_lda'].map(lambda x: Counter(dict(x)))

		### Step 2: get log lines sent, log lines received ###
		self.user_dataframe['log_lines_sent_vector'] 		= self.user_dataframe['lines_sent_vector'].map(lambda vec: [int(math.floor(math.log(x))) + 1 if x > 0 else 0 for x in vec])
		self.user_dataframe['log_lines_received_vector'] 	= self.user_dataframe['lines_received_vector'].map(lambda vec: [int(math.floor(math.log(x))) + 1 if x > 0 else 0 for x in vec])	

		### Step 3: get total log lines sent, log lines received ###
		self.user_dataframe['total_log_lines_sent'] 		= self.user_dataframe['log_lines_sent_vector'].map (lambda vec: sum(vec))
		self.user_dataframe['total_log_lines_received'] 	= self.user_dataframe['log_lines_received_vector'].map (lambda vec: sum(vec))	

		### Step 4: get weighted topic vector ###
		def get_weighted_topic_vector (user_series):
			total = float(user_series['total_log_lines_sent'])
			for log_length, topics in zip (user_series['log_lines_sent_vector'], user_series['lda_counter_vector']):
				for i in range(log_length):
					user_series['weighted_topic_vector'].update (topics)
			for key in user_series['weighted_topic_vector'].iterkeys():
				user_series['weighted_topic_vector'][key] /= total

		self.user_dataframe['weighted_topic_vector'] = self.user_dataframe['lda_counter_vector'].map(lambda x: Counter({}))
		self.user_dataframe.apply (get_weighted_topic_vector, axis=1)

		### Step 5: convert it to a numpy array ###
		self.user_dataframe['weighted_topic_vector'] = self.user_dataframe['weighted_topic_vector'].map (lambda x: np.array([x[i] for i in range(100)]))

		### Step 5: drop the superfluous columns ###
		self.user_dataframe = self.user_dataframe.drop (['lda_vector', 'profile_lda'], axis=1)
 

	# Function: perform_gensim_preprocessing
	# --------------------------------------
	# does everything to get bow, lda vectors for all profiles
	# and conversations
	def perform_topic_modelling (self):

		### Step 1: get the corpus ###
		print_status ("Topic Modeling", "getting corpus from documents")
		self.get_gensim_corpus ()

		### Step 2: convert everything to gensim bow ###
		print_status ("Topic Modeling", "converting word vectors to gensim bow representaion")
		self.convert_to_gensim_bow ()

		### Step 3: get lda vectors ###
		print_status ("Topic Modeling", "Getting lda vectors")
		self.get_chat_lda_vectors ()

		### Step 4: add weighted ldas to users ###
		print_status ("Gensim Preprocessing", "Adding weighted LDA vectors to users")
		self.add_weighted_topic_vectors_to_users ()





	####################################################################################################
	#######################[ --- NETWORKS --- ]#########################################################
	####################################################################################################

	# Function: add_single_edge
	# -------------------------
	# adds a single edge for the network	
	def add_single_edge(self, chat_series):

		node1 = chat_series["first_user_id"]
		node2 = chat_series["second_user_id"]
		first_user_num_lines = chat_series['first_user_number_lines']
		second_user_num_lines = chat_series['second_user_number_lines']
		who_disconnected = chat_series['who_disconnected']
		self.graph.add_edge(node1, node2, first_user_num_lines=first_user_num_lines, second_user_num_lines=second_user_num_lines, who_disconnected=who_disconnected)


	# Function: create_network
	# -----------------------
	# creates a network
	def create_network(self):
		self.graph = nx.Graph()
		self.chat_dataframe.apply(self.add_single_edge, axis=1)	

		### Saving copies of the graph with removed 
		for i in self.REMOVED_EDGE_RANGE:
			currGraph = self.graph.copy()

			toRemove = [(n1,n2) for n1,n2,dat in currGraph.edges(data=True) if (dat['first_user_num_lines'] + dat['second_user_num_lines']) < i]
			currGraph.remove_edges_from(toRemove)
			self.removed_edge_graphs.append(currGraph)	



	####################################################################################################
	#######################[ --- CLUSTERING --- ]#######################################################
	####################################################################################################

	# Function: print_cluster_sizes
	# -----------------------------
	# given X and a cluster model, this function
	# prints out the sizes of the clusters
	def print_cluster_sizes (self, model, X):

		print_header ("CLUSTER RESULTS")
		labels = Counter(model.predict(X))
		for key, value in labels.iteritems ():
			print "cluster ", key, ": ", value


	# Function: cluster_users_kmeans
	# ------------------------------
	# assigns each user to a cluster via k_means
	def cluster_users_kmeans (self):

		### Step 1: get the X out ###
		X = get_weighted_topic_vectors_as_features (self.user_dataframe)

		### Step 2: create kmeans algorithm, fit-predict ###
		self.kmeans = sklearn.cluster.KMeans (n_clusters=self.num_personalities, init='k-means++', n_init=10, max_iter=100, precompute_distances=True)
		self.kmeans.fit (X)

		### Step 3: add personality cluster to dataframe ###
		self.user_dataframe['personality_kmeans'] = self.user_dataframe['weighted_topic_vector'].map (lambda x: self.kmeans.predict(x))


	# Function: cluster_users_GMM 
	# ---------------------------
	# Parameters (1):
	#	- X: numpy matrix, where each row is a data point
	# Description: 
	# 	trains and stores in self.personality_GMM the gaussian mixture
	# 	model representing personalities.
	def cluster_users_GMM (self, X):

		#--- PARAMETERS ---
		n_iter = 100 	# max. number of iterations that EM will run for
		n_init = 5 		# number of times it re-initializes the centers. (takes the best outcome)

		### Step 1: create the GMM object ###
		self.personality_GMM = GMM (n_components=self.num_personalities, n_iter=n_iter, n_init=n_init)

		### Step 3: perform the clustering ###
		print_inner_status ("Running EM Algorithm")
		self.personality_GMM.fit (X)





	####################################################################################################
	#######################[ --- EVALUATION/TESTING --- ]###############################################
	####################################################################################################

	# Function: evaluate_model
	# ------------------------
	# Parameters (3):
	#	- classifier: the model that you want to use to evaluate it
	#	- feature_function: function to extract X (list of feature vectors) from chat dataframe
	#	- label_function: function to extract Y (list of true labels) from chat dataframe
	# Description:
	# 	the feature function should return an N*m matrix, where N = number of users, m = number of features;
	#	label function will return an N*1 matrix, which is the list of labels (i.e. chat buckets)
	#	this will extract the features, labels, then run n-fold cross-validation on a trained classifier
	# 	and report the results
	#	returns a numpy array of scores.
	def evaluate_model (self, classifier, feature_function, label_function):

		### Step 1: extract the features ###
		print_status ("Evaluate Model", "extracting features from chats using " + feature_function.__name__)
		X = feature_function (self.chat_dataframe, self.user_dataframe, self.removed_edge_graphs)

		### Step 2: extract the true labels ###
		print_status ("Evaluate Model", "extracting labels from chats using " + label_function.__name__)
		y = label_function (self.chat_dataframe)

		### Step 3: perform cross validation ###
		print_status ("Evaluate Model", "performing cross validation")
		scores = cross_validation.cross_val_score (classifier, X, y=y, cv=5)


		classifier.fit(X,y)
		print classifier.coef_
		return scores



	# Function: evaluate_pairings
	# ---------------------------
	# evaluates the classifier by finding the percent of time time that it picks the larger of 
	# two unknown edges from a given user
	def evaluate_pairings (self, classifier, feature_function, label_function):

		### Step 1: extract features, X ###
		print_status ("Evaluate Model", "extracting features from chats using " + feature_function.__name__)
		X = feature_function (self.chat_dataframe, self.user_dataframe, self.removed_edge_graphs)

		### Step 2: extract labels, y ###
		print_status ("Evaluate Model", "extracting labels from chats using " + label_function.__name__)
		y = label_function (self.chat_dataframe)	

		### Step 3: train the classifier ###
		self.classifier = classifier.fit (X, y)

		### Step 4: 




	# Function: evaluate_pairings
	# -------------------------
	# TODO
	def evaluate_pairings (self, classifier, feature_function, label_function, series_feature_function, num_pairings = 1000):
	
		### Step 1: extract the features ###
		print_status ("Evaluate Model", "extracting features from chats using " + feature_function.__name__)
		X = feature_function (self.chat_dataframe, self.user_dataframe, self.removed_edge_graphs)

		### Step 2: extract the true labels ###
		print_status ("Evaluate Model", "extracting labels from chats using " + label_function.__name__)
		y = label_function (self.chat_dataframe)	

		### Step 3: training the classifier ###
		self.classifier = classifier.fit(X, y)

		### Step 4: prediction ###
		num_correct = 0
		for index in range(num_pairings):

			### Step 1: pick the node that we are evaluating ###
			picked_node = None
			while True:
				picked_node = self.user_dataframe.ix[sample(self.user_dataframe.index, 1)]
				num = len(picked_node['chat_id_vector'].values[0])
				if num > 1: break

			chat_ids = sample(picked_node['chat_id_vector'].values[0], 2)
			chats = [self.chat_dataframe.ix[chat_ids[0]], self.chat_dataframe.ix[chat_ids[1]]]
		
			best_num_lines = max(chats[0]['num_lines'], chats[1]['num_lines'])
			if chats[0]['num_lines'] == chats[1]['num_lines']: free_wins += 1

			features = [series_feature_function(curr, self.removed_edge_graphs) for curr in chats]
			#features = [series_feature_function(curr) for curr in chats]
			predictions  = [classifier.predict(curr) for curr in features]
			
			predicted_winner = 0
			if predictions[1] > predictions[0]:
				predicted_winner = 1
			predicted_best_num_lines = chats[predicted_winner]['num_lines']

			#print chats[0]['num_lines'], chats[1]['num_lines'], predictions[0], predictions[1]

			if best_num_lines == predicted_best_num_lines: num_correct += 1
			else: num_incorrect += 1

		print "correct:", num_correct, "incorrect:", num_incorrect, "error_rate:", (num_incorrect) / float(num_correct+num_incorrect)
		print "free_wins:", free_wins
		print 'PERCENT_CORRECT', (num_correct-free_wins)/float(num_incorrect+num_correct-free_wins)
		return (num_incorrect) / float(num_correct+num_incorrect)



####################################################################################################
#######################[ --- MAIN OPERATION --- ]###################################################
####################################################################################################

if __name__ == "__main__":


	### Step 1: set up arg parser ###
	parser = argparse.ArgumentParser(description='Reasoning on the Chatous network')

	# Note: a total of 9050713 chats, default to 100000 of them
	parser.add_argument(	'--chats', 
							type=int, 
							nargs=1,  
							default='10000',
							help='the number of chats to load')

	# Note: total of 332888 profiles, defaults to all of them
	parser.add_argument(	'--profiles', 
							type=int, 
							nargs=1, 
							default='332888',
							help='the number of profiles to load')	

	parser.add_argument(	'--personalities', 
							type=int, 
							nargs=1, 
							default='10',
							help='the number of distinct personality types (for clustering)')	

	parser.add_argument (	'--topics',
							type=int,
							nargs=10,
							default=None,
							help='number of topics to use for LDA')

	parser.add_argument (	'--chat_dataframe',
							type=str,
							nargs=1,
							default=None,
							help='name of the dataframe to load in as the chat_dataframe')


	args = parser.parse_args ()

	### Step 1: create/initialize the chatous object ###
	num_chats = 0
	num_profiles = 0
	num_personalities = 10

	if type(args.chats) != type(1):
		num_chats = args.chats[0]
	else:
		num_chats = args.chats

	if type(args.profiles) != type(1):
		num_profiles = args.profiles[0]
	else:
		num_profiles = args.profiles

	if type(args.personalities) != type(1):
		num_personalities = args.personalities[0]
	else:
		num_personalitites = args.personalities

	chatous = Chatous (num_chats=num_chats, num_profiles=num_profiles, num_personalities=num_personalities)

