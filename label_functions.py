# -------------------------------------------------- #
# File: label_functions
# ---------------------
# contains all feature extracting functions
# each one takes in a chat_dataframe
# -------------------------------------------------- #
#--- Standard ---
import math

#--- Pandas ---
import pandas as pd

#-- Numpy ---
import numpy as np

#-- Networkx --
import networkx as nx

#--- My Files ---
from util import *


####################################################################################################
#######################[ --- LABEL FUNCTIONS --- ]##################################################
####################################################################################################

# Function: chatlength_label_extractor
# ------------------------------------
# returns the label as just the total length in lines 
# of the chat 
def chatlength_label_extractor (chat_dataframe):

	labels = []

	def get_baseline_label (chat_series):
		
		total_lines = chat_series['first_user_number_lines'] + chat_series['second_user_number_lines']
		labels.append (total_lines)

	### Step 1: extract labels ###
	chat_dataframe.apply (get_baseline_label, axis=1)

	### Step 2: convert to numpy matrix ###
	labels = np.array (labels)
	return labels.transpose ()


# Function: log_chatlength_label_extractor
# ----------------------------------------
# returns the label as the log of the total length in lines of the chat
def log_chatlength_label_extractor (chat_dataframe):

	labels = []

	def get_log_chatlength (chat_series):
		total_lines = chat_series['first_user_number_lines'] + chat_series['second_user_number_lines']
		labels.append (math.log (total_lines+1)) # Plus one to avoid log(0)

	### Step 1: extract labels ###
	chat_dataframe.apply (get_log_chatlength)

	### Step 2: convert to numpy matrix ###
	labels = np.array (labels)
	return labels.transpose ()


# Function: log_2_buckets_label_extractor
# ---------------------------------------
# returns the label as the floor of log base 2 of the 
# chat length
def log_2_buckets_label_extractor (chat_dataframe):

	labels = []

	def get_log_2_bucket (chat_series):
		
		total_lines = chat_series['first_user_number_lines'] + chat_series['second_user_number_lines']
		if total_lines == 0: log_2_bucket = 0
		else: log_2_bucket = math.floor (math.log(total_lines, 2))
		labels.append (log_2_bucket)

	### Step 1: extract labels ###
	chat_dataframe.apply (get_log_2_bucket, axis=1)

	### Step 2: convert to numpy matrix ###
	labels = np.array (labels)
	return labels.transpose ()








