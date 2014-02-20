# -------------------------------------------------- #
# File: util
# ----------
# contains utilities for working with pandas, 
# json, etc.
# -------------------------------------------------- #

#--- Standard ---
from collections import Counter
import json
import math




####################################################################################################
#######################[ --- INTERFACE --- ]########################################################
####################################################################################################

# Function: print_welcome
# -----------------------
# prints out a welcome message for this project
# in particular
def print_welcome ():
	print "##################################################"
	print "#######[ Chatous Analysis Project ]###############"
	print "#######[ -- Jay, Sam and Alex  -- ]###############"
	print "##################################################"


# Function: print_header
# ----------------------
# prints out the header for a section of output
def print_header (header_name):
	
	print "\n=====[ " + str(header_name) + " ]====="


# Function: print_status
# ----------------------
# prints out a status message; only call this from large, significant 
# functions
def print_status (stage, status):

	print "---> " + str(stage) + ": " + str(status)


# Function: print_inner_status
# ----------------------------
# prints out an indented status message; call this from inner functions
def print_inner_status (status):

	print "		---> " + str(status)


# Function: print_error
# ---------------------
# prints out an error message then exits
def print_error (top_line, bottom_line):
	print "ERROR: ", top_line
	print "-"*len("ERROR: " + str(top_line))
	print bottom_line


# Function: print_parameters
# --------------------------
# displays all of the parameters to the user
def print_parameters (chatous):

	print_header("Info on Parameters")
	print "	- # of chats in use: ", chatous.num_chats
	print "	- # of profiles in use: ", chatous.num_profiles
	print "	- # of personalities: ", chatous.num_personalities
	print "	- # of topics: ", chatous.num_topics
	print "\n"



####################################################################################################
#######################[ --- GENERAL UTILITIES --- ]################################################
####################################################################################################

# Function: strip_5
# -----------------
# strips first 5 characters, turns rest into an int
def strip_5 (x):
	return int(x[5:]) if not x == 'null' else None

# Function: strip_8
# -----------------
# strips first 5 characters, turns rest into an int
def strip_8 (x):
	return int(x[8:]) if not x == 'null' else None












