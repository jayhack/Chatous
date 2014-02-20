--------------------------------------------
CS 224W Project, Fall 2013
User Compatability Classification on Chatous
by Jay Hack, Sam Beder and Alex Zamoschin
--------------------------------------------

NOTE: since you restrict the size of ./more.zip, I was not able
to include data with this project (we have about 5gb of data.)


1: Setup/Installation
=====================

1.1: Packages
-------------
You need to install the following packages to run this:

• pandas (for moving around data)

• networkx (for reasoning on networks)

• genism (for topic modelling)

• scikit-learn (for all supervised learning, clustering)

these can be installed on corn/myth/cmnXX via the following commands:

easy_install --user [...]

or

pip install --User [...]	//not sure about this last one


1.2: Running Basic Commands
---------------------------
• in source, run 
~$: python
~$: import chatous
~$: c = chatous.Chatous (state_name='1m')
to get dataframes of 1m users in



2: Getting Data in
==================

2.0: Data Organization
----------------------
Make sure the csv files are named 'chats.csv' and 'profiles.csv'
then run 
	~$: source ./mark_data.sh
to mark them with indeces. (If they are already marked, this will do nothing.)

Note: it looks like on corn, you need to do 
	~$: `which bash` ./mark_data.sh
on corn (and myth?) for some reason, otherwise it freaks out.


2.1: How to do it
-----------------

to load in a saved state, the following is sufficient: (30s on 1 million)

	~$: python

	>>> import chatous

	>>> c = chatous.Chatous (state_name='1m')


to save your state once you have preprocessed it, the following:
	
	>>> c.save_state ('my_state')


to make a new state, the following:

	~$: python

	>>> import chatous

	>>> c = chatous.Chatous (num_chats=x, num_profiles=y, num_personalities=z, num_topics=w)


2.2: Stats on data
------------------
total number of chats: 		9050713
total number of profiles: 	333288
total number of users: 		(see below)

2.3: Stats on data loading/processing speed
-------------------------------------------
If you remove the empty chats (where neither user says anything), you get
a huge increase in speed. Assuming empty chats are removed:
• To load 100k chats:
	- it takes ~ 2 minutes
	- 14,314 unique users
• To load 1,000,000 chats:
	- it takes ~ ten minutes (including clustering)
	- 24,595 unique users


• On the full 9 Million: 
	- 1,891,300 non-empty chats total
	- ~1m to load the original pandas dataframe
	- ~1m to convert word vectors to json
	- ~1m to remove all stopwords
	- ~10s to add chatlength stats
	- ~10m to convert over to gensim bow model
	- ~2m to construct lda model (using itertools.chain, on individual convos)
	- ~20m/user to convert to lda vectors...

	- takes ~5m to pickle without 'all_words'
		- 1GB when pickled w/ dict bow
		- .75GB when pickled w/ gensim bow
		- 0.5GM when pickled w/ lda
	- takes about 1m to unpickle w/ dict bow



3: Preliminary Results
======================

3.1: Data Reduction
-------------------
• With the first 100,000 and 1,000,000 chats, removing all empty chats (neither user spoke)
	yields a total of 14,314 and 24,595 unique users, respectively
• It takes about 40 seconds on my Macbook Pro to load in all of the users on 100k. (Much more without
	the reduction!)
• It takes about 20 minutes on my Macbook Pro to load in all of the users on 1M, assuming reduction.


• With the first 9m chats, you get:
	- 1891300 non-empty chats
	- 84141 unique users


3.2: Clustering:
----------------
• TF.IDF Vector: ith element is the tf.idf of the ith word w/r/t that user's 'document,' 
	or combined textual history

• Applying KMeans to the tf.idf vectors, for 10 different clusters, produced the following cluster
	sizes: (This is once empty chats are removed)


	--- On 100k chats ---
	cluster  0 :  5163
	cluster  1 :  1235
	cluster  2 :  1447
	cluster  3 :  697
	cluster  4 :  1717
	cluster  5 :  303
	cluster  6 :  1955
	cluster  7 :  1139
	cluster  8 :  522
	cluster  9 :  136

	--- on 1M chats ---
	cluster  0 :  727
	cluster  1 :  8621
	cluster  2 :  2803
	cluster  3 :  537
	cluster  4 :  2923
	cluster  5 :  1977
	cluster  6 :  1197
	cluster  7 :  890
	cluster  8 :  2740
	cluster  9 :  2180


3.3: Clustering Results
-----------------------

features: weighted topic vectors w/out profile included
results (10):


3.4: Classification Results
---------------------------

label: log 2 buckets for chat length
features: absolute age diff, gender
results:
[ 0.3450674   0.3450674   0.34508384  0.34508384  0.34513148]

label: log 2 buckets for chat length
features: topic vector absolute difference
results:
array([ 0.34440051,  0.34535321,  0.34498857,  0.34627477,  0.34484566])


label: chatlength
features: absolute age diff, gender
results:
[ -8.56507832e-05,  -9.85388507e-05,  -7.13176254e-05, -2.61171582e-04,  -5.85799507e-05 ]

label: chatlength
features: enhanced baseline (more than just the original)
results:
[ 0.00030124,  0.00132571,  0.00034594,  0.00047929,  0.00045741]

label: chatlength
features: topic vector absolute difference
results:
[ 0.0046338 ,  0.00148313,  0.00267125,  0.00444872,  0.00345473 ]

label: chatlength
features: topic + enhanced baseline
[ 0.0049204 ,  0.00274771,  0.00295398,  0.00493439,  0.00392888]

label: chatlength
features: personality combos
results:
[ 3.95841956e-04,   3.98540792e-04,   2.55439238e-04, 2.17616919e-04,  -8.19841754e-05 ]















