import pandas as pd
import numpy as np
import seaborn as sns
import os
import scipy
import sys
import dlatk
from functools import reduce
import argparse
import pronoun_transformation.pronoun_transformation_pipeline as pronoun_pp
import matplotlib
matplotlib.use('agg')

pd.set_option("display.max_rows", 500)


################## Example arguments ####################################
# db = 'politeness' #database name
# message_table = 'twitter' #original message table
# user_initials = 'sc' #initials of user running the pipeline
# features_used = 'ngr_liwc_plex' #Only used for naming the final table, 
# eg. if using LIWC plus politelex use liwc_politelex

# lexicon_table_name = 'dd_twitter_politeness_npl'
# weighted_lexicon_flag = True

# category_table = 'feat$cat_LIWC2015$twitter$sid$1gra' #table from which to subset categories for transformation
# category_col = 'feat' #category column
# category_name = 'PRONOUN' #name of the category of messages for which to perform transformation

# ngram_table_name = 'feat$1to3gram$twitter$sid$16to16'
# old_score_table = 'feat$cat_dd_twitter_politeness_npl_w$twitter$sid$1to3'

# plots_path = "gender_swap_plots"


############ EXAMPLE COMMAND ##################

# python run_pipeline.py politeness twitter dd_twitter_politeness_npl \
# 'feat$1to3gram$twitter$sid$16to16' 'feat$cat_dd_twitter_politeness_npl_w$twitter$sid$1to3' gender_swap_plots \
# --weighted_lexicon --user sc --features_used ngr_liwc_plex

if __name__ == '__main__':

	# Create the parser
	my_parser = argparse.ArgumentParser(prog='Gender Perturbation Pipeline',
		description="Run Gender Perturbation Pipeline to uncover model bias")

	# Add the arguments
	my_parser.add_argument('db',
                       type=str,
                       help='the database for the message text')

	my_parser.add_argument('message_table',
                       type=str,
                       help='the table that contains the message text') 

	my_parser.add_argument('lexicon',
                       type=str,
                       help='the lexicon table to be used')

	my_parser.add_argument('ngram_table',
                       type=str,
                       help='the table containing ngrams from the message text')
	
	my_parser.add_argument('score_table',
                       type=str,
                       help='the table containing scores')   

	my_parser.add_argument('plots_path',
                       type=str,
                       help='the path for storing any plots generated')

	my_parser.add_argument('--category_table',
                       type=str,
                       help='the table containing category classifications for the message text',
                       default = 'feat$cat_LIWC2015$twitter$sid$1gra')

	my_parser.add_argument('--category_column',
                       type=str,
                       help='the column from the category table, that contains assigned categories',
                       default = 'feat')
	
	my_parser.add_argument('--category_value',
                       type=str,
                       help='the category value with which to subset the message table to be used to run the pipeline',
                       default = 'PRONOUN')  

	my_parser.add_argument('--weighted_lexicon',
                       help='flag for whether the lexicon is weighted',
                       action = "store_true") 

	my_parser.add_argument('--user',
                       type=str,
                       help='the initials for the user running the pipeline',
                       default = "")

	my_parser.add_argument('--features_used',
                       type=str,
                       help='a string representation of the features used for the pipeline being run',
                       default = "")    

	args = my_parser.parse_args()

	pronoun_pp.run_pipeline(db = args.db,
                message_table = args.message_table,
                user_initials = args.user,
                features_used = args.features_used,
                lexicon_table_name = args.lexicon,
                weighted_lexicon_flag = args.weighted_lexicon,
                ngram_table_name = args.ngram_table,
                old_score_table = args.score_table,
                plots_path = args.plots_path,
                category_table = args.category_table,
                category_col = args.category_column,
                category_name = args.category_value)





















