import pandas as pd
import numpy as np
import seaborn as sns
import os
import scipy
import sys
import dlatk
from functools import reduce
import pronoun_transformation.pronoun_transformation_pipeline as pronoun_pp
from pronoun_transformation.swap_gender_pronouns import SWAP_DICTIONARY
import matplotlib
matplotlib.use('agg')

pd.set_option("display.max_rows", 500)


# SPECIFYING VARIABLE NAMES
db = 'politeness' #database name
message_table = 'twitter' #original message table
user_initials = 'sc' #initials of user running the pipeline
features_used = 'ngr_liwc_plex' #Only used for naming the final table, eg. if using LIWC plus politelex use liwc_politelex

lexicon_table_name = 'dd_twitter_politeness_npl'
weighted_lexicon_flag = True

category_table = 'feat$cat_LIWC2015$twitter$sid$1gra' #table from which to subset categories for transformation
category_col = 'feat' #category column
category_name = 'PRONOUN' #name of the category of messages for which to perform transformation

ngram_table_name = 'feat$1to3gram$twitter$sid$16to16'
old_score_table = 'feat$cat_dd_twitter_politeness_npl_w$twitter$sid$1to3'

plots_path = "gender_swap_plots"

if __name__ == '__main__':

	if not os.path.exists(plots_path):
		os.mkdir(plots_path)

	final_tables = []

	print("Starting Gender Swap Pipeline...\n\n")

	for swap_type in list(SWAP_DICTIONARY.keys()):

		basetable_name = message_table + "_" + user_initials + "_" + swap_type
		message_table_ref = "$" + message_table + "$"
		base_table_ref = "$" + basetable_name + "$"
		transformed_ngram_table_name = ngram_table_name.replace(message_table_ref, base_table_ref)
		new_score_table = old_score_table.replace(message_table_ref, base_table_ref)

		gender_from_names = SWAP_DICTIONARY.get(swap_type).get('gender_from_names')
		gender_to_name = SWAP_DICTIONARY.get(swap_type).get('gender_to_name')
		transformation_name = SWAP_DICTIONARY.get(swap_type).get('transformation_name')



		print("\n\nPerforming transformation: {}".format(transformation_name))

		### Create Basetable with Message IDs
		print("\nStep 1: Creating Basetable containing Message IDs to be transformed.\n")
		pronoun_pp.create_base_table(basetable_name, category_table, category_col, category_name, db)

		### Transform ngrams with Gender Swap
		print("\nStep 2: Swapping gender terms in ngram table.\n")
		transformed_df = pronoun_pp.transform_ngrams(ngram_table_name, basetable_name, gender_from_names, gender_to_name, db)
		print("Example:")
		print(transformed_df.head(10))

		### Create updated ngram df and transformation metadata df
		onegram_df = pronoun_pp.create_transformed_ngram_table(transformed_df)
		metadata_df = pronoun_pp.create_tranformation_metadata_table(transformed_df)

		### Calculate Updated Politness Scores on Swapped Table
		print("\nStep 3: Push updated ngram table to the database and re-run lexica-based model to gather updated scores.\n")
		pronoun_pp.calculate_transformed_scores(onegram_df, transformed_ngram_table_name, basetable_name, 
	                                        lexicon_table_name, weighted_lexicon_flag, db)

		### Calculate the difference in scores before and after the gender swap
		print("\nStep 4: Calculate score differences.\n")
		effect_df = pronoun_pp.compare_transform_effect(old_score_table, new_score_table, message_table, db)
		print("Messages with a change in scores after gender swap:")
		print(effect_df[effect_df.score_difference != 0].head(10))

		swap_final_df = metadata_df.merge(effect_df, left_on = 'group_id', right_on = 'id')[['id', 'message', 'original_score', 'transformed_score']]
		score_column_name = swap_type + "_score"
		swap_final_df.columns = ['id', 'message', 'original_score', score_column_name]

		print(swap_final_df.head(10))

		final_tables.append(swap_final_df)

	print("\nCompiling results from all gender transformations...\n")

	final_df = reduce(lambda left, right: pd.merge(left, right, on = ['id', 'message', 'original_score'], how = 'outer'), final_tables)

	print(final_df.head(10))

	final_table_name = message_table + "_" + user_initials + "_" + features_used + "_" +  "gender_swap"

	pronoun_pp.store_table(final_df, final_table_name, db)

	print("\nCreating boxplot from results...\n")

	pronoun_pp.generate_boxplot(final_df, save_path = plots_path + "/" + final_table_name + ".png")


	print("\nPipeline is complete!\n") 
	print("Your results can be found in the table {}.{}".format(db, final_table_name))
	print("Your boxplot can be found at {}/{}.png".format(plots_path, final_table_name))



















