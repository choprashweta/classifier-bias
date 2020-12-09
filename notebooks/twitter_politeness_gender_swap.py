import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy
import sys
import dlatk
from functools import reduce
import pronoun_transformation.pronoun_transformation_pipeline as pronoun_pp
from pronoun_transformation.swap_gender_pronouns import SWAP_DICTIONARY

pd.set_option("display.max_rows", 500)


# SPECIFYING VARIABLE NAMES
db = 'politeness' #database name
message_table = 'twitter' #original message table
user_initials = 'sc' #initials of user running the pipeline

lexicon_table_name = 'dd_twitter_politeness'

category_table = 'feat$cat_LIWC2015$twitter$sid$1gra' #table from which to subset categories for transformation
category_col = 'feat' #category column
category_name = 'PRONOUN' #name of the category of messages for which to perform transformation

onegram_table_name = 'feat$1gram$twitter$sid$16to16'
old_score_table = 'feat$cat_dd_twitter_politeness_w$twitter$sid$1gra'


if __name__ == '__main__':

	final_tables = []

	print("Starting Gender Swap Pipeline...\n\n")

	for swap_type in list(SWAP_DICTIONARY.keys()):

		basetable_name = message_table + "_" + user_initials + "_" + swap_type
		message_table_ref = "$" + message_table + "$"
		base_table_ref = "$" + basetable_name + "$"
		transformed_1gram_table_name = onegram_table_name.replace(message_table_ref, base_table_ref)
		new_score_table = old_score_table.replace(message_table_ref, base_table_ref)

		gender_from_names = SWAP_DICTIONARY.get(swap_type).get('gender_from_names')
		gender_to_name = SWAP_DICTIONARY.get(swap_type).get('gender_to_name')
		transformation_name = SWAP_DICTIONARY.get(swap_type).get('transformation_name')


		print("Performing transformation: {}".format(transformation_name))

		### Create Basetable with Message IDs
		print("\nStep 1: Creating Basetable containing Message IDs to be transformed.\n")
		pronoun_pp.create_base_table(basetable_name, category_table, category_col, category_name, db)

		### Transform 1grams with Gender Swap
		print("\nStep 2: Swapping gender terms in 1gram table.\n")
		transformed_df = pronoun_pp.transform_1grams(onegram_table_name, basetable_name, gender_from_names, gender_to_name, db)
		print("Example:")
		print(transformed_df.head(10))

		### Create updated 1gram df and transformation metadata df
		onegram_df = pronoun_pp.create_transformed_1gram_table(transformed_df)
		metadata_df = pronoun_pp.create_tranformation_metadata_table(transformed_df)

		### Calculate Updated Politness Scores on Swapped Table
		print("\nStep 3: Push updated 1gram table to the database and re-run lexica-based model to gather updated scores.\n")
		pronoun_pp.calculate_transformed_scores(onegram_df, transformed_1gram_table_name, basetable_name, 
	                                        lexicon_table_name, db)

		### Calculate the difference in scores before and after the gender swap
		print("\nStep 4: Calculate score differences.\n")
		effect_df = pronoun_pp.compare_transform_effect(old_score_table, new_score_table, message_table, db)
		print("Messages with a change in scores after gender swap:")
		print(effect_df[effect_df.score_difference != 0].head(10))

		swap_final_df = metadata_df.merge(effect_df, left_on = 'group_id', right_on = 'id')[['id', 'message', 'original_score', 'transformed_score']]
		score_column_name = swap_type + "_score"
		swap_final_df.columns = ['id', 'message', 'original_score', score_column_name]

		final_tables.append(swap_final_df)

	print("\nCompiling results from all gender transformations...\n")

	final_df = reduce(lambda left, right: pd.merge(left, right, on = ['id', 'message', 'original_score'], how = 'outer'), final_tables)

	final_table_name = message_table + "_" + user_initials + "_" + "gender_swap"

	pronoun_pp.store_table(final_df, final_table_name, db)

	print("\nPipeline is complete! Your results can be found in the table {}.{}".format(db, final_table_name))



















