import os
import pandas as pd
from sys import argv
from .get_engine import engine_from_config
from .swap_gender_pronouns import remap_df, remap_df_swap, gender_name_to_id


def create_base_table(basetable_name : str, category_table : str, category_col : str = 'feat', category_name : str = 'PRONOUN',
    db : str = 'politeness'):
    """
    Create a base messages table with ids of the messages to be analyzed.
    This table doesn't require any information about the messages themselves, so
    it contains only the ids.

    Parameters
    ----------
    basetable_name
        The name of the basetable created with ids of messages to be transformed
    category_table
        The name of the category table from which to draw the ids of messages from a specified category
    category_col
        The name of the column under which to match the specified category
    category_name
        The name of the category from which to select messages
    db
        The name of the db

    """
    engine = engine_from_config(database = db)
    if basetable_name in engine.table_names():
        print(basetable_name, "already exists! Skipping creation...")
    else:
        with engine.connect() as conn:
            conn.execute(
                """CREATE TABLE {basetable_name} AS
                (
                SELECT {category_table}.group_id AS sid
                FROM {category_table}
                WHERE {category_table}.{category_col} = '{category_name}'
                );""".format(basetable_name = basetable_name, category_table = category_table,
                    category_col = category_col, category_name = category_name)
            )


def transform_1grams(onegram_table_name : str, basetable_name : str, 
                        gender_from_names: list, gender_to_name: str,
                        db : str = 'politeness') -> pd.DataFrame:
    """
    Collect all of the 1-grams from the messages to be analyzed, and perform the
    specified gender swaps.

    Parameters
    ----------
    onegram_table_name
        The name of the onegram table to be used to perform gender swaps
    basetable_name
        The name of the basetable containing ids of messages to be transformed
    gender_from_name
        The names of the genders whose pronouns will be replaced
    gender_to_name
        The name of the target gender for the replaced pronouns
    db
        The name of the db

    Returns
    -------
    A pandas DataFrame which contains the 1-gram table with the `feat` column
    containing the transformed pronouns.
    """
    engine = engine_from_config(database = db)
    with engine.connect() as conn:
        df = pd.read_sql(
            """SELECT {onegram_table_name}.*
            FROM {onegram_table_name} INNER JOIN {basetable_name}
            ON {onegram_table_name}.group_id={basetable_name}.sid;""".format(onegram_table_name = onegram_table_name, 
                basetable_name = basetable_name),
            conn,
        )

    gender_from_ids = list(map(gender_name_to_id, gender_from_names))
    df = remap_df(
        df, gender_from_ids, gender_name_to_id(gender_to_name)
    )
    return df


def transform_1grams_swap(onegram_table_name : str, basetable_name : str, 
                        a_gender: str, b_gender: str,
                        db : str = 'politeness') -> pd.DataFrame:
    """
    Collect all of the 1-grams from the messages to be analyzed, and perform the
    specified gender swaps.

    Parameters
    ----------
    onegram_table_name
        The name of the onegram table to be used to perform gender swaps
    basetable_name
        The name of the basetable containing ids of messages to be transformed
    a_gender, b_gender
        The genders between which to swap pronouns
    db
        The name of the db

    Returns
    -------
    A pandas DataFrame which contains the 1-gram table with the `feat` column
    containing the transformed pronouns.
    """
    engine = engine_from_config(database = db)
    with engine.connect() as conn:
        df = pd.read_sql(
            """SELECT {onegram_table_name}.*
            FROM {onegram_table_name} INNER JOIN {basetable_name}
            ON {onegram_table_name}.group_id={basetable_name}.sid;""".format(onegram_table_name = onegram_table_name, 
                basetable_name = basetable_name),
            conn,
        )

    df = remap_df_swap(
        df, gender_name_to_id(a_gender), gender_name_to_id(b_gender)
    )
    return df


def create_transformed_1gram_table(transformed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate an uploadable 1gram table with transformed gender tokens

    Parameters
    ----------
    transformed_df
        The name of the df created by the transform step

    Returns
    -------
    A pandas DataFrame which contains the 1-gram table with the `feat` column
    containing the transformed pronouns.

    """
    df = transformed_df.copy()
    df = df.drop("transformation", axis = 1)
    return df



def create_tranformation_metadata_table(transformed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate an uploadable metadata table that stores information on which messages 
    were transformed and in what direction

    Parameters
    ----------
    transformed_df
        The name of the df created by the transform step

    Returns
    -------
    A pandas DataFrame which contains the metadata

    """

    df = transformed_df.copy()
    df = df[df.transformation != "null"]
    df = df[['group_id', 'transformation']].drop_duplicates()

    return df

    

def calculate_transformed_scores(transformed_df: pd.DataFrame, 
    transformed_1gram_table_name : str, 
    basetable_name : str,
    lexicon_table_name : str,
    db : str = 'politeness'):
    """
    Create a 1-gram table from the `transformed_df`, and runs dlatk to compute
    lexicon scores on the transformed messages.

    Parameters
    ----------
    transformed_df
        A DataFrame which contains the 1-grams to be analyzed with the lexicon.
        This should be the output of `transform_1grams`.
    transformed_1gram_table_name
        The name for the gender swapped 1gram table to be uploaded to sql
    basetable_name
        The name of the basetable created for this task
    lexicon_table_name
        The name of the lexicon table to be applied to the basetable
    db
        The name of the db
    """

    table_name = "{transformed_1gram_table_name}".format(transformed_1gram_table_name = transformed_1gram_table_name)
    #upload 1grams to table
    store_table(transformed_df, table_name, db)

    # Use dlatk to create lex table
    # TODO: Call dlatk from python?
    dlatk_command = "~/dlatkInterface.py -d {db} -t {basetable_name} -c sid --add_lex_table -l {lexicon_table_name} --weighted_lexicon".format(
        db = db, basetable_name = basetable_name, lexicon_table_name = lexicon_table_name)
    # os.system(dlatk_command)
    print(dlatk_command)


def compare_transform_effect(old_score_table : str, new_score_table : str, message_table : str, csv_name : str, db : str = 'politeness'):
    """
    Select the message id, the lexicon-predicted scores of the original message,
    and the lexicon-predicted scores of the transformed message. Compute the
    difference between the original and transformed scores, and save the whole
    DataFrame to a csv file.

    Parameters
    ----------
    old_score_table
        The name of the onegram table to be used to perform gender swaps
    new_score_table
        The name of the basetable containing ids of messages to be transformed
    message_table
        The name of the gender whose pronouns will be replaced
    csv_name
        The name of the target gender for the replaced pronouns
    db
        The name of the db

    Returns
    -------
    A pandas DataFrame which contains original message text along with the annotated, old and transformed lexicon score after
    gender swapping

    """
    sql = """SELECT {old_score_table}.group_id AS 'id',
    {message_table}.message as message,
    {message_table}.stdzd_avg as annotated_score,
    {old_score_table}.group_norm AS 'original_score',
    {new_score_table}.group_norm AS 'transformed_score'

    FROM {old_score_table} INNER JOIN {new_score_table}
    ON {old_score_table}.group_id={new_score_table}.group_id
    LEFT JOIN {message_table} 
    ON {old_score_table}.group_id={message_table}.sid""".format(
        old_score_table = old_score_table, new_score_table = new_score_table, message_table = message_table)
    engine = engine_from_config(database = db)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
        df["score_difference"] = df["original_score"] - df["transformed_score"]
        df.to_csv(
            "{csv_name}.csv".format(csv_name = csv_name), index=False
        )
        return df

def store_table(df : pd.DataFrame, table_name : str, db : str = 'politeness'):
    """
    Upload a table to the database

    Parameters
    ----------
    df
        The name of the df to be uploaded
    table_name
        The name to be used for storing the table
    db
        The database to be uploaded to

    """

    engine = engine_from_config(database = db)
    # Upload 1grams to table
    with engine.connect() as conn:
        df.to_sql(table_name, conn, index=False, if_exists="replace")


def read_table(table_name : str, db : str = 'politeness') -> pd.DataFrame:
    """
    Read a table from the database

    Parameters
    ----------
    table_name
        The name to be read
    db
        The database to be accessed

    Returns
    ----------

    A pandas dataframe

    """

    engine = engine_from_config(database = db)
    with engine.connect() as conn:
        sql = "SELECT * FROM `{}`.`{}`".format(db, table_name)
        df = pd.read_sql(sql, conn)
    return df
