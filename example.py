import pandas as pd
import nlp_tools as nlp
import openai

# I have a dataset of movie reviews
df = pd.read_csv('IMDB Dataset.csv')
# add index as a column
df['index'] = df.index

df_sample = df.head(10)

# Deidentify
nlp.replace_ner(df_sample.review[0])

deidentified_df = nlp.deidentify_dataframe(df_sample, input_text_column="review", output_text_column="deidentified_review")

# LIWC
nlp.process_dataframe_with_liwc(deidentified_df, text_column="deidentified_review", id_column="index")

# Custom dictionary
dictionary = [r"horror", r"gore", r"scary"]
dictionary = [r"love"]
nlp.count_dictionary_matches(deidentified_df, text_column = "deidentified_review", mode = "count",dictionary=dictionary)
nlp.count_dictionary_matches(deidentified_df, text_column = "deidentified_review", mode = "proportion",dictionary=dictionary)
# Topic Modeling

# GPT Rating
df['id']  = df.index
openai.api_key = "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"
custom_prompt = """
I will show you a list of movie reviews.
Please respond with a 1 if the participant liked the movie.
Respond with a 0 if the participant did not like the movie.
Your response should be formatted as a python dictionary, with spaces for an explanation for your rating and an your rating.
"""
ratings_df = nlp.generate_ratings(deidentified_df, "index", "review", custom_prompt, "data/ratings_binary", verbose=True)
ratings_df.to_csv("ratings_binary.csv", index=False)