source("nlp_tools.r")

data = read_csv("IMDB Dataset.csv")

run_topic_model(data, text_column = "review", output_folder = "movie_lda", sample = FALSE, sample_size = 200, n_topics = 50)
