run_topic_model <- function(data, text_column, output_folder, sample = FALSE, sample_size = 200, n_topics = 50) {
  # Libraries
  library(stm)
  library("tidytext")
  library('lda')
  library("quanteda")
  library("dplyr")
  library("ggplot2")
  library(tidyverse)
  library(glue)
  library(LDAvis)
  library(trelliscopejs)

  # Sampling
  if(sample) {
    data <- data %>% slice_sample(n = sample_size)
  }

  # Filtering based on character count
  data <- data %>%
    filter(nchar(as.character(!!sym(text_column))) > 50)

  # Convert text, add spaces after punctuations, and remove numbers 
  raw_text <- as.character(data[[text_column]])
  raw_text <- gsub(".", ". ", raw_text, fixed = TRUE)
  raw_text <- gsub(",", ", ", raw_text, fixed = TRUE)
  raw_text <- gsub("[0-9]+", " ", raw_text)
  data[[text_column]] <- raw_text

  # Check for unicode/UTF-8 errors
  data[[text_column]] %>% 
    iconv("latin1", "ASCII", sub="I_WAS_NOT_ASCII") %>% 
    grep("I_WAS_NOT_ASCII", .)

  # Process data and create a document-feature matrix
  text_dfm <- dfm(as.character(data[[text_column]]), 
                  remove = stopwords("english"), 
                  remove_punct = TRUE,
                  stem = TRUE, 
                  verbose = FALSE, 
                  tolower = TRUE)
  processed_text <- convert(text_dfm, to = "stm")

  # Prepare corpus for processing and analysis
  processed_out <- prepDocuments(processed_text$documents, 
                                 processed_text$vocab,
                                 processed_text$meta)
  processed_docs <- processed_out$documents
  processed_vocab <- processed_out$vocab
  processed_meta <- processed_out$meta

  dir.create(glue("{output_folder}"), showWarnings = FALSE)


  # Conduct STM
  processed_fit <- stm(processed_docs, 
                       processed_vocab, 
                       K = n_topics,
                       data = processed_meta,
                       seed = 325,
                       max.em.its = 20, 
                       init.type = "LDA",
                       verbose = TRUE)


  # Post-processing and visualization
  topic_labels = labelTopics(processed_fit)
  gamma <- tidy(processed_fit, matrix = "gamma")
  gamma <- gamma %>%
    spread(key = topic, value = gamma)
  colnames(gamma) <- c("document_number", paste0("t_", 1:n_topics))
  essays_w_gamma <- cbind(processed_meta, gamma %>% select(-document_number))


  tidytext::tidy(processed_fit) |> 
    group_by(topic) |> 
    slice_max(n = 10, order_by = beta) |> 
    mutate(term = reorder_within(term, beta, topic)) |> 
    ggplot(aes(beta, term)) +
    geom_col() +
    scale_y_reordered() +
    trelliscopejs::facet_trelliscope(facets = "topic", scales = "free", path = glue("{output_folder}/topic_labels"))
    
    toLDAvis(processed_fit,processed_docs, out.dir = glue("{output_folder}/LDAvis/"), reorder.topics = TRUE)

  # save enbirotnment
  save.image(glue("{output_folder}/environment.RData"))
}

data = tibble(text = c(
  "I love books about nature and other things. Nature and books are cool, but I also like other things. But mostly nature and books.",
  "The best book is about nature. I love nature. Nature is the best. I love nature and books. Nature and books are the best.",
  "I love my parents. My parents love me. I love my parents and my parents love me. I love my parents and my parents love me and I love my parents. ",
  "I love my parents and my parents love me",
  "I love my parents and my parents love me and I love my parents"
))

# # Example usage:
# data <- arrow::read_parquet("full.parquet")
# run_topic_model(data, "response", "test", n_topics = 20, sample = TRUE, sample_size = 1000)