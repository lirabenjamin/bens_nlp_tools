# Add LIWC
import pandas as pd

def process_dataframe_with_liwc(df: pd.DataFrame, id_column:str, text_column: str, save_to_csv: bool = False, output_filename: str = 'liwc_output.csv') -> pd.DataFrame:
    import pandas as pd
    import subprocess
    import os
    # First, save the DataFrame to a temporary CSV file
    temp_filename = 'temp_for_liwc.csv'
    df.to_csv(temp_filename, index=False)

    # Command to run LIWC on the CSV file
    cmd_to_execute = ["LIWC-22-cli",
                      "--mode", "wc",
                      "--input", temp_filename,
                      "--row-id-indices", str(df.columns.get_loc(id_column)+1),  # Assuming the first column (index 0) is the identifier
                      "--column-indices", str(df.columns.get_loc(text_column)+1),  # Index of the text column
                      "--output", output_filename if save_to_csv else 'liwc_temp_output.csv']

    # Execute the command
    result = subprocess.call(cmd_to_execute)

    # Check if the command was successful
    if result != 0:
        os.remove(temp_filename)
        raise RuntimeError("Error occurred while running LIWC-22. Ensure the LIWC-22 application is running.")

    # Read the LIWC output into a pandas DataFrame
    if os.path.exists(output_filename if save_to_csv else 'liwc_temp_output.csv'):
        liwc_output = pd.read_csv(output_filename if save_to_csv else 'liwc_temp_output.csv')
    else:
        os.remove(temp_filename)
        raise FileNotFoundError(f"Expected output file {output_filename if save_to_csv else 'liwc_temp_output.csv'} not found.")
    
    # Clean up temporary files
    os.remove(temp_filename)
    if not save_to_csv:
        os.remove('liwc_temp_output.csv')

    return liwc_output

df = pd.DataFrame({'some text' : ['this is some text', 'this is some more text'], 'id' : [1, 2]})

process_dataframe_with_liwc(df,'id', 'some text', save_to_csv=True, output_filename='liwc_output.csv')
# custom dictionary: score with sum/ presence absence, tfidf or not. preprocess or not.

# Sanitize text
import pandas as pd
import spacy
from spacy import displacy
import en_core_web_trf
from tqdm import tqdm
import swifter  # for faster pandas apply
import spacy_transformers

# Initialize spacy model
nlp = en_core_web_trf.load()

def replace_ner(text, level=1):
    """
    Replace named entities in a text.
    
    Parameters:
    - text (str): Input text
    - level (int): Level of deidentification. 
                   1 = Replace only persons with first and last names.
                   2 = Replace all persons.
                   (More levels can be added as required)
                   
    Returns:
    - str: Deidentified text
    """
    doc = nlp(text)
    clean_text = text
    for ent in reversed(doc.ents):
        if ent.label_ == "PERSON":
            if level == 1 and " " in ent.text:
                clean_text = clean_text[:ent.start_char] + "[PERSON]" + clean_text[ent.end_char:]
            elif level == 2:
                clean_text = clean_text[:ent.start_char] + "[PERSON]" + clean_text[ent.end_char:]
    return clean_text

replace_ner("John Smith is a person.")

def deidentify_dataframe(df, text_column, id_column=None, level=1, save=False, output_filename="deidentified_data.parquet"):
    """
    Deidentify a pandas dataframe.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe
    - text_column (str): Name of the text column to deidentify
    - id_column (str, optional): Name of the ID column
    - level (int): Level of deidentification. Refer to `replace_ner` for levels.
    - save (bool): Whether to save the resulting dataframe
    - output_filename (str): Name of the output file if `save` is True
    
    Returns:
    - pd.DataFrame: Deidentified dataframe
    """
    tqdm.pandas()  # Initialize tqdm for pandas
    df['redacted'] = df[text_column].swifter.progress_bar(enable=True).apply(lambda x: replace_ner(x, level))
    
    if save:
        df.to_parquet(output_filename)
        
    return df

# Add topics

# Add GPT rating
def generate_ratings(data: pd.DataFrame, id_col: str, text_col: str, prompt: str, output_dir: str, verbose: bool = False) -> pd.DataFrame:
    import openai
    import concurrent.futures
    import os
    import datetime
    import ast
    
    # Check OpenAI API key
    if not openai.api_key:
        raise ValueError("OpenAI API key not set. Please set it before calling this function.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def rate_conversation(id, conversation):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=1,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Here are the participant comments:\n{conversation}"},
            ]
        )
        result = response.choices[0].message.content
        if verbose:
            print(result)
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        with open(f"{output_dir}/{id}_{now}_temp1.txt", "w") as f:
            f.write(result)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(rate_conversation, data[id_col], data[text_col])
    
    def read_all_files_to_dataframe(directory):
        all_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.txt')]
        df_list = []

        for filename in all_files:
            with open(filename, 'r') as f:
                content = f.read()
                df_list.append({"filename": filename, "content": content})

        return pd.DataFrame(df_list)

    combined_df = read_all_files_to_dataframe(output_dir)
    combined_df.columns = ["id", "content"]
    combined_df["id"] = combined_df["id"].str.replace(f"{output_dir}/", "")
    combined_df["id"] = combined_df["id"].str.replace(".txt", "")
    combined_df[["id", "timestamp", "temp"]] = combined_df["id"].str.split("_", expand=True)

    # Unroll the dictionary
    combined_df["content"] = combined_df["content"].apply(ast.literal_eval)
    df = pd.DataFrame(combined_df.content.tolist())

    # Combine df and combined_df
    df = pd.concat([combined_df, df], axis=1)
    df = df.drop(["content", "timestamp", "temp"], axis=1)
    
    # Join df with data on id
    df['id'] = df['id'].astype(int)
    df = df.merge(data, on=id_col)

    return df
