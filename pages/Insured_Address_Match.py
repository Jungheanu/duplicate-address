import pandas as pd
import numpy as np
import base64
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to normalize addresses
def normalize_address(address):
    address = address.lower().strip()
    address = address.replace('.', '').replace(',', '')  # Remove punctuation
    # Replace common abbreviations
    abbreviations = {
        ' st ': ' street ', ' ave ': ' avenue ', ' rd ': ' road ', ' blvd ': ' boulevard ',
        ' dr ': ' drive ', ' ct ': ' court ', ' ln ': ' lane ', ' fl ': ' floor ',
        ' ste ': ' suite ', ' apt ': ' apartment ', ' pkwy ': ' parkway ', ' hwy ': ' highway '
    }
    for abbr, full in abbreviations.items():
        address = address.replace(abbr, full)
    return address

# Function to compute similarity
def compute_similarity(df, batch_size=1000):
    # Create combined address fields for Address1 and Address2
    df['Combined_Address1'] = df['Address1'].astype(str).fillna('') + ' ' + df['City'].astype(str).fillna('') + ' ' + df['State'].astype(str).fillna('') + ' ' + df['Zip'].astype(str).fillna('')
    df['Combined_Address2'] = df['Address2'].astype(str).fillna('') + ' ' + df['City'].astype(str).fillna('') + ' ' + df['State'].astype(str).fillna('') + ' ' + df['Zip'].astype(str).fillna('')

    # Normalize both combined address fields
    df['Combined_Address1'] = df['Combined_Address1'].apply(normalize_address)
    df['Combined_Address2'] = df['Combined_Address2'].apply(normalize_address)

    # Mark rows with invalid addresses (where both Address1 and Address2 are blank)
    invalid_address_mask = df['Address1'].fillna('').str.strip().eq('') & df['Address2'].fillna('').str.strip().eq('')
    df.loc[invalid_address_mask, 'Combined_Address1'] = np.nan  # or some invalid marker
    df.loc[invalid_address_mask, 'Combined_Address2'] = np.nan  # or some invalid marker

    # Initialize TF-IDF Vectorizer and process valid address fields
    tfidf_vectorizer = TfidfVectorizer(dtype=np.float32)
    # Use dropna() to exclude invalid addresses from TF-IDF vectorization
    tfidf_matrix1 = tfidf_vectorizer.fit_transform(df['Combined_Address1'].dropna())
    tfidf_matrix2 = tfidf_vectorizer.transform(df['Combined_Address2'].dropna())


    # Initialize TF-IDF Vectorizer and process both address fields
    tfidf_vectorizer = TfidfVectorizer(dtype=np.float32)
    tfidf_matrix1 = tfidf_vectorizer.fit_transform(df['Combined_Address1'])
    tfidf_matrix2 = tfidf_vectorizer.transform(df['Combined_Address2'])

    # Initialize an empty list to store DataFrames for matches
    matches_list = []

    # Process in batches
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        
        # Compute cosine similarity for each combination of address fields
        batch_cos_sim_matrix1 = cosine_similarity(tfidf_matrix1[start:end], tfidf_matrix1)
        batch_cos_sim_matrix2 = cosine_similarity(tfidf_matrix2[start:end], tfidf_matrix2)
        batch_cos_sim_matrix_cross = cosine_similarity(tfidf_matrix1[start:end], tfidf_matrix2)

        # Process similarity results for each matrix
        for batch_cos_sim_matrix in [batch_cos_sim_matrix1, batch_cos_sim_matrix2, batch_cos_sim_matrix_cross]:
            cos_sim_df = pd.DataFrame(batch_cos_sim_matrix, index=df.index[start:end], columns=df.index)

            # Identify index pairs with high cosine similarity scores in this batch
            matches = cos_sim_df.stack().reset_index()
            matches.columns = ['Index1', 'Index2', 'CosineSim']
            matches = matches[matches['Index1'] != matches['Index2']]  # Remove self-comparison

            # Filter matches based on a threshold
            threshold = 0.8  # Can be adjusted
            potential_matches = matches[matches['CosineSim'] >= threshold]
            matches_list.append(potential_matches)

    # Concatenate all matches DataFrames
    all_matches = pd.concat(matches_list, ignore_index=True)

    # Filter by different InsuredID and remove duplicates
    all_matches = all_matches.drop_duplicates(subset=['Index1', 'Index2'])
    all_matches = all_matches.merge(df[['InsuredID']], left_on='Index1', right_index=True)
    all_matches = all_matches.merge(df[['InsuredID']], left_on='Index2', right_index=True, suffixes=('_1', '_2'))
    all_matches = all_matches[all_matches['InsuredID_1'] != all_matches['InsuredID_2']]

    # Add 'Y' to rows with matches, 'N' otherwise
    matched_indices = set(all_matches['Index1']).union(set(all_matches['Index2']))
    df['MatchFlag'] = 'N'
    df.loc[df.index.isin(matched_indices), 'MatchFlag'] = 'Y'

    return df

# Function to compute the hit ratio
def compute_hit_ratio(df):
    # Calculate the ratio of 'Y' in MatchFlag for each AgentName
    hit_ratio = df.groupby('AgentName')['MatchFlag'].apply(lambda x: (x == 'Y').sum() / len(x))
    
    # Convert Series to DataFrame and reset index
    hit_ratio = hit_ratio.reset_index()
    hit_ratio.columns = ['AgentName', 'HitRatio']

    return hit_ratio

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some browsers need base64 encoding
    href = f'<a href="data:file/csv;base64,{b64}" download="similar_addresses.csv">Download CSV file</a>'
    return href

# Streamlit app interface
st.title('Address Matching App')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Convert address components to string and concatenate
    df['Combined_Address'] = df['Address1'].astype(str).fillna('') + ' ' + \
                             df['Address2'].astype(str).fillna('') + ' ' + \
                             df['City'].astype(str).fillna('') + ' ' + \
                             df['State'].astype(str).fillna('') + ' ' + \
                             df['Zip'].astype(str).fillna('')
    df['Combined_Address'] = df['Combined_Address'].apply(normalize_address)


    st.write('Data Preview:', df.head())

 # Button to find matches
    if st.button('Find Similar Addresses'):
        with st.spinner('Finding similar addresses...'):
            df = compute_similarity(df)
            hit_ratio_df = compute_hit_ratio(df)
            df = df.merge(hit_ratio_df, on='AgentName', how='left')
            st.dataframe(df[['AccountID', 'InsuredID', 'AgentName', 'Insured', 'Combined_Address', 'MatchFlag', 'HitRatio']], height=800)

            # Generate download link for the processed DataFrame
            csv = df.to_csv(index=False).encode('utf-8')
            b64 = base64.b64encode(csv).decode()  # Encode to base64
            href = f'<a href="data:file/csv;base64,{b64}" download="similar_addresses.csv">Download processed data as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
else:
    st.write("Please upload a CSV file to proceed.")
