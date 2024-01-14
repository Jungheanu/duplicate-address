import streamlit as st
import pandas as pd
from thefuzz import fuzz, process

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

# Vectorized function to pre-hash addresses
def pre_hash_addresses(df, hash_len=5):
    # Create a quick hash for preliminary comparison
    df['AddressHash'] = df['NormalizedAddress'].str[:hash_len]
    return df

# Vectorized function to find preliminary similar addresses based on hash
def preliminarily_find_similar_addresses(df):
    # Create a DataFrame to store potential matches
    potential_matches = pd.DataFrame()
    
    # Get unique hashes
    unique_hashes = df['AddressHash'].unique()
    
    # For each unique hash, find all entries with that hash and mark them for further comparison
    for hash_value in unique_hashes:
        same_hash_df = df[df['AddressHash'] == hash_value]
        if len(same_hash_df) > 1:
            # Cartesian product of all pairs with the same hash, excluding same rows
            product_df = same_hash_df.merge(same_hash_df, on='AddressHash')
            potential_matches = pd.concat([potential_matches, product_df])
    
    potential_matches = potential_matches[potential_matches['InsuredID_x'] != potential_matches['InsuredID_y']]
    return potential_matches.reset_index(drop=True)

# Function to apply fuzz.partial_ratio to the preliminary matches
def apply_fuzzy_matching_to_preliminary(df, threshold=80):
    df['FuzzyScore'] = df.apply(
        lambda x: fuzz.partial_ratio(x['NormalizedAddress_x'], x['NormalizedAddress_y']), axis=1
    )
    df['MatchFlag'] = df['FuzzyScore'].apply(lambda x: 'Y' if x >= threshold else 'N')
    return df

# Streamlit app interface
st.title('Address Matching App')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write('Data Preview:', df.head())
        
        # Button to find matches
        if st.button('Find Similar Addresses'):
            with st.spinner('Finding similar addresses...'):
                matched_df = find_similar_addresses(df)
                st.write('Matches Found:', matched_df)
    except Exception as e:
        st.error(f"An error occurred: {e}")

