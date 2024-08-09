import pandas as pd
from apyori import apriori
from ucimlrepo import fetch_ucirepo 
import streamlit as st
# Fetch the Zoo dataset
zoo = fetch_ucirepo(id=111) 
  
# Extract features
df = zoo.data.features

# Create Bins for each category
df['legs_bins'] = pd.cut(df['legs'], bins=6, labels=['no leg', '2 leg', '4 leg', '5 leg', '6 leg', '8 leg'])
df['hair_bins'] = pd.cut(df['hair'], bins=2, labels=['no hair', 'has hair'])
df['feather_bins'] = pd.cut(df['feathers'], bins=2, labels=['no feathers', 'has feathers'])
df['eggs_bins'] = pd.cut(df['eggs'], bins=2, labels=['no eggs', 'lays eggs'])
df['milk_bins'] = pd.cut(df['milk'], bins=2, labels=['no milk', 'has milk'])
df['airborne_bins'] = pd.cut(df['airborne'], bins=2, labels=['not airborne', 'airborne'])
df['aquatic_bins'] = pd.cut(df['aquatic'], bins=2, labels=['not aquatic', 'aquatic'])
df['predator_bins'] = pd.cut(df['predator'], bins=2, labels=['not predator', 'predator'])
df['toothed_bins'] = pd.cut(df['toothed'], bins=2, labels=['not toothed', 'toothed'])
df['backbone_bins'] = pd.cut(df['backbone'], bins=2, labels=['no backbone', 'has backbone'])
df['breathes_bins'] = pd.cut(df['breathes'], bins=2, labels=['does not breathe', 'breathes'])
df['venomous_bins'] = pd.cut(df['venomous'], bins=2, labels=['not venomous', 'venomous'])
df['fins_bins'] = pd.cut(df['fins'], bins=2, labels=['no fins', 'has fins'])
df['tail_bins'] = pd.cut(df['tail'], bins=2, labels=['no tail', 'has tail'])
df['domestic_bins'] = pd.cut(df['domestic'], bins=2, labels=['not domestic', 'domestic'])
df['catsize_bins'] = pd.cut(df['catsize'], bins=2, labels=['not big cat', 'bigcat'])
st.title('Observations on Animal characteristics')
st.header('Animal Dataset')
st.write(df)

st.header('Discretizing the data')
# Create set with discretized bins
animals = df[['legs_bins', 'hair_bins', 'feather_bins', 'eggs_bins', 'milk_bins', 'airborne_bins', 'aquatic_bins', 'predator_bins', 'toothed_bins', 'backbone_bins', 'breathes_bins', 'venomous_bins', 'fins_bins', 'tail_bins', 'domestic_bins', 'catsize_bins']].values.tolist()

st.write(animals)

st.header('Conclusions')
# Apply Apriori algorithm
itemsets_list = apriori(animals, min_support=0.75
                        , min_confidence=0.7)

# Print out the discovered rules
for itemset in itemsets_list:
    for rule_i in range(len(itemset.ordered_statistics)):
        output_text = f"{list(itemset.ordered_statistics[rule_i].items_base)} --> {list(itemset.ordered_statistics[rule_i].items_add)} " \
                      f"Support: {itemset.support}, Confidence: {itemset.ordered_statistics[rule_i].confidence}"
        st.text(output_text)
