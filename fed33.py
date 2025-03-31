# Working replication of IDENTIFYING MONETARY POLICY SHOCKS:A NATURAL LANGUAGE APPROACH by S. Borağan Aruoba and Thomas Drechsel.
import os
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams
from collections import Counter, defaultdict
import pandas as pd
import re
from PyPDF2 import PdfReader
from datetime import datetime
import matplotlib.pyplot as plt


# old_directory = "C:\Users\lenze"
# Specify the path of the directory you want to change to. This is where the FED documents are located.
new_directory = r"D:\FOMC"
# Change the current working directory
os.chdir(new_directory)
# Verify the change
print("Current working directory:", os.getcwd())

#############
# Program code to read all pdfs in D:\FOMC
# Set your folder path containing the Federal Reserve PDFs
folder_path = r"D:\FOMC"
# Test path for faster trial and error.
#folder_path = r"D:\FOMC\test"

date_pattern = r"\d{4}_\d{2}_\d{2}"  # Matches YYYY_MM_DD format
grouped_data = {}

# Get a list of all PDFs in the folder
pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
#############

# Loop through each PDF file and read it
for pdf_file in pdf_files:
    match = re.search(date_pattern, os.path.basename(pdf_file))
    file_date = match.group()  # Extract date from filename
    file_datetime = datetime.strptime(file_date, "%Y_%m_%d").date()  # Convert to date format
    
    pdf_path = os.path.join(folder_path, pdf_file)
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        # Extract text from all pages in the PDF
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()  # Concatenate text from each page
    # Group text by date
    if file_datetime not in grouped_data:
        grouped_data[file_datetime] = " "

    grouped_data[file_datetime] += full_text + "\n"

# Group PDFs by date before processing
merged_text_by_date = {}
for date, text in grouped_data.items():
    merged_text_by_date[date] = merged_text_by_date.get(date, "") + " " + text  # Merge PDFs with same date
    
def load_lm_master_dictionary(filepath):
    # Read CSV file (skip header if necessary)
    df = pd.read_csv(filepath)
    df['Word'] = df['Word'].astype('string').fillna('')
    
     # Convert to dictionary for quick lookups
    lm_lexicon = defaultdict(lambda: {"Negative": 0, "Positive": 0})
    
    for _, row in df.iterrows():
        word = row["Word"].lower()  # Convert to lowercase for matching
        lm_lexicon[word]["Negative"] = int(row["Negative"])
        lm_lexicon[word]["Positive"] = int(row["Positive"])
        
    return lm_lexicon

def compute_lm_master_sentiment(text, lm_lexicon):
    words = word_tokenize(text.lower())
    sentiment_score = 0  # Net sentiment adjustment
    
    for word in words:
        if word in lm_lexicon:
            sentiment_score += lm_lexicon[word]["Positive"]
            sentiment_score -= lm_lexicon[word]["Negative"] 

    return sentiment_score

# ngrams_with_n will be a list of tuples, all_ngrams.
def find_ngram_sentiments_master(text_data_by_date, ngrams_with_n, lm_lexicon):
    all_sentiment_data = []
    total_words_per_date = {}

    # Extract n-grams and n-values
    ngram_list, n_values = zip(*ngrams_with_n)
    ngram_set = set(ngram_list)  # Convert list to set for fast lookups

    # Process each date separately
    for date, text in text_data_by_date.items():
        words = word_tokenize(text.lower())
        total_words_per_date[date] = len(words)  # Count words once

        # Generate n-grams once for each n-value
        text_ngrams_dict = {n: list(ngrams(words, n)) for n in set(n_values)}

        sentiment_data = []

        for n_value in text_ngrams_dict:
            text_ngrams = text_ngrams_dict[n_value]  # Get precomputed n-grams
            
            for i, ngram in enumerate(text_ngrams):
                if ngram in ngram_set:  # Check if the n-gram is in the target list
                    start = max(i - 10, 0)
                    end = min(i + n_value + 10, len(words))

                    context_words = words[start:end]
                    context_text = " ".join(context_words)

                    # Compute sentiment
                    sentiment_score = compute_lm_master_sentiment(" ".join(ngram), lm_lexicon)
                    before_sentiment = compute_lm_master_sentiment(" ".join(words[start:i]), lm_lexicon)
                    after_sentiment = compute_lm_master_sentiment(" ".join(words[i + n_value:end]), lm_lexicon)

                    adjusted_sentiment = sentiment_score + before_sentiment + after_sentiment

                    sentiment_data.append({
                        "date": date,
                        "ngram": " ".join(ngram),
                        "context": context_text,
                        "base_sentiment": sentiment_score,
                        "before_sentiment": before_sentiment,
                        "after_sentiment": after_sentiment,
                        "adjusted_sentiment": adjusted_sentiment,
                        "n": n_value
                    })

        all_sentiment_data.extend(sentiment_data)  # Use extend instead of append

    # Convert list to DataFrame once
    full_df = pd.DataFrame(all_sentiment_data)

    # Map total words per date
    full_df["total_words"] = full_df["date"].map(total_words_per_date)

    # Fill missing sentiment scores
    full_df["total_sentiment"] = full_df["adjusted_sentiment"].fillna(0)

    return full_df

#############
# Loading the Master Dictionary from Loughran-McDonald
lm_filepath = "Loughran-McDonald_MasterDictionary_1993-2023.csv"  # Update with actual path
lm_lexicon = load_lm_master_dictionary(lm_filepath)

lm_positive_words = {word for word, scores in lm_lexicon.items() if scores["Positive"] > 0}
lm_negative_words = {word for word, scores in lm_lexicon.items() if scores["Negative"] > 0}

singles = [('borrowing',), ('brazil',), ('banks',), ('canada',), ('credit',), ('china',), ('consumption',), ('construction',), ('currencies',), ('deposits',), ('employment',), ('equipment',), ('euro',), ('exports',), ('germany',), ('hiring',), ('hours',), ('housing',), ('imports',), ('inflation',), ('inventories',), ('investment',), ('japan',), ('liquidity',), ('loans',), ('leasing',), ('lending',), ('machinery',), ('mexico',), ('mortgage',), ('output',), ('productivity',), ('profits',), ('recovery',), ('reserves',), ('savings',), ('spread',), ('structures',), ('tourism',), ('unemployment',), ('utilization',), ('wages',), ('weather',), ('yield',), ('income',), ('gdp',), ('cpi',), ('nairu',), ('services',), ('bonds',), ('economy',), ('outlays',), ('financing',), ('assets',), ('finance',), ('shipments',), ('capacity',), ('office',), ('computers',), ('industries',), ('producers',), ('supply',), ('homes',), ('sectors',), ('agriculture',), ('merchandise',), ('investors',), ('aircraft',), ('stocks',), ('buildings',), ('cash',), ('trucks',), ('semiconductors',), ('farm',), ('uncertainty',), ('households',), ('crop',), ('apparel',), ('steel',), ('automotive',), ('metals',), ('permits',), ('commerce',), ('transportation',), ('municipal',), ('commodities',), ('corporations',), ('liabilities',), ('consumers',), ('firms',), ('trading',), ('corn',), ('asia',), ('taxes',), ('software',), ('mining',), ('losses',), ('jobs',), ('cars',), ('depreciation',), ('recession',), ('france',), ('korea',), ('italy',), ('lumber',), ('volatility',), ('wheat',), ('livestock',), ('rents',), ('petroleum',), ('traffic',), ('fuel',), ('plants',), ('technology',), ('argentina',), ('cattle',), ('crisis',), ('utilities',), ('travel',), ('payrolls',), ('factory',), ('transfers',), ('drought',), ('gold',), ('salaries',), ('cotton',), ('coal',), ('philippines',), ('singapore',), ('taiwan',), ('thailand',), ('soybean',), ('swaps',), ('harvest',), ('environment',), ('deflator',), ('delinquencies',), ('chemicals',), ('mergers',), ('rigs',), ('indonesia',), ('political',), ('peso',), ('retirement',), ('tobacco',), ('hurricane',), ('equities',), ('russia',), ('workers',), ('contractors',), ('borrowers',), ('brazilian',), ('bank',), ('banking',), ('bankers',), ('canadian',), ('chinese',), ('export',), ('german',), ('hires',), ('houses',), ('import',), ('inventory',), ('investments',), ('japanese',), ('loan',), ('lenders',), ('mortgages',), ('profit',), ('saving',), ('spreads',), ('wage',), ('yields',), ('durables',), ('manufacturers',), ('manufacturer',), ('treasuries',), ('gnp',), ('service',), ('asset',), ('computer',), ('building',), ('builders',), ('truck',), ('semiconductor',), ('farmers',), ('autos',), ('automobile',), ('metal',), ('soybeans',), ('christmas',)
 ]

doubles = [('employment', 'cost'), ('aggregate, demand'), ('auto', 'sales'), ('bond', 'issuance'), ('budget', 'deficit'), ('business', 'activity'), ('business', 'confidence'), ('business', 'spending'), ('capital', 'expenditures'), ('consumer', 'confidence'), ('current', 'account'), ('debt', 'growth'), ('defense', 'spending'), ('delinquency', 'rates'), ('developing', 'countries'), ('domestic', 'demand'), ('drilling', 'activity'), ('durable', 'goods'), ('economic', 'growth'), ('energy', 'prices'), ('equity', 'issuance'), ('equity', 'prices'), ('euro', 'area'), ('exchange', 'rate'), ('federal', 'debt'), ('financial', 'conditions'), ('financial', 'developments'), ('fiscal', 'policy'), ('fiscal', 'stimulus'), ('food', 'prices'), ('foreign', 'economies'), ('gas', 'prices'), ('gasoline', 'prices'), ('government', 'purchases'), ('home', 'prices'), ('home', 'sales'), ('hourly', 'compensation'), ('household', 'debt'), ('household', 'spending'), ('import', 'prices'), ('industrial', 'production'), ('industrial', 'supplies'), ('inflation', 'compensation'), ('inflation', 'expectations'), ('initial', 'claims'), ('input', 'prices'), ('intermediate', 'materials'), ('international', 'developments'), ('labor', 'market'), ('manufacturing', 'activity'), ('manufacturing', 'firms'), ('monetary', 'aggregates'), ('mortgage', 'interest'), ('natural', 'rate'), ('net', 'exports'), ('new', 'orders'), ('nondefense', 'capital'), ('oil', 'prices'), ('output', 'gap'), ('potential', 'output'), ('price', 'pressures'), ('producer', 'prices'), ('refinancing', 'activity'), ('residential', 'investment'), ('retail', 'prices'), ('retail', 'sales'), ('retail', 'trade'), ('share', 'prices'), ('social', 'security'), ('stock', 'market'), ('trade', 'balance'), ('trade', 'deficit'), ('trade', 'surplus'), ('treasury', 'securities'), ('treasury', 'yield'), ('vacancy', 'rates'), ('wholesale', 'prices'), ('wholesale', 'trade'), ('yield', 'curve'), ('foreign', 'exchange'), ('nominal', 'gdp'), ('core', 'inflation'), ('motor', 'vehicles'), ('financial', 'institutions'), ('depository', 'institutions'), ('credit', 'standards'), ('consumer', 'prices'), ('crude', 'oil'), ('loan', 'demand'), ('united', 'kingdom'), ('money', 'market'), ('market', 'participants'), ('commercial', 'paper'), ('housing', 'starts'), ('housing', 'activity'), ('natural', 'gas'), ('consumer', 'goods'), ('balance', 'sheet'), ('financial', 'markets'), ('economic', 'indicators'), ('final', 'sales'), ('credit', 'quality'), ('international', 'transactions'), ('finished', 'goods'), ('latin', 'america'), ('economic', 'outlook'), ('domestic', 'developments'), ('oil', 'imports'), ('home', 'equity'), ('headline', 'inflation'), ('raw', 'materials'), ('holiday', 'season'), ('inflationary', 'pressures'), ('loan', 'officer'), ('health', 'care'), ('economic', 'expansion'), ('economic', 'data'), ('canadian', 'dollar'), ('corporate', 'profits'), ('insurance', 'companies'), ('wage', 'pressures'), ('market', 'expectations'), ('consumer', 'spending'), ('car', 'sales'), ('vehicle', 'sales'), ('real', 'activity'), ('business', 'conditions',), ('economic', 'conditions'), ('capital', 'spending'), ('consumer', 'sentiment'), ('durable', 'equipment'), ('energy', 'price'), ('equity', 'price'), ('stock', 'prices'), ('stock', 'price'), ('exchange', 'rates'), ('food', 'price'), ('gas', 'price'), ('gasoline', 'price'), ('home', 'price'), ('house', 'prices'), ('house', 'price'), ('hourly', 'earnings'), ('import', 'price'), ('input', 'price'), ('labor', 'markets'), ('manufacturing', 'sector'), ('mortgage', 'rates'), ('oil', 'price'), ('potential', 'gdp'), ('producer', 'price'), ('retail', 'price'), ('share', 'price'), ('treasury', 'bills'), ('treasury', 'security'), ('treasury', 'yields'), ('vacancy', 'rate'), ('wholesale', 'price'), ('nominal', 'gnp'), ('thrift', 'institutions'), ('lending', 'standards'), ('consumer', 'price'), ('imported', 'oil'), ('crude', 'materials'), ('district', 'banks'), ('import', 'prices'), ('inflation', 'expectations'), ('inflation', 'compensation'), ('core', 'inflation'), ('headline', 'inflation'), ('loan', 'rates'), ('mortgage', 'rate'), ('unemployment', 'insurance'), ('national', 'income'), ('income', 'tax',), ('foreign', 'gdp'), ('asset', 'purchases'), ('oil', 'price'), ('oil', 'prices'), ('commodity', 'prices'), ('commodity', 'price')

]

triples = [('advanced', 'foreign', 'economies'), ('commercial', 'real', 'estate'), ('compensation', 'per', 'hour'), ('domestic', 'final', 'purchases'), ('domestic', 'financial', 'developments'), ('emerging', 'market', 'economies'), ('foreign', 'industrial', 'countries'), ('gross', 'domestic', 'purchases'), ('household', 'net', 'worth'), ('international', 'financial', 'transactions'), ('labor', 'force', 'participation'), ('major', 'industrial', 'countries'), ('market', 'interest', 'rates'), ('nondefense', 'capital', 'goods'), ('output', 'per', 'hour'), ('real', 'estate', 'activity'), ('real', 'estate', 'market'), ('real', 'interest', 'rate'), ('real', 'interest', 'rates'), ('residential', 'real', 'estate'), ('unit', 'labor', 'cost'), ('unit', 'labor', 'costs'), ('money', 'market', 'mutual'), ('foreign', 'net', 'purchases'), ('real', 'estate', 'markets'), ('gross', 'domestic', 'product'), ('gross', 'national', 'product'), ('foreign', 'direct', 'investment'), ('money', 'market', 'mutual')
]

# Combine all ngrams into a list of tuples (ngram, n)
all_ngrams = [(ngram, 1) for ngram in singles] + \
             [(ngram, 2) for ngram in doubles] + \
             [(ngram, 3) for ngram in triples]
        
#############

# Perform sentiment analysis by date
for date, text in merged_text_by_date.items():

# NLP section (cleaning pdf text)
    sent = sent_tokenize(text)
    words_1 = [word_tokenize(t) for t in sent]
    list_words = sum(words_1,[])
    token_words = [word for word in list_words if not (len(word) == 1 and word.lower() not in {"a", "i"})]  # Remove single letters
    token_words = [word for word in token_words if re.search(r'[aeiou]', word, re.I)]  # Remove gibberish words
    low_words = [w.lower() for w in token_words]
    remove_words = [w for w in low_words if w not in stopwords.words('english')]
    punc_words = [w for w in remove_words if w.isalnum()]
     
    common_fixes = {
        'devel': 'development',
        'reserv' : 'reserves',
        'dom' : 'domestic',
        'purch' : 'purchases',
        'prod' : 'product',
        'cons' : 'consumption',
        'prices3' : 'prices',
        'econ' : 'economic',
        'int' : 'international',
        'pe' : 'percent'
    }
        
    corrected_tokens = [common_fixes[word] if word in common_fixes else word for word in punc_words]
    unique_string_v2=(" ").join(corrected_tokens)
    merged_text_by_date[date] += unique_string_v2
    
# Create dataframe of sentiments from cleaned text, ngram lists, and Loughran-McDonald dictionary
df_all = find_ngram_sentiments_master(merged_text_by_date, all_ngrams, lm_lexicon)

# Insert scaling and standardizing here
# Aggregate by date and ngram. Call it total_sentiment which is the sum of adjusted_sentiment by date and ngram.
ngram_sentiment_df = df_all.groupby(["date", "ngram"]).agg(
    total_sentiment=("adjusted_sentiment", "sum"),
    total_words=("total_words", "first"),  # Use "first" since total_words is constant per date
    base_sentiment=("base_sentiment", "sum")
).reset_index()

# Get all unique dates and ngrams
all_dates = ngram_sentiment_df["date"].unique()
all_ngrams = ngram_sentiment_df["ngram"].unique()

# Create full DataFrame with all (date, ngram) combinations
full_index = pd.MultiIndex.from_product([all_dates, all_ngrams], names=["date", "ngram"])
full_df = pd.DataFrame(index=full_index).reset_index()

# Merge the full grid with ngram_sentiment_df to ensure all pairs exist
ngram_sentiment_df = full_df.merge(ngram_sentiment_df, on=["date", "ngram"], how="left")

# Some words aren't found in FOMC documents (missing/NaN), but still receive "0" sentiment scores. Therefore, when finding scaled and standardized sentiment, scaled_sentiment = total_sentiment / total_words = 0 / NaN = NaN. If the count of total_words is 0 (it shouldn't be, but for testing with a small number of FOMC documents we may not acquire all of the ngrams from Drechsel/Arouba), then we fill total_words with 1 instead of 0.

# Fill missing sentiment scores with 0 BEFORE standardization (maybe delete if previous line works)
ngram_sentiment_df["total_sentiment"] = ngram_sentiment_df["total_sentiment"].fillna(0)
    
# Scale and standardize. Divide total_sentiment by total_words and call it scaled_sentiment. Then standardize it.
ngram_sentiment_df["total_words"] = ngram_sentiment_df["total_words"].fillna(1)

# Define n-grams to include and exclude as dictionaries
include_dict = {
    "borrowing": ["borrowers"],
    "brazil": ["brazilian"],
    "banks": ["bank", "banking", "bankers"],
    "canada": ["canadian"],
    "china": ["chinese"],
    "consumption": [("consumer", "spending")],
    "exports": ["export"],
    "germany": ["german"],
    "hiring": ["hires"],
    "housing": ["houses"],
    "imports": ["import"],
    "inventories": ["inventory"],
    "investment": ["investments"],
    "japan": ["japanese"],
    "loans": ["loan"],
    "lending": ["lenders"],
    "mortgage": ["mortgages"],
    "profits": ["profit"],
    "savings": ["saving"],
    "spread": ["spreads"],
    "wages": ["wage"],
    "yield": ["yields"],
    ("auto", "sales"): [("car", "sales"), ("vehicle", "sales")],
    ("business", "activity"): [("business", "activity"), ("real", "activity"), ("business", "conditions"), ("economic", "conditions")],
    ("capital", "expenditures"): [("capital", "spending")],
    ("commodity", "prices"): [("commodity", "price")],
    ("consumer", "confidence"): [("consumer", "sentiment")],
    ("durable", "goods"): ["durables", ("durable", "equipment")],
    ("energy", "prices"): [("energy", "price")],
    ("equity", "prices"): [("equity", "price"), ("stock", "prices"), ("stock", "price")],
    ("exchange", "rate"): [("exchange", "rates")],
    ("food", "prices"): [("food", "price")],
    ("gas", "prices"): [("gas", "price")],
    ("gasoline", "prices"): [("gasoline", "price")],
    ("home", "prices"): [("home", "price"), ("house", "prices"), ("house", "price")],
    ("hourly", "compensation"): [("hourly", "earnings")],
    ("import", "prices"): [("import", "price")],
    ("input", "prices"): [("input", "price")],
    ("labor", "market"): [("labor", "markets")],
    ("manufacturing", "firms"): ["manufacturers", "manufacturer",("manufacturing", "sector")],
    ("mortgage", "interest"): [("mortgage", "rate"), ("mortgage", "rates")],
    ("oil", "prices"): [("oil", "price")],
    ("potential", "output"): [("potential", "gdp")],
    ("producer", "prices"): [("producer", "price")],
    ("retail", "prices"): [("retail", "price")],
    ("share", "prices"): [("share", "price")],
    ("treasury", "securities"): ["treasuries", ("treasury", "bills"), ("treasury", "security")],
    ("treasury", "yield"): [("treasury", "yields")],
    ("vacancy", "rates"): [("vacancy", "rate")],
    ("wholesale", "prices"): [("wholesale", "price")],
    ("real", "estate", "market"): [("real", "estate", "markets")],
    ("real", "interest", "rate"): [("real", "interest", "rates")],
    ("unit", "labor", "cost"): [("unit", "labor", "costs")],
    "gdp": [("gross", "domestic", "product"), "gnp", ("gross", "national", "product")],
    ("nominal", "gdp"): [("nominal", "gnp")],
    "services": ["service"],
    ("depository", "institutions"): [("thrift", "institutions")],
    "assets": ["asset"],
    ("credit", "standards"): [("lending", "standards")],
    "computers": ["computer"],
    "buildings": ["building", "builders"],
    ("consumer", "prices"): [("consumer", "price")],
    "trucks": ["truck"],
    "semiconductors": ["semiconductor"],
    "farm": ["farmers"],
    "automotive": ["autos", "cars", "automobile"],
    "metals": ["metal"],
    ("oil", "imports"): [("imported", "oil")],
    "soybean": ["soybeans"],
    ("raw", "materials"): [("crude", "materials")],
    ("holiday", "season"): ["christmas"]
}

exclude_dict = {
    "banks": [("district", "banks")],
    "credit": [("credit", "standards"), ("credit", "quality")],
    "employment": [("employment", "cost")],
    "euro": [("euro", "area")],
    "exports": [("net", "exports")],
    "housing": [("housing", "starts"), ("housing", "activity")],
    "imports": [("import", "prices")],
    "inflation": [("inflation", "expectations"), ("inflation", "compensation"), ("core", "inflation"), ("headline", "inflation")],
    "investment": [("residential", "investment"), ("foreign", "direct", "investment")],
    "loans": [("loan", "demand"), ("loan", "officer"), ("loan", "rates")],
    "mortgage": [("mortgage", "interest"), ("mortgage", "rates"), ("mortgage", "rate")],
    "output": [("output", "gap"), ("potential", "output"), ("output", "per", "hour")],
    "unemployment": [("unemployment", "insurance")],
    "wages": [("wage", "pressures")],
    "yield": [("yield", "curve"), ("treasury", "yield")],
    "income": [("national", "income"), ("income", "tax")],
    ("treasury", "yield"): [("yield", "curve")],
    ("advanced", "foreign", "economies"): [("foreign", "economies")],
    "gdp": [("nominal", "gdp"), ("potential", "gdp"), ("nominal", "gnp"), ("foreign", "gdp")],
    "assets": [("asset", "purchases")],
    ("crude", "oil"): [("oil", "price"), ("oil", "prices")],
    ("money", "market"): [("money", "market", "mutual")],
    ("natural", "gas"): [("gas", "price"), ("gas", "prices")],
    "commodities": [("commodity", "prices"), ("commodity", "price")],
    "firms": [("manufacturing", "firms")]
}

# Initialize an empty list to store sentiment data
adjusted_sentiment_data = []

# Iterate through target n-grams and their inclusion/exclusion lists
for target_ngram, include_list in include_dict.items():
    exclude_list = exclude_dict.get(target_ngram, []) # Get corresponding exclude list
    
    # Ensure ngrams are strings
    if isinstance(target_ngram, tuple):
        target_ngram = " ".join(target_ngram)

    # Convert ngram_sentiment_df["ngram"] values to strings for comparison
    ngram_values = ngram_sentiment_df["ngram"].astype(str).str.strip().str.lower().values
    target_ngram = target_ngram.strip().lower()
    
    # Filter data for this target_ngram
    filtered_df = ngram_sentiment_df[ngram_sentiment_df["ngram"] == target_ngram]
    
    for date in filtered_df["date"].unique():  # Iterate over available dates
        target_sentiment = filtered_df.loc[
            filtered_df["date"] == date, "total_sentiment"
        ].values[0]
    
        total_words = filtered_df.loc[
            filtered_df["date"] == date, "total_words"
        ].values[0]
         
        # Compute sentiment adjustments based on included/excluded n-grams. included_sentiment and excluded_sentiment are computed using .sum() on filtered values, making the process more direct.
        included_sentiment = ngram_sentiment_df.loc[
            (ngram_sentiment_df["date"] == date) & 
            (ngram_sentiment_df["ngram"].isin(include_list)), 
            "base_sentiment"
        ].sum()

        # Aggregate excluded sentiment
        excluded_sentiment = ngram_sentiment_df.loc[
            (ngram_sentiment_df["date"] == date) & 
            (ngram_sentiment_df["ngram"].isin(exclude_list)), 
            "base_sentiment"
        ].sum()
            
        # Compute adjusted sentiment. Sum them up in place.
        adjusted_sentiment = target_sentiment + included_sentiment - excluded_sentiment
                
        # Append the computed values to the list
        adjusted_sentiment_data.append({
            "date": date,
            "ngram": target_ngram,
            "total_sentiment": adjusted_sentiment,  # Adjusted before scaling
            "total_words": total_words  # Used for scaling
        })

# Convert the list to a DataFrame
adjusted_sentiment_df = pd.DataFrame(adjusted_sentiment_data)

# Ensure adjusted_values is a dictionary of (date, ngram) → adjusted_sentiment
adjusted_values = adjusted_sentiment_df.set_index(["date", "ngram"])["total_sentiment"].to_dict()

# Map the adjusted sentiment values while keeping the original DataFrame structure
ngram_sentiment_df["adjusted_sentiment"] = ngram_sentiment_df.apply(
    lambda row: adjusted_values.get((row["date"], row["ngram"]), row["total_sentiment"]), axis=1
)

# Scaling and Standardization
ngram_sentiment_df["scaled_sentiment"] = (
    ngram_sentiment_df["adjusted_sentiment"] / ngram_sentiment_df["total_words"]
)

ngram_sentiment_df["standardized_sentiment"] = ngram_sentiment_df.groupby("ngram")["scaled_sentiment"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Create pivoted dataframe and export to Excel with multiple sheets
pivot_df = ngram_sentiment_df.pivot(index="date", columns="ngram", values="standardized_sentiment")

# Define two n-grams to plot
ngrams_to_plot = ["credit", "oil prices"]

# Define recession periods (adjust as needed)
recession_periods = [("2008-01", "2009-06"), ("2020-02", "2020-04")]

# Create subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# Loop through each n-gram and plot
for i, ngram in enumerate(ngrams_to_plot):
    if ngram in pivot_df.columns:
        axes[i].plot(pivot_df.index, pivot_df[ngram], linestyle='-')  # No markers
        axes[i].set_title(f"Sentiment for '{ngram}'")
        axes[i].set_xlabel("Date")
        axes[i].set_ylabel("Standardized Sentiment")
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True)

        # Add horizontal line at sentiment = 0
        axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1)

        # Add shaded regions for recession periods
        for start, end in recession_periods:
            axes[i].axvspan(start, end, color='red', alpha=0.3)

    else:
        axes[i].text(0.5, 0.5, f"'{ngram}' not found", ha='center', va='center', fontsize=12)
        axes[i].set_title(f"'{ngram}' Not Found")
        axes[i].axis("off")

plt.tight_layout()
plt.show()

with pd.ExcelWriter("output.xlsx") as writer:
    pivot_df.to_excel(writer, sheet_name="Standardized Sentiment", index=True)
