# Working Python replication of IDENTIFYING MONETARY POLICY SHOCKS:A NATURAL LANGUAGE APPROACH by S. Borağan Aruoba and Thomas Drechsel.
############################ Import block ############################
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

############################ Global constants ############################

# old_directory = "C:\Users\lenze"
# Specify the path of the directory you want to change to. This is where the FED documents are located.
new_directory = r"D:\FOMC"
# Change the current working directory
os.chdir(new_directory)
# Verify the change
print("Current working directory:", os.getcwd())

# Program code to read all pdfs in D:\FOMC
# Set your folder path containing the Federal Reserve PDFs. Unzip the zip file from my Google drive.
folder_path = r"D:\FOMC"

# For faster debugging/trial and error, set folder_path to a folder with a small sample of PDFs. 
#folder_path = r"D:\FOMC\test"

# Create string to match YYYY_MM_DD format of the PDFs and an empty dictionary for the dates and text.
date_pattern = r"\d{4}_\d{2}_\d{2}"  
grouped_data = {}

# Lists of singles, doubles, and triples (unigrams, bigrams, and trigrams known generally as ngrams). The lists of tuples are economic concepts chosen by Drechsel and Aruoba. I also include ngrams that will have sentiments included and excluded.
singles = [('borrowing',), ('brazil',), ('banks',), ('canada',), ('credit',), ('china',), ('consumption',), ('construction',), ('currencies',), ('deposits',), ('employment',), ('equipment',), ('euro',), ('exports',), ('germany',), ('hiring',), ('hours',), ('housing',), ('imports',), ('inflation',), ('inventories',), ('investment',), ('japan',), ('liquidity',), ('loans',), ('leasing',), ('lending',), ('machinery',), ('mexico',), ('mortgage',), ('output',), ('productivity',), ('profits',), ('recovery',), ('reserves',), ('savings',), ('spread',), ('structures',), ('tourism',), ('unemployment',), ('utilization',), ('wages',), ('weather',), ('yield',), ('income',), ('gdp',), ('cpi',), ('nairu',), ('services',), ('bonds',), ('economy',), ('outlays',), ('financing',), ('assets',), ('finance',), ('shipments',), ('capacity',), ('office',), ('computers',), ('industries',), ('producers',), ('supply',), ('homes',), ('sectors',), ('agriculture',), ('merchandise',), ('investors',), ('aircraft',), ('stocks',), ('buildings',), ('cash',), ('trucks',), ('semiconductors',), ('farm',), ('uncertainty',), ('households',), ('crop',), ('apparel',), ('steel',), ('automotive',), ('metals',), ('permits',), ('commerce',), ('transportation',), ('municipal',), ('commodities',), ('corporations',), ('liabilities',), ('consumers',), ('firms',), ('trading',), ('corn',), ('asia',), ('taxes',), ('software',), ('mining',), ('losses',), ('jobs',), ('cars',), ('depreciation',), ('recession',), ('france',), ('korea',), ('italy',), ('lumber',), ('volatility',), ('wheat',), ('livestock',), ('rents',), ('petroleum',), ('traffic',), ('fuel',), ('plants',), ('technology',), ('argentina',), ('cattle',), ('crisis',), ('utilities',), ('travel',), ('payrolls',), ('factory',), ('transfers',), ('drought',), ('gold',), ('salaries',), ('cotton',), ('coal',), ('philippines',), ('singapore',), ('taiwan',), ('thailand',), ('soybean',), ('swaps',), ('harvest',), ('environment',), ('deflator',), ('delinquencies',), ('chemicals',), ('mergers',), ('rigs',), ('indonesia',), ('political',), ('peso',), ('retirement',), ('tobacco',), ('hurricane',), ('equities',), ('russia',), ('workers',), ('contractors',), ('borrowers',), ('brazilian',), ('bank',), ('banking',), ('bankers',), ('canadian',), ('chinese',), ('export',), ('german',), ('hires',), ('houses',), ('import',), ('inventory',), ('investments',), ('japanese',), ('loan',), ('lenders',), ('mortgages',), ('profit',), ('saving',), ('spreads',), ('wage',), ('yields',), ('durables',), ('manufacturers',), ('manufacturer',), ('treasuries',), ('gnp',), ('service',), ('asset',), ('computer',), ('building',), ('builders',), ('truck',), ('semiconductor',), ('farmers',), ('autos',), ('automobile',), ('metal',), ('soybeans',), ('christmas',)
 ]

doubles = [('employment', 'cost'), ('aggregate, demand'), ('auto', 'sales'), ('bond', 'issuance'), ('budget', 'deficit'), ('business', 'activity'), ('business', 'confidence'), ('business', 'spending'), ('capital', 'expenditures'), ('consumer', 'confidence'), ('current', 'account'), ('debt', 'growth'), ('defense', 'spending'), ('delinquency', 'rates'), ('developing', 'countries'), ('domestic', 'demand'), ('drilling', 'activity'), ('durable', 'goods'), ('economic', 'growth'), ('energy', 'prices'), ('equity', 'issuance'), ('equity', 'prices'), ('euro', 'area'), ('exchange', 'rate'), ('federal', 'debt'), ('financial', 'conditions'), ('financial', 'developments'), ('fiscal', 'policy'), ('fiscal', 'stimulus'), ('food', 'prices'), ('foreign', 'economies'), ('gas', 'prices'), ('gasoline', 'prices'), ('government', 'purchases'), ('home', 'prices'), ('home', 'sales'), ('hourly', 'compensation'), ('household', 'debt'), ('household', 'spending'), ('import', 'prices'), ('industrial', 'production'), ('industrial', 'supplies'), ('inflation', 'compensation'), ('inflation', 'expectations'), ('initial', 'claims'), ('input', 'prices'), ('intermediate', 'materials'), ('international', 'developments'), ('labor', 'market'), ('manufacturing', 'activity'), ('manufacturing', 'firms'), ('monetary', 'aggregates'), ('mortgage', 'interest'), ('natural', 'rate'), ('net', 'exports'), ('new', 'orders'), ('nondefense', 'capital'), ('oil', 'prices'), ('output', 'gap'), ('potential', 'output'), ('price', 'pressures'), ('producer', 'prices'), ('refinancing', 'activity'), ('residential', 'investment'), ('retail', 'prices'), ('retail', 'sales'), ('retail', 'trade'), ('share', 'prices'), ('social', 'security'), ('stock', 'market'), ('trade', 'balance'), ('trade', 'deficit'), ('trade', 'surplus'), ('treasury', 'securities'), ('treasury', 'yield'), ('vacancy', 'rates'), ('wholesale', 'prices'), ('wholesale', 'trade'), ('yield', 'curve'), ('foreign', 'exchange'), ('nominal', 'gdp'), ('core', 'inflation'), ('motor', 'vehicles'), ('financial', 'institutions'), ('depository', 'institutions'), ('credit', 'standards'), ('consumer', 'prices'), ('crude', 'oil'), ('loan', 'demand'), ('united', 'kingdom'), ('money', 'market'), ('market', 'participants'), ('commercial', 'paper'), ('housing', 'starts'), ('housing', 'activity'), ('natural', 'gas'), ('consumer', 'goods'), ('balance', 'sheet'), ('financial', 'markets'), ('economic', 'indicators'), ('final', 'sales'), ('credit', 'quality'), ('international', 'transactions'), ('finished', 'goods'), ('latin', 'america'), ('economic', 'outlook'), ('domestic', 'developments'), ('oil', 'imports'), ('home', 'equity'), ('headline', 'inflation'), ('raw', 'materials'), ('holiday', 'season'), ('inflationary', 'pressures'), ('loan', 'officer'), ('health', 'care'), ('economic', 'expansion'), ('economic', 'data'), ('canadian', 'dollar'), ('corporate', 'profits'), ('insurance', 'companies'), ('wage', 'pressures'), ('market', 'expectations'), ('consumer', 'spending'), ('car', 'sales'), ('vehicle', 'sales'), ('real', 'activity'), ('business', 'conditions',), ('economic', 'conditions'), ('capital', 'spending'), ('consumer', 'sentiment'), ('durable', 'equipment'), ('energy', 'price'), ('equity', 'price'), ('stock', 'prices'), ('stock', 'price'), ('exchange', 'rates'), ('food', 'price'), ('gas', 'price'), ('gasoline', 'price'), ('home', 'price'), ('house', 'prices'), ('house', 'price'), ('hourly', 'earnings'), ('import', 'price'), ('input', 'price'), ('labor', 'markets'), ('manufacturing', 'sector'), ('mortgage', 'rates'), ('oil', 'price'), ('potential', 'gdp'), ('producer', 'price'), ('retail', 'price'), ('share', 'price'), ('treasury', 'bills'), ('treasury', 'security'), ('treasury', 'yields'), ('vacancy', 'rate'), ('wholesale', 'price'), ('nominal', 'gnp'), ('thrift', 'institutions'), ('lending', 'standards'), ('consumer', 'price'), ('imported', 'oil'), ('crude', 'materials'), ('district', 'banks'), ('import', 'prices'), ('inflation', 'expectations'), ('inflation', 'compensation'), ('core', 'inflation'), ('headline', 'inflation'), ('loan', 'rates'), ('mortgage', 'rate'), ('unemployment', 'insurance'), ('national', 'income'), ('income', 'tax',), ('foreign', 'gdp'), ('asset', 'purchases'), ('oil', 'price'), ('oil', 'prices'), ('commodity', 'prices'), ('commodity', 'price')

]

triples = [('advanced', 'foreign', 'economies'), ('commercial', 'real', 'estate'), ('compensation', 'per', 'hour'), ('domestic', 'final', 'purchases'), ('domestic', 'financial', 'developments'), ('emerging', 'market', 'economies'), ('foreign', 'industrial', 'countries'), ('gross', 'domestic', 'purchases'), ('household', 'net', 'worth'), ('international', 'financial', 'transactions'), ('labor', 'force', 'participation'), ('major', 'industrial', 'countries'), ('market', 'interest', 'rates'), ('nondefense', 'capital', 'goods'), ('output', 'per', 'hour'), ('real', 'estate', 'activity'), ('real', 'estate', 'market'), ('real', 'interest', 'rate'), ('real', 'interest', 'rates'), ('residential', 'real', 'estate'), ('unit', 'labor', 'cost'), ('unit', 'labor', 'costs'), ('money', 'market', 'mutual'), ('foreign', 'net', 'purchases'), ('real', 'estate', 'markets'), ('gross', 'domestic', 'product'), ('gross', 'national', 'product'), ('foreign', 'direct', 'investment'), ('money', 'market', 'mutual')
]

# Define n-grams to include and exclude as dictionaries. Drechsel and Aruoba combine the sentiments of some economic concepts with the sentiments of select singles, doubles, and triples (ngrams). I set the economic concepts as values associated with main ngrams. Later, I add and/or subtract the sentiments of the values in include_dict and exclude_dict to form final_adjusted_sentiment. 
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

############################ Functions ############################

# Define function load_lm_master_dictionary to load the Loughran McDonald dictionary. Function returns a dictionary, lm_lexicon, containing all dictionary words and their positive or negative sentiment. filepath is the location of the .csv file in your working directory.
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

# Define function find_ngram_sentiments_master to associate words and sentiments from lm_lexicon to user-defined text (cleaned text from Fed documents organized by date). ngrams_with_n will be a list of tuples. The matched ngrams with positive sentiment receive a score of 1 and negative sentiment a score of -1. Context words also receive sentiment scores and sum up to adjusted_sentiment which is associated with an ngram and date. The function returns data frame full_df. 
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

                    # Compute sentiment using direct lookup from lm_lexicon
                    sentiment_score = (
                        1 if lm_lexicon[" ".join(ngram)]["Positive"] > 0 else
                        -1 if lm_lexicon[" ".join(ngram)]["Negative"] > 0 else
                        0
                    )

                    # Compute sentiment before n-gram
                    before_sentiment = sum(
                        1 if lm_lexicon[word]["Positive"] > 0 else -1 if lm_lexicon[word]["Negative"] > 0 else 0
                        for word in words[start:i]
                    )

                    # Compute sentiment after n-gram
                    after_sentiment = sum(
                        1 if lm_lexicon[word]["Positive"] > 0 else -1 if lm_lexicon[word]["Negative"] > 0 else 0
                        for word in words[i + n_value:end]
                    )

                    # Compute adjusted sentiment
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

# Create function to combine included and excluded sentiments to the main sentiment (base_sentiment). This function returns a series with final_adjusted_sentiment and total_words organized by date and ngram. I apply the function and update a dataframe, df_all. 
def apply_inclusion_exclusion(row):
    # Retrieve adjusted sentiment & total_words safely
    ngram_info = adjusted_values.get((row["date"], row["ngram"]), {"adjusted_sentiment": 0, "total_words": 1})
    
    base_sentiment = ngram_info["adjusted_sentiment"]
    total_words = ngram_info["total_words"]
    
    # Compute included/excluded sentiment
    included_sentiment = sum(adjusted_values.get((row["date"], inc), {"adjusted_sentiment": 0})["adjusted_sentiment"] 
                             for inc in include_dict.get(row["ngram"], []))
    
    excluded_sentiment = sum(adjusted_values.get((row["date"], exc), {"adjusted_sentiment": 0})["adjusted_sentiment"] 
                             for exc in exclude_dict.get(row["ngram"], []))

    # Compute final adjusted sentiment
    final_sentiment = base_sentiment + included_sentiment - excluded_sentiment

    return pd.Series({"final_adjusted_sentiment": final_sentiment, "total_words": total_words})

############################ Main execution block ############################

# Get a list of all PDFs in the folder
pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

# Loop through each PDF file and read it. Match the date_pattern with names of PDF files and group by date for file_date. Convert file_date to a datetime variable, file_datetime. Read in pdf files to full_text and store text as values in dictionary, grouped_data, with file_datetime as keys. Use defaultdict to automatically handle the grouping.
grouped_data = defaultdict(str)

for pdf_file in pdf_files:
    file_date = re.search(date_pattern, os.path.basename(pdf_file)).group()
    file_datetime = datetime.strptime(file_date, "%Y_%m_%d").date()
    
    pdf_path = os.path.join(folder_path, pdf_file)
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        full_text = "".join(page.extract_text() for page in reader.pages)

    grouped_data[file_datetime] += full_text + "\n"

# Merging PDFs with the same date. After grouping, the text for each date is merged. I used .split() to break the text into words and then .join() to concatenate them back, which also handles removing excess whitespace between words.
merged_text_by_date = {date: " ".join(text.split()) for date, text in grouped_data.items()}

# Loading the Master Dictionary from Loughran-McDonald
lm_filepath = "Loughran-McDonald_MasterDictionary_1993-2023.csv"  # Update with actual path
lm_lexicon = load_lm_master_dictionary(lm_filepath)

# Combine all ngrams into a list of tuples (ngram, n). all_ngrams is used in the find_ngram_sentiments_master function to calculate all sentiment scores at once rather than for each list of tuples (singles, doubles, and triples).
all_ngrams = [(ngram, 1) for ngram in singles] + \
             [(ngram, 2) for ngram in doubles] + \
             [(ngram, 3) for ngram in triples]
        
# Perform sentiment analysis by date. merged_text_by_date is a dictionary with dates as keys and text as values. It was formed previously by grouped_data which is a similar dictionary formed from the PDFs read in with PDFreader.
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
    
# Create dataframe of sentiments from the cleaned text (merged_text_by_date), ngram lists (all_ngrams), and Loughran-McDonald dictionary (lm_lexicon).
df_all = find_ngram_sentiments_master(merged_text_by_date, all_ngrams, lm_lexicon)

# Aggregate by date and ngram. There may be many instances of the same ngram for each date (and within each document). Therefore, adjusted_sentiment is now the sum of adjusted_sentiment by date and ngram.
df_all = df_all.groupby(["date", "ngram"]).agg(
    adjusted_sentiment=("adjusted_sentiment", "sum"),
    total_words=("total_words", "first"),  # Use "first" since total_words is constant per date
).reset_index()

# Make a full grid of dates and ngrams since some ngrams in singles, doubles, and triples may not be found for every date (and in every document) and we still need sentiment scores of 0 for those ngrams. This helps later for standardization by date for each ngram.

# Get all unique dates and ngrams
all_dates = df_all["date"].unique()
all_ngrams = df_all["ngram"].unique()

# Create full DataFrame with all (date, ngram) combinations
full_index = pd.MultiIndex.from_product([all_dates, all_ngrams], names=["date", "ngram"])
full_df = pd.DataFrame(index=full_index).reset_index()

# Merge the full grid with ngram_sentiment_df to ensure all pairs exist
df_all = full_df.merge(df_all, on=["date", "ngram"], how="left")

# Create dictionary mapping (date, ngram) → (adjusted_sentiment, total_words)
adjusted_values = df_all.set_index(["date", "ngram"])[["adjusted_sentiment", "total_words"]].to_dict(orient="index")

# Apply the function and update full_df
df_all[["final_adjusted_sentiment", "total_words"]] = df_all.apply(apply_inclusion_exclusion, axis=1)

# If testing a small number of documents, some words aren't found (missing/NaN), but still receive "0" sentiment scores. In that case, when finding scaled and standardized sentiment, scaled_sentiment = total_sentiment / total_words = 0 / NaN = NaN. If the count of total_words is 0, then we fill total_words with 1 instead of 0. This is helpful to allow program completion when testing changes to code.

# Fill missing sentiment scores with 0 BEFORE standardization and missing word totals with 1.
df_all["final_adjusted_sentiment"] = df_all["final_adjusted_sentiment"].fillna(0)
df_all["total_words"] = df_all["total_words"].fillna(1)

# Scale and standardize. Divide final_adjusted_sentiment by total_words and call it scaled_sentiment. Then standardize it.
df_all["scaled_sentiment"] = df_all["final_adjusted_sentiment"] / df_all["total_words"]

df_all["standardized_sentiment"] = df_all.groupby("ngram")["scaled_sentiment"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Create a pivoted dataframe and export to Excel in one sheet. I want ngrams as columns, dates as row numbers, and standardized sentiments as values in the center. This matches the format in Drechsel and Aruoba's Excel file in the sheet: Sentiments by meeting.
pivot_df = df_all.pivot(index="date", columns="ngram", values="standardized_sentiment")

# Excel export with pivot_df
with pd.ExcelWriter("output.xlsx") as writer:
    pivot_df.to_excel(writer, sheet_name="Standardized Sentiment", index=True)

############################ Plot section ############################

# Plot two ngrams' sentiment over time in separate plots.
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
