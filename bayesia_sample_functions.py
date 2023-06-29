'''
Functions to clean speed dating data and train Bayesia.

@author: Julia Biswas
'''

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def clean_data(data):
    '''
    Takes in speed dating data to clean and returns cleaned data.
    
    data: the speed dating data
    
    Returns cleaned data.
    '''
    
    data = drop_unneeded_columns(data)
    data = fix_pref_scores(data)
    
    data = analyze_statistical_significance_of_race(data)
    data = analyze_statistical_significance_of_hometown(data)
    data = analyze_statistical_significance_of_having_met_prior(data)
    
    traits = ['attr', 'sinc', 'intel', 'fun', 'amb']
    interests = ['go_out', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 
                 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 
                 'concerts', 'music', 'shopping', 'yoga']
    
    data = determine_personalities(data, traits)
    
    data = data.rename(columns={'iid':'p1', 'pid':'p2', 'pf_o_attr':'p2_pref_attr',
                         'pf_o_sinc':'p2_pref_sinc', 'pf_o_intel':'p2_pref_intel', 
                         'pf_o_fun':'p2_pref_fun', 'pf_o_amb':'p2_pref_amb', 
                         'pf_o_shar':'p2_pref_shar', 'field_cd':'p1_acad_interest',
                         'pf_o_att':'p2_pref_att', 'attr1_1':'p1_pref_attr',
                         'sinc1_1':'p1_pref_sinc','intel1_1':'p1_pref_intel',
                         'fun1_1':'p1_pref_fun', 'amb1_1':'p1_pref_amb',
                         'shar1_1':'p1_pref_shar', 'attr3_1':'p1_attr',
                         'sinc3_1':'p1_sinc','fun3_1':'p1_fun',
                         'intel3_1':'p1_intel', 'amb3_1':'p1_amb', 
                         'dec_o':'p2_dec', 'dec':'p1_dec'})

    data = rescale(data, traits, interests)
    data = combine_rows(data, traits, interests)
    data = label_data(data)
    
    return data

def train(train_data):
    '''
    Trains the Naive Bayes Classifier using the given training data.
    
    train_data: the given training data
    
    Returns the prior probabilities of each label and the probability of each value of 
    each feature given each label.
    '''
    
    cols = train_data.columns
    num_labels = max(train_data['label'].unique())+1
    num_features = len(cols)-1

    p_Y = np.zeros([num_labels]) #P(label)
    P_Xi_Y = [[dict() for j in range(num_features)] for i in range(num_labels)] #P(value | label)

    for i in range(num_labels):
        label_occurences = len(train_data[train_data['label'] == i])
        p_Y[i] = label_occurences/len(train_data)
        for j in range(num_features): 
            feature_vals = np.unique(train_data[cols[j]].to_numpy())
            p = {} #P_Xi_Y values for each value of feature j for label i
            for k in range(len(feature_vals)):
                #using a Laplace Maximum A Posteriori estimate
                occurences = len(train_data[(train_data[cols[j]] == feature_vals[k]) & 
                                            (train_data['label'] == i)])
                p[feature_vals[k]] = (occurences+1)/(label_occurences+2)
            P_Xi_Y[i][j] = p
            
    return p_Y, P_Xi_Y

'''Helper Functions'''

def drop_unneeded_columns(data):
    '''
    Drops data provided with the speed dating data that is not relevant to the model.
    
    data: the speed dating data
    
    Returns the speed dating data without the unneeded columns. 
    '''
    
    data = data.drop(columns=['id', 'idg', 'condtn', 'round', 'position', 'positin1', 
                              'order', 'partner', 'age_o', 'race_o', 'prob_o',
                             'age', 'field', 'undergra', 'mn_sat', 'tuition', 'race',
                             'imprelig', 'goal', 'career', 'career_c', 'you_call',
                             'them_cal', 'date_3', 'numdat_3', 'num_in_3', 'exphappy',
                             'expnum', 'prob', 'match_es', 'length', 'numdat_2', 
                             'gender', 'date', 'like_o', 'like', 'int_corr'])
    
    for col in data.columns:
        if '4_' in col:
            data = data.drop(columns=[col])
        elif '2_' in col:
            data = data.drop(columns=[col])
        elif '5_' in col:
            data = data.drop(columns=[col])
        elif 's_2' in col:
            data = data.drop(columns=[col])
        elif '3_s' in col or '1_s' in col:
            data = data.drop(columns=[col])
        elif '_2' in col:
            data = data.drop(columns=[col])
        elif '_3' in col:
            data = data.drop(columns=[col])
    
    return data

def fix_pref_scores(data):
    '''
    Standardizes participants' scoring of their preferences (waves 6 through 9 of 
    the speed dates scored their preferences differently).
    
    data: the speed dating data
    
    Returns the speed dating data with the standardized columns and without the wave 
    column (since it's no longer needed).
    '''
    one = []
    two = []
    for col in data.columns:
        if '1_1' in col:
            one.append(col)
        elif '1_2' in col:
            two.append(col)

    for i, row in data.iterrows():
        sum_one = 0
        for col in one:
            sum_one += data.loc[i, col]
        for col in one:
            data.loc[i, col] = (data.loc[i, col]/sum_one) * 100

        sum_two = 0
        for col in two:
            sum_two += data.loc[i, col]
        for col in two:
            data.loc[i, col] = (data.loc[i, col]/sum_two) * 100

    data = data.drop(columns=['wave'])
    
    return data
    
def analyze_statistical_significance_of_race(data):
    '''
    Analyzes the statistical significance of race. While the model should not learn racial 
    bias nor would two people using the model do so if the other's race would be an issue, 
    but it did need to be ensured that racial bias didn't impact the data. It was found 
    that the relationship was not statistically significant, so the race data could be 
    discarded.
    
    data: the speed dating data
    
    Returns the speed dating data without data on race.
    '''

    race_data = data.dropna(subset=['samerace'])
    race_data = data.dropna(subset=['imprace'])
    same_race = race_data[race_data['samerace'] == 1]
    diff_race = race_data[race_data['samerace'] == 0]
    
    sample_diff = abs(np.mean(same_race['match']) - 
                      np.mean(diff_race['match']))
    if not check_statistical_significance(race_data, sample_diff, len(same_race)): 
        data = data.drop(columns=['samerace', 'imprace'])
        
    return data
        
def analyze_statistical_significance_of_hometown(data):
    '''
    Analyzes the statistical significance of hometown median income. The model should not 
    learn socioeconomic bias, but it was important that this didn't affect the training 
    data. The dataset was split between those whose hometown's household median income was 
    above the 2004 median household income—44,334 dollars according to the U.S. Census 
    Bureau—and those whose was below an income above and those with an income below the 
    2004. It was found that the relationship was not statistically significant, so the 
    hometown data could be discarded.
    
    data: the speed dating data
    
    Returns the speed dating data without data on hometowns.
    '''

    income_data = data.dropna(subset = ['income'])
    median_income = 44334

    #find average income
    for i, row in income_data.iterrows():
        income_data.loc[i, 'income'] = float(income_data.loc[i, 'income'].replace(',', ''))

    #split between above and below average
    above_median = income_data[income_data['income'] > median_income]
    below_median = income_data[income_data['income'] <= median_income]

    sample_diff = abs(np.mean(above_median['match']) - 
                      np.mean(below_median['match']))
    if not check_statistical_significance(income_data, sample_diff, len(above_median)):
        data = data.drop(columns=['income', 'from', 'zipcode'])
        
    return data

def analyze_statistical_significance_of_having_met_prior(data):
    '''
    Analyzes the statistical significance of the two individuals on the date having met 
    prior to the date. It was found that the relationship was statistically significant. 
    Because this would not be a feature involved in the model, this presents a challenge 
    that was ultimately resolved by dropping all rows of data about interactions between 
    individuals who have already met. This was done so that that factor doesn't affect 
    results (i.e. increases compatability score based off of a preexisting relationship). 
    The individuals who use Bayesia evidently have already met, so that isn't necesarily 
    an applicable factor in the analysis. There is extremely limited data on interactions 
    between people who've already met, and it would be beneficial to have data that is 
    uninfluenced by this factor.
    
    data: the speed dating data
    
    Returns the speed dating data without data on people who've already met and data on 
    whether the two have met (since it's now assumed they haven't).
    '''

    met_data = data
    met_data['have_met'] = ""
    for i, row in met_data.iterrows():
        met = row['met']
        met_o = row['met_o']

        #drop data if there is confusion over the two people having met before
        if met != met_o and not math.isnan(met) and not math.isnan(met_o) and (
            met == 1 or met == 2) and (met_o == 1 or met_o == 2):
            met_data = met_data.drop(i)
        elif not (met == 1 or met == 2) and not (met_o == 1 or met_o == 2):
            met_data = met_data.drop(i)
        #if the data is valid
        else:
            met_data.loc[i, 'have_met'] = met

    met_data = met_data.drop(columns=['met', 'met_o'])

    #(1 = have met, 2 = haven't met)
    have_met = met_data[met_data['have_met'] == 1]
    have_met = np.array(have_met['match'])
    havent_met = met_data[met_data['have_met'] == 2]
    havent_met = np.array(havent_met['match'])

    sample_diff = abs(np.mean(havent_met) - np.mean(have_met))
    
    if check_statistical_significance(met_data, sample_diff, len(havent_met)):
        data = data[(data['met'] != 1) & (data['met_o'] != 1) ]
        data = data.drop(columns=['met', 'met_o', 'have_met'])
        
    return data

def check_statistical_significance(data, sample_diff, split):
    '''
    Determines whether the relationship found by the user is statistically significant or 
    not by calculating the p-value of the given data using bootstrapping over 50,000 trials.
    It calculates what % of arbitrary splits of the data has a higher difference in 
    means than the split determined by the user; if the % is less than 5%, the relationship
    is statistically significant.
    
    data: the user's data
    sample_diff: the difference in means between the two groups the user created from the data
    split: how to split the data into the two arbitrary groups (it's the number of datapoints 
    in one of the two groups)
    
    precondition: split <= len(data)
    
    Returns true if the relationship is statistically significant; otherwise, false.
    '''
    trials = 50000

    occurrences = 0

    for i in range(trials):
        matches = np.array(data['match'])
        
        a = np.random.choice(matches, size = split)
        b = np.random.choice(matches, size = len(data) - split)
        diff = abs(np.mean(a) - np.mean(b))

        if diff >= sample_diff:
            occurrences += 1

    if occurrences/trials < 0.05:
        return True
    else:
        return False
    
def determine_personalities(data, traits):
    '''
    Drop people whose perceptions of their own traits were vastly different from other's 
    perceptions of them so there can be a stable definition of each person's personality.
    Because Bayesia will only have access to what each person thinks of themselves and not
    what the other person thinks of them, it was important to check people's perceptions of 
    themselves. This could be done as most perceptions were correct (over 91%). 
    
    data: the speed dating data
    traits: the personality traits being assessed
    
    Returns the data without people whose perceptions of their own traits were incorrect
    and other people's perceptions of the individual. 
    '''

    data = data.rename({'pf_o_att': 'pf_o_attr', 'pf_o_sin':'pf_o_sinc',
                        'pf_o_int':'pf_o_intel','pf_o_sha':'pf_o_shar'}, axis=1)
    
    all_correct = 0
    to_check = data
    for i, row in data.iterrows():
        if not check_perception(row, traits):
            to_check = to_check.drop(i)

    #dropping other people's perception of the individual
    data = data.drop(columns=['attr', 'sinc', 'intel', 'fun', 'amb', 'shar',
                             'attr_o', 'sinc_o','intel_o', 'fun_o', 'amb_o', 'shar_o',])
    
    data = data.reset_index(drop=True)
    
    return data

def check_perception(row, traits): 
    '''
    Checks whether people's perceptions of themselves are accurate since the scores people 
    gave themselves for each of the characteristics (attractiveness, sincereity, 
    intelligence, fun, ambitiousness) will be used as their personality traits. All traits are 
    scored on a scale of 1-10: a perception is deemed accurate if a person scores themself above 5 
    and so does the other person, both people score the person at 5, or both score them under 5.
    
    row: data on how someone perceives themselves and how another person perceives them
    traits: the personality traits being assessed
    
    Returns true if the number of times a person perceived themselves correctly for all 
    traits; otherwise, false.
    '''
    count = 0 #of wrong
    
    for trait in traits:
        by_p1 = trait+'3_1' #what person perceives themselves as
        by_p2 = trait #what the other person perceived
        
        if ((row[by_p1] < 5) and (row[by_p2] > 5)) or ((row[by_p2] > 5) and (row[by_p2] < 5)):
            count+=1
            
    return count == 0

def rescale(data, traits, interests):
    '''Rescale the data so that the model can interpret it more easily.
    
    data: the speed dating data
    traits: the various interests that the data includes
    interests: the various interests that the data includes
    
    Returns the rescaled data.
    '''

    data = data.dropna().reset_index(drop=True)

    data = data.astype({'p1_acad_interest':'int'})

    for i in range(len(data)):
        p1_prefs_max = 0

        for trait in traits:
            if data.loc[i, 'p1_pref_'+trait] > p1_prefs_max:
                p1_prefs_max = data.loc[i, 'p1_pref_'+trait]

        for trait in traits:
            data.loc[i, 'p1_pref_'+trait] = round((data.loc[i, 'p1_pref_'+trait]/p1_prefs_max)*7)
            data.loc[i, 'p1_'+trait] = round((data.loc[i, 'p1_'+trait]/10)*7)

        data.loc[i, 'p1_pref_shar'] = round((data.loc[i, 'p1_pref_shar']/p1_prefs_max)*7)

        for interest in interests:
            data.loc[i, interest] = round((data.loc[i, interest]/10)*7)

        #gaming's max is 14 for some reason, reading's is 13
        data.loc[i, 'gaming'] = round((data.loc[i, 'gaming']/14)*10)
        data.loc[i, 'reading'] = round((data.loc[i, 'reading']/13)*10)
        
    return data

def combine_rows(data, traits, interests):
    '''
    Combines rows in the data since the given speed dating data has two rows per date
    whereas the model needs all the data in one row.
    
    data: the speed dating data
    traits: the various interests that the data includes
    interests: the various interests that the data includes
    
    Returns the data with its rows combined.
    '''
    for i in interests:
        data = data.rename(columns={i:'p1_'+i})

    count = 0
    for i in range(int(len(data)/2)):
        match = data[(data['p1'] == data.loc[i, 'p2']) & 
                         (data['p2'] == data.loc[i, 'p1'])].reset_index()
        if not match.empty:
            data.loc[i, 'p2_acad_interest'] = match.loc[0, 'p1_acad_interest']

            for j in interests:
                data.loc[i, 'p2_'+j] = match.loc[0, 'p1_'+j]

            for t in traits:
                data.loc[i, 'p2_pref_'+t] = match.loc[0, 'p1_pref_'+t]
                data.loc[i, 'p2_'+t] = match.loc[0, 'p1_'+t]

            data.loc[i, 'p2_pref_shar'] = match.loc[0, 'p1_pref_shar']

        #drop the duplicate
        data = data.drop(match['index']).reset_index(drop=True)
        
    #getting rid of rows with missing values
    data = data.dropna().reset_index(drop=True)
    
    return data

def label_data(data):
    '''
    Labels the data by identifying each date as a match, a one-sided attraction (and on whose 
    side), or an unlikely relationship.
    
    data: the speed dating data
    
    Returns the labeled data.
    '''
    data['label'] = ''

    for i, row in data.iterrows():
        p1_chose = (row['p1_dec'] == 1) #if p1 was attracted
        p2_chose = (row['p2_dec'] == 1) #if p2 was attracted
        if p1_chose and p2_chose: #match!
            data.loc[i, 'label'] = 0

        elif p1_chose: #one-sided attraction (only p1 was attracted)
            data.loc[i, 'label'] = 1

        elif p2_chose: #one-sided attraction (only p2 was attracted)
            data.loc[i, 'label'] = 2

        else: #unlikely relationship
            data.loc[i, 'label'] = 3

    #drop the ids/match because they are no longer needed
    data = data.drop(columns=['p1', 'p2', 'match', 'p1_dec', 'p2_dec'])
    data = data.reset_index(drop=True)
    
    return data

