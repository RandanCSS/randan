from itertools import combinations
from scipy.stats import norm
import pandas as pd
import numpy as np

#extend this list
available_colors = [
    'cadetblue1',
    'bisque',
    'pink',
    'aquamarine',
    'olivedrab1',
    'slategray1',
    'darkseagreen1'
]

def binning(series, n_intervals):
    series = pd.qcut(series, n_intervals, duplicates='drop')
    return series

def merge_two_cats(series, cat1, cat2):
    joint_cat = str(cat1) + ' / ' + str(cat2)
    return series.apply(lambda x: joint_cat if x in (cat1, cat2) else x).astype(str)

def merge_two_intervals(series, cat1, cat2):
    #display(series, cat1, cat2)
    pair = cat1, cat2
    lower_bound = cat1.left if cat1.left < cat2.left else cat2.left
    upper_bound = cat2.right if cat2.right > cat1.right else cat1.right
    return series.apply(lambda x: pd.Interval(lower_bound, upper_bound) if x in pair else x)

def get_categories(series):
    return sorted(series.unique())

def get_unordered_combinations(categories, k):
    return list(combinations(categories, k))

def get_ordered_combinations(categories, k):
    categories_combo = [combo for combo in combinations(categories, k)]
    categories_combo = [combo for combo in categories_combo \
                        if abs(categories.index(combo[0])-categories.index(combo[1]))==1]
    return categories_combo

#move to descriptive statistics
def confidence_interval_mean(mean, std, n, sig_level=0.05):
    z_crit = norm.isf(sig_level / 2)
    error = z_crit * (std/(n**(1/2)))
    if pd.isnull(error):
        error = 0
    lower_bound = mean - error
    upper_bound = mean + error
    return pd.Interval(lower_bound, upper_bound, closed='both')

#move to descriptive statistics ?
def confidence_interval_comparison(first_ci, second_ci):
    if first_ci.overlaps(second_ci):
        return 'overlap'
    elif first_ci < second_ci:
        return 'lower'
    elif first_ci > second_ci:
        return 'bigger'

def classification_table(y_true, y_predicted):
    classification = pd.crosstab(y_true, 
                                    y_predicted,
                                    margins=True)
    classification.index.name = 'Observed'
    classification.columns.name = 'Predicted'
    all_categories = list(classification.index)
    empty_categories = [value for value in all_categories if value not in classification.columns]
    for category in empty_categories:
        classification[category] = 0
    classification = classification[all_categories]

    all_categories.remove('All')
    
    n = classification.loc['All', 'All']
    
    for category in all_categories:
        classification.loc[category, 'All'] =  classification.loc[category, category] / classification.loc[category, 'All'] * 100
        classification.loc['All', category] =  classification.loc['All', category] / n * 100
    
    
    classification.loc['All', 'All'] = np.diagonal(classification.loc[all_categories, all_categories]).sum() / n * 100
    classification.index = all_categories + ['Percent predicted']
    classification.index.name = 'Observed'
    classification.columns = all_categories + ['Percent correct']
    classification.columns.name = 'Predicted'
    return classification

def precision_and_recall(classification_table):
    """
    Estimate precision, recall, and F-score for all the categories.
    """

    preds = classification_table.iloc[:-1, :-1]
    results = []
    categories = list(preds.index)
    for current_category in categories:
        idx = [cat for cat in categories if cat!=current_category]
        tp = preds.loc[current_category, current_category]
        fp = preds.loc[idx, current_category].sum()
        fn = preds.loc[current_category, idx].sum()
        if fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall != 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        results.append([precision, recall, f1])
    results = pd.DataFrame(results, 
                            index=categories, 
                            columns = ['Precision', 'Recall', 'F score'])
    results.loc['Mean'] = results.mean()
    return results