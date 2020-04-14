from itertools import combinations
from scipy.stats import norm
import pandas as pd

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
    joint_cat = cat1 + ' / ' + cat2
    return series.apply(lambda x: joint_cat if x in (cat1, cat2) else x)

def merge_two_intervals(series, cat1, cat2):
    pair = cat1, cat2
    lower_bound = cat1.left if cat1.left < cat2.left else cat2.left
    upper_bound = cat2.right if cat2.right > cat1.right else cat1.right
    return series.apply(lambda x: pd.Interval(lower_bound, upper_bound) if x in pair else x)

# def get_categories(series):
#     return sorted(list(series.value_counts(sort=False).index))
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