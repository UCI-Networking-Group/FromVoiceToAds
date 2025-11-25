import os
import csv
import json
import re
import numpy as np
import scipy
from collections import defaultdict
import matplotlib.pyplot as plt

import pandas as pd
from statsmodels.stats.multitest import multipletests

SAMPLE_NUM = 10000


voice_age = {
    "p294" : 33,
    "p297" : 20,
    "p299" : 25,
    "p300" : 23,
    "p301" : 23,
    "p305" : 19,
    "p306" : 21,
    "p308" : 18,
    "p310" : 21,
    "p311" : 21,
    "p315" : 18,
    "p318" : 32,
    "p329" : 23,
    "p330" : 26,
    "p333" : 19,
    "p334" : 18,
    "p341" : 26,
    "p345" : 22,
    "p360" : 19,
    "p361" : 19,
    "p362" : 29
}

voice_gender = {
    "p294" : "female",
    "p297" : "female",
    "p299" : "female",
    "p300" : "female",
    "p301" : "female",
    "p305" : "female",
    "p306" : "female",
    "p308" : "female",
    "p310" : "female",
    "p311" : "male",
    "p315" : "male",
    "p318" : "female",
    "p329" : "female",
    "p330" : "female",
    "p333" : "female",
    "p334" : "male",
    "p341" : "female",
    "p345" : "male",
    "p360" : "male",
    "p361" : "female",
    "p362" : "female"
}


def apply_multiple_tests_correction(p_values, method='holm'):
    """
    Apply multiple tests correction to get adjusted p-values.
    """
    return multipletests(p_values, method=method)[1]


def gender_table(df, max_age=None):
    counts = pd.DataFrame(index=df.index)
    for gender in ["male", "female"]:
        voices = [v for v in df.columns if voice_gender.get(v) == gender]
        if max_age is not None:
            voices = [v for v in voices if voice_age.get(v, 0) <= max_age]
        counts[f"{gender}_count"] = df[voices].sum(axis=1)
    return counts


def age_table(df, gender="female"):
    counts = pd.DataFrame(index=df.index)
    voices = df.columns
    if gender is not None:
        voices = [v for v in voices if voice_gender.get(v) == gender]
    counts["18-24"] = df[voices].loc[:, [v for v in voices if 18 <= voice_age.get(v, 0) <= 24]].sum(axis=1)
    counts["25+"] = df[voices].loc[:, [v for v in voices if voice_age.get(v, 0) >= 25]].sum(axis=1)
    return counts


def gender_table_with_norm(df, max_age=None, normalize=True):
    counts = pd.DataFrame(index=df.index)

    # group voices by gender
    voices_by_gender = {
        g: [v for v in df.columns if voice_gender.get(v) == g]
        for g in ["male", "female"]
    }
    if max_age is not None:
        voices_by_gender = {
            g: [v for v in vs if voice_age.get(v, 0) <= max_age]
            for g, vs in voices_by_gender.items()
        }

    group_sizes = {g: len(vs) for g, vs in voices_by_gender.items()}

    # raw counts
    for gender, voices in voices_by_gender.items():
        counts[f"{gender}_count"] = df[voices].sum(axis=1)

    # normalized proportions
    if normalize:
        for gender in voices_by_gender:
            if group_sizes[gender] > 0:
                counts[f"{gender}_prop"] = (counts[f"{gender}_count"] / group_sizes[gender]).round(3)

    counts = counts.sort_index()
    return counts


def age_table_with_norm(df, gender="female", normalize=True):
    counts = pd.DataFrame(index=df.index)

    # filter voices by gender if specified
    voices = df.columns
    if gender is not None:
        voices = [v for v in voices if voice_gender.get(v) == gender]

    # split by age group
    age_groups = {
        "18-24": [v for v in voices if 18 <= voice_age.get(v, 0) <= 24],
        "25+": [v for v in voices if voice_age.get(v, 0) >= 25],
    }
    group_sizes = {k: len(vs) for k, vs in age_groups.items()}

    # raw counts
    for label, vs in age_groups.items():
        counts[f"{label}_count"] = df[vs].sum(axis=1)

    # normalized proportions
    if normalize:
        for label in age_groups:
            if group_sizes[label] > 0:
                counts[f"{label}_prop"] = (counts[f"{label}_count"] / group_sizes[label]).round(3)

    counts = counts.sort_index()
    return counts


def dataframe_gender_categories(df, min_age=0, max_age=100):
    categories = df.index
    genders = ['male', 'female']
    data = {g: [] for g in genders}

    for category in categories:
        for g in genders:
            # select puppets of this gender in the age range
            voices = [v for v in df.columns if voice_gender.get(v[:-1], voice_gender.get(v)) == g
                      and min_age <= voice_age.get(v[:-1], voice_age.get(v, 0)) <= max_age]
            # count how many puppets saw this category
            count = df.loc[category, voices].sum()
            data[g].append(count)

    df_gender = pd.DataFrame(data, index=categories)
    df_gender = df_gender.transpose()  # rows: male/female, columns: categories
    return df_gender


def dataframe_age_categories(df, gender="female"):
    categories = df.index
    age_groups = ['18-24', '25+']
    data = {ag: [] for ag in age_groups}

    for category in categories:
        for ag in age_groups:
            # select voices of the given gender
            voices = [v for v in df.columns if voice_gender.get(v[:-1], voice_gender.get(v)) == gender]

            if ag == '18-24':
                voices = [v for v in voices if 18 <= voice_age.get(v[:-1], voice_age.get(v, 0)) <= 24]
            else:  # 25+
                voices = [v for v in voices if voice_age.get(v[:-1], voice_age.get(v, 0)) >= 25]

            # count how many voices saw this category
            count = df.loc[category, voices].sum()
            data[ag].append(count)

    df_age = pd.DataFrame(data, index=categories)
    df_age = df_age.transpose()  # rows: age group, columns: categories
    return df_age


def generate_random_permutation(df, class1, class2):
    tot_class1 = int(df.loc[class1].sum())
    tot_class2 = int(df.loc[class2].sum())
    # Calculate the sum of each column
    column_sums = df.sum(axis=0).astype(int)
    classes = [class1]*tot_class1 + [class2]*tot_class2
    np.random.shuffle(classes)

    # Initialize a new DataFrame with the same shape
    random_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=int)

    # Generate random integers for the first row
    for col in df.columns:
        cat_classes = classes[:column_sums[col]]
        classes = classes[column_sums[col]:]
        random_df.loc[class1, col] = len([c for c in cat_classes if c==class1])
        random_df.loc[class2, col] = len([c for c in cat_classes if c==class2])

    return random_df


def category_stat(df, category):
    genders = df.sum(axis=1)
    new_df = df[category]

    normalized_df = new_df.div(genders, axis=0)
    diff = normalized_df.iloc[0]-normalized_df.iloc[1]
    return diff


def estimate_probability(values, target_value, bins=1000):
    target = None
    # Calculate the histogram
    frequencies, bin_edges = np.histogram(values, bins=bins)

    # Find the bin index for the target value
    bin_index = np.digitize(target_value, bin_edges) - 1
    # this is a fix as digitize() is not compatible to histrogram() at the edges, see https://github.com/numpy/numpy/issues/4217
    if abs(target_value - bin_edges[-1]) < 0.000001:
        bin_index = len(frequencies) -1

    # Calculate the total number of observations
    total_observations = len(values)

    probability = sum(frequencies[:bin_index+1]) / total_observations
    target = 1

    if probability > sum(frequencies[bin_index:]) / total_observations:
        probability = sum(frequencies[bin_index:]) / total_observations
        target = 0

    return (probability, target)


def contingency_table(df, category):
    table = []
    cat0 = df.iloc[0][category]
    non_cat0 = df.iloc[0].sum() - cat0
    cat1 = df.iloc[1][category]
    non_cat1 = df.iloc[1].sum() - cat1
    return [[cat0, non_cat0], [cat1, non_cat1]]


def fisher_pvalue(table):
    p_fisher = scipy.stats.fisher_exact(table, alternative="less").pvalue
    p_fisher = min(p_fisher, scipy.stats.fisher_exact(table, alternative="greater").pvalue)
    return p_fisher


def boschloo_pvalue(table):
    p_boschloo = scipy.stats.boschloo_exact(table, alternative="less").pvalue
    p_boschloo = min(p_boschloo, scipy.stats.boschloo_exact(table, alternative="greater").pvalue)
    return p_boschloo


def permutation_test_plot(values, highlight_value, bins=100, savefile=None, title=None):
    plt.figure()
    ax = plt.subplot(111)

    ax.hist(values, bins=bins, color='blue', alpha=0.7, edgecolor='black', label=f'${title}(R_k)$')

    y_min, y_max = plt.ylim()
    ax.vlines(x=highlight_value, ymin=y_min, ymax=y_max, colors='red', linestyles='dashed', label=f'${title}(O)$')

    ax.set_ylabel('Frequency')
    ax.legend(loc='upper center', ncol=2)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)

    if savefile:
        plt.savefig(savefile)
    else:
        plt.show()


def print_table(categories, groups, header):
    print(f"\\begin{{table}}")
    print(f"\\begin{{tabular}}{{|l|c|c|c|c|c|c|}}")
    #print(f"&  \\multicolumn{{5}}{{c}}{{{header}}} \\\\")
    #print(f"&  & & \\multicolumn{{3}}{{c}}{{Target}} \\\\")
    print(f"\\hline")
    print(f"Category & P-value & P-value (adj.) & Fisher & Fisher (adj.) & Boschloo & Boschloo (adj.) \\\\")
    print(f"\\hline")

    for cat in categories:
        row = categories[cat]
        is_significant = row.get('p-value', 1.0) < 0.05
        is_significant_adj = row.get('p-value-adj', 1.0) < 0.05

        if is_significant or is_significant_adj:
            print("\\bfseries ", end='')

        print(f"{cat}", end='')

        values = [
            #row['sum'],
            #row['mean_group1'],
            #row['mean_group2'],
            round(row['p-value'], 3) if isinstance(row['p-value'], float) else "N.D.",
            round(row['p-value-adj'], 3) if 'p-value-adj' in row else "N.D.",
            round(row['fisher'], 3) if isinstance(row['p-value'], float) else "N.D.",
            round(row['fisher-adj'], 3) if 'p-value-adj' in row else "N.D.",
            round(row['boschloo'], 3) if isinstance(row['p-value'], float) else "N.D.",
            round(row['boschloo-adj'], 3) if 'p-value-adj' in row else "N.D."
        ]

        for val in values:
            print(" & ", end='')
            if is_significant or is_significant_adj:
                print("\\bfseries ", end='')
            print(f"{val}", end='')

        print(f"\\\\")

    print(f"\\hline")
    print(f"\\end{{tabular}}")
    print(f"\\caption{{Statistical test results on {header}.}}")
    print(f"\\label{{tab:permutation_{header}}}")
    print(f"\\end{{table}}")


def table_plot(target_file, df, permutations, group1, group2, label, header):
    print("----------Table Plot for ", target_file, "----------")
    categories = {}
    cats = set(list(df.columns))# + list(df_oracle.columns))

    raw_p_values = []
    raw_p_values_fisher = []
    raw_p_values_boschloo = []
    category_names = []

    for category in sorted(cats):
        if category in df.columns:
            value = category_stat(df, category)
            perm_values = []
            for rdf in permutations:
                perm_values.append(category_stat(rdf, category))
            p,target = estimate_probability(perm_values, value)

            raw_p_values.append(p)

            category_names.append(category)

            table = contingency_table(df, category)
            p_fisher = fisher_pvalue(table)
            p_boschloo = boschloo_pvalue(table)
            raw_p_values_fisher.append(p_fisher)
            raw_p_values_boschloo.append(p_boschloo)
            permutation_test_plot(perm_values, value, savefile=target_file+"_permutation_"+label+"_"+category.lower().replace(":", "_")+".pdf", title=f"S")
        else:
            p_fisher = "N.D."
            p_boschloo = "N.D."
            p = "N.D."

        categories[category] = {"p-value": p, group1: df[category][group1], group2: df[category][group2], "sum": df[category].sum(), "fisher": p_fisher, "boschloo": p_boschloo, "target": target}

    corrected_p_values = apply_multiple_tests_correction(raw_p_values)
    for i, cat in enumerate(category_names):
        categories[cat]['p-value-adj'] = corrected_p_values[i]

    corrected_p_values_fisher = apply_multiple_tests_correction(raw_p_values_fisher)
    for i, cat in enumerate(category_names):
        categories[cat]['fisher-adj'] = corrected_p_values_fisher[i]

    corrected_p_values_boschloo = apply_multiple_tests_correction(raw_p_values_boschloo)
    for i, cat in enumerate(category_names):
        categories[cat]['boschloo-adj'] = corrected_p_values_boschloo[i]

    print_table(categories, [group1, group2], header)


def conduct_tests(df, target_file):
    # Filter out rare categories
    threshold = 5

    df_agg = dataframe_gender_categories(df, max_age=24)
    df_agg = df_agg.loc[:, df_agg.sum() >= threshold]
    permutations = []
    for _ in range(SAMPLE_NUM):
        rdf = generate_random_permutation(df_agg, "male", "female")
        permutations.append(rdf)

    table_plot(target_file, df_agg, permutations, "male", "female", "gender", "(18-24)")

    df_agg = dataframe_age_categories(df, gender="female")
    df_agg = df_agg.loc[:, df_agg.sum() >= threshold]
    permutations = []
    for _ in range(SAMPLE_NUM):
        rdf = generate_random_permutation(df_agg, "18-24", "25+")
        permutations.append(rdf)

    table_plot(target_file, df_agg, permutations, "18-24", "25+", "age", "(female)")


def group_advertisers(df):
    with open("advertisers_mapping.json", "r", encoding="utf-8") as f:
        advertiser_to_category = json.load(f)

    # Step 1: add category column
    df = df.reset_index()
    df["Category"] = df["Item"].map(advertiser_to_category).fillna("Other")

    # Step 2: group by category and sum
    df_grouped = df.drop(columns=["Item"]).groupby("Category").sum()

    # optional: sort categories alphabetically or by total
    df_grouped = df_grouped.sort_index()

    return df_grouped


def analyze_amazon_audiences(data_file):
    # Analyze: individual puppets
    # Rows are categories. For each category, 1 means that puppet saw the category and 0 means not.
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_items = set().union(*data.values())
    rows = []
    for item in all_items:
        row = {"Item": item}
        for puppet in data:
            row[puppet] = int(item in data[puppet])
        row["Total"] = sum(row[g] for g in data)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Item")

    print(df.sort_values("Total", ascending=False))

    # Analyze: gender and age based

    print("=== Gender table (max_age=24) ===")
    print(gender_table_with_norm(df, max_age=24, normalize=True).to_string())

    print("\n=== Age table (gender='female') ===")
    print(age_table_with_norm(df, gender="female", normalize=True).to_string())

    print("--------------------------------------")
    puppets = defaultdict(set)
    for key, cates in data.items():
        puppets[key].update(cates)  # keep key as-is (p294a, p294b)
    # build table
    all_items = set().union(*puppets.values())
    rows = []
    for item in all_items:
        row = {"Item": item}
        for p in puppets:
            row[p] = int(item in puppets[p])
        rows.append(row)
    df_indep = pd.DataFrame(rows).set_index("Item")

    df_indep["Total"] = df_indep.sum(axis=1)

    print(df_indep.transpose().to_latex())
    conduct_tests(df_indep, "AmazonAudiences")


def analyze_advertiser_audiences(data_file, grouping=True):
    # Analyze: individual puppets
    # Rows are categories. For each category, 1 means that puppet saw the category and 0 means not.
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_items = set().union(*data.values())
    rows = []
    for item in all_items:
        row = {"Item": item}
        for puppet in data:
            row[puppet] = int(item in data[puppet])
        row["Total"] = sum(row[g] for g in data)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Item")

    print(df.sort_values("Total", ascending=False))

    # Analyze: gender and age based

    print("=== Gender table (max_age=24) ===")
    print(gender_table_with_norm(df, max_age=24, normalize=True).to_string())

    print("\n=== Age table (gender='female') ===")
    print(age_table_with_norm(df, gender="female", normalize=True).to_string())

    print("--------------------------------------")
    puppets = defaultdict(set)
    for key, cates in data.items():
        puppets[key].update(cates)  # keep key as-is (p294a, p294b)
    # build table
    all_items = set().union(*puppets.values())
    rows = []
    for item in all_items:
        row = {"Item": item}
        for p in puppets:
            row[p] = int(item in puppets[p])
        rows.append(row)
    df_indep = pd.DataFrame(rows).set_index("Item")

    if grouping:
        df_indep = group_advertisers(df_indep)

    df_indep["Total"] = df_indep.sum(axis=1)

    print(df_indep.transpose().to_latex())
    conduct_tests(df_indep, "AdvertiserAudiences")


if __name__ == "__main__":
    analyze_amazon_audiences("AmazonAudiences.json")
    analyze_advertiser_audiences("AdvertiserAudiences.json")
    plt.close('all')
