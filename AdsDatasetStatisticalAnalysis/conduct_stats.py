import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import scipy
import math
from statsmodels.stats.multitest import multipletests

plt.rcParams.update({'text.usetex': True})
plt.tight_layout()

SAMPLE_NUM = 10000

persona_age = {
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

persona_gender = {
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

np.random.seed(1)


def apply_multiple_tests_correction(p_values, method='bonferroni'):
    """
    Apply multiple tests correction to get adjusted p-values.
    """
    return multipletests(p_values, method=method)[1]


def generate_couples_from_dataframe(df):
    couples_list = []
    for row_index, row in df.iterrows():
        for col_index, count in row.items():
            # Append the couple (row_index, col_index) count times
            couples_list.extend([(row_index, col_index)] * count)
    return couples_list


def from_csv(filename):
    d = pd.read_csv(filename, index_col=0)
    d = generate_couples_from_dataframe(d)
    return d


def l1_categories(df):
    genders = df.sum(axis=1)
    normalized_df = df.div(genders, axis=0)
    diff = normalized_df.iloc[0]-normalized_df.iloc[1]
    value = diff.abs().sum()
    return value


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

def dataframe_age_categories(key, gender="female"):
    young_cat = {}
    old_cat = {}
    with open("ads_db.json", 'r') as f:
        d = json.load(f)
    for entry in d:
        g = persona_gender[entry["persona"][:-1]]
        if g == gender:
            age = persona_age[entry["persona"][:-1]]
            if age <=24:
                young_cat[entry[key]] = young_cat.get(entry[key], 0) + 1
            else:
                old_cat[entry[key]] = old_cat.get(entry[key], 0) + 1
    df = pd.DataFrame([young_cat, old_cat], index=['18-24', '25+'])
    df = df.fillna(0)
    return df

def dataframe_gender_categories(key, min_age=0, max_age=100):
    male_cat = {}
    female_cat = {}
    with open("ads_db.json", 'r') as f:
        d = json.load(f)
    for entry in d:
        age = persona_age[entry["persona"][:-1]]
        if age <= max_age and age >= min_age:
            gender = persona_gender[entry["persona"][:-1]]
            if gender == "male":
                male_cat[entry[key]] = male_cat.get(entry[key], 0) + 1
            else:
                female_cat[entry[key]] = female_cat.get(entry[key], 0) + 1
    df = pd.DataFrame([male_cat, female_cat], index=['male', 'female'])
    df = df.fillna(0)
    return df


def test_llm_labels_multi():
    df = dataframe_gender_categories("category", max_age=24)
    value = l1_categories(df)
    perm_values = []
    for _ in range(SAMPLE_NUM):
        rdf = generate_random_permutation(df, "male", "female")
        perm_values.append(l1_categories(rdf))
    p,target = estimate_probability(perm_values, value)
    permutation_test_plot(perm_values, value, savefile="permutation_gender_categories.pdf", title=f"S_m")

    df = dataframe_age_categories("category", gender="female")
    value = l1_categories(df)
    perm_values = []
    for _ in range(SAMPLE_NUM):
        rdf = generate_random_permutation(df, "18-24", "25+")
        perm_values.append(l1_categories(rdf))
    p,target = estimate_probability(perm_values, value)
    permutation_test_plot(perm_values, value, savefile="permutation_age_categories.pdf", title=f"S_m")


def category_stat(df, category):
    genders = df.sum(axis=1)
    new_df = df[category]

    normalized_df = new_df.div(genders, axis=0)
    diff = normalized_df.iloc[0]-normalized_df.iloc[1]
    return diff


def single_category_gender_analysis():
    df = dataframe_gender_categories("category_oracle",max_age=24)
    cats = df.columns
    for category in sorted(cats):
        value = category_stat(df, category)
        perm_values = []
        for _ in range(SAMPLE_NUM):
            rdf = generate_random_permutation(df, "male", "female")
            perm_values.append(category_stat(rdf, category))
        p = estimate_probability(perm_values, value)
        permutation_test_plot(perm_values, value, savefile=category+".pdf", title=f"S")
        print(f"{category} & {df[category]['male']} & {df[category]['female']} & {p}({int(df[category].sum())})")


def single_category_age_analysis():
    df = dataframe_age_categories("category_oracle", gender="female")
    cats = df.columns
    for category in sorted(cats):
        value = category_stat(df, category)
        perm_values = []
        for _ in range(SAMPLE_NUM):
            rdf = generate_random_permutation(df, "18-24", "25+")
            perm_values.append(category_stat(rdf, category))
        p = estimate_probability(perm_values, value)
        permutation_test_plot(perm_values, value, savefile=category+".pdf", title=f"S")
        print(f"{category} & {df[category]['18-24']} & {df[category]['25+']} & {p}({int(df[category].sum())})")


def ads_numbers():
    df = dataframe_age_categories("category", gender="female")
    young_female = df.loc['18-24'].sum()
    old_female = df.loc['25+'].sum()
    df = dataframe_age_categories("category", gender="male")
    young_male = df.loc['18-24'].sum()
    old_male = df.loc['25+'].sum()
    #print(f"{young_male+young_female}, male: {young_male}, female: {young_female}")

    total = old_male + young_male + young_female + old_female
    print(f"\\begin{{table}}")
    print(f"\\begin{{tabular}}{{l|c|c}}")
    print(f"Age Range & Male & Female \\\\")
    print(f"\\hline")
    print(f"\\hline")
    print(f"18-24 & {int(young_male)} ({round(young_male*100/total, 2)}\\%) & {int(young_female)} ({round(young_female*100/total, 2)}\\%) \\\\")
    print(f"25+ & {int(old_male)} ({round(old_male*100/total, 2)}\\%) & {int(old_female)} ({round(old_female*100/total,2)}\\%) \\\\")
    print(f"\\end{{tabular}}")
    print(f"\\caption{{Total received ads: {int(total)}.}}")
    print(f"\\label{{tab:adv_received}}")
    print(f"\\end{{table}}")


def print_table(categories, groups, header):
    print(f"\\begin{{table}}")
    print(f"\\begin{{tabular}}{{l|cc|ccc}}")
    #print(f"&  \\multicolumn{{5}}{{c}}{{{header}}} \\\\")
    #print(f"&  & & \\multicolumn{{3}}{{c}}{{Target}} \\\\")
    print(f"Category & Advertisements & P-value & Direction & Log Odds & Odds \\\\")
    print(f"\\hline")
    print(f"\\hline")
    for cat in categories:
        if categories[cat]['p-value'] < 0.05:
            print("\\bfseries ", end='')
        print(f"{cat} ", end='')
        for el in [categories[cat]['sum'], round(categories[cat]['p-value'], 3), groups[categories[cat]['target']], categories[cat]['logodds'] , categories[cat]['horseodds']]:
            print(f" & ", end='')
            if categories[cat]['p-value'] < 0.05:
                print("\\bfseries ", end='')
            print(f"{el}", end='')
        print(f"\\\\")
    print(f"\\end{{tabular}}")
    print(f"\\caption{{Statistical test results on {header}.}}")
    print(f"\\label{{tab:permutation_{header}}}")
    print(f"\\end{{table}}")


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


def odds(df, group1, group2, category):
    groups_num = {'male': 5, 'female': 11, '18-24': 11, '25+': 5}
    #res = (df.loc[group1][category]/groups_num[group1], df.loc[group2][category]/groups_num[group2])
    res = (df.loc[group1][category]/df.loc[group1].sum(), df.loc[group2][category]/df.loc[group2].sum())
    return res


def log_odds(df, group1, group2, category):
    p1, p2 = odds(df, group1, group2, category)
    if p1 > 0 and p2 > 0:
        return round(math.log(p1/p2), 3)
    else:
        return "N.D."


def horse_odds(df, group1, group2, category):
    p1, p2 = odds(df, group1, group2, category)
    minp = min(p1, p2)
    if minp > 0:
        return f"{round(p1/minp, 2)}:{round(p2/minp, 2)}"
    else:
        return "N.D."


def table_plot(df_llm, df_oracle, permutations_llm, permutations_oracle, group1, group2, label, header):
    categories = {}
    cats = set(list(df_llm.columns))# + list(df_oracle.columns))
    for category in sorted(cats):
        if category in df_llm.columns:
            value = category_stat(df_llm, category)
            perm_values = []
            for rdf in permutations_llm:
                perm_values.append(category_stat(rdf, category))
            p,target = estimate_probability(perm_values, value)
            table = contingency_table(df_llm, category)
            p_fisher = fisher_pvalue(table)
            p_boschloo = boschloo_pvalue(table)
            logodds = log_odds(df_llm, group1, group2, category)
            horseodds = horse_odds(df_llm, group1, group2, category)
            permutation_test_plot(perm_values, value, savefile="permutation_"+label+"_"+category.lower()+".pdf", title=f"S")
        else:
            p_fisher = "N.D."
            p_boschloo = "N.D."
            p = "N.D."
            logodds = "N.D."
            horseodds = "N.D."
        if category in df_oracle.columns:
            value = category_stat(df_oracle, category)
            perm_values = []
            for rdf in permutations_oracle:
                perm_values.append(category_stat(rdf, category))
            p_oracle,target = estimate_probability(perm_values, value)
            p_oracle = round(p_oracle, 3)
        else:
            p_oracle = "N.D."

        categories[category] = {"p-value": p, "p-value-oracle": p_oracle, group1: df_llm[category][group1], group2: df_llm[category][group2], "sum": int(df_llm[category].sum()), "fisher": p_fisher, "boschloo": p_boschloo, "target": target, "logodds": logodds, "horseodds": horseodds}
    print_table(categories, [group1, group2], header)

def test_llm_labels():
    df_llm = dataframe_gender_categories("category",max_age=24)
    permutations_llm = []
    for _ in range(SAMPLE_NUM):
        rdf = generate_random_permutation(df_llm, "male", "female")
        permutations_llm.append(rdf)

    df_oracle = dataframe_gender_categories("category_oracle",max_age=24)
    permutations_oracle = []
    for _ in range(SAMPLE_NUM):
        rdf = generate_random_permutation(df_oracle, "male", "female")
        permutations_oracle.append(rdf)
    table_plot(df_llm, df_oracle, permutations_llm, permutations_oracle, "male", "female", "gender", "(18-24)")

    df_llm = dataframe_age_categories("category", gender="female")
    permutations_llm = []
    for _ in range(SAMPLE_NUM):
        rdf = generate_random_permutation(df_llm, "18-24", "25+")
        permutations_llm.append(rdf)

    df_oracle = dataframe_age_categories("category_oracle", gender="female")
    permutations_oracle = []
    for _ in range(SAMPLE_NUM):
        rdf = generate_random_permutation(df_oracle, "18-24", "25+")
        permutations_oracle.append(rdf)
    table_plot(df_llm, df_oracle, permutations_llm, permutations_oracle, "18-24", "25+", "age", "(female)")


def table_cmp(cats, llm_df, human_df, group1, group2):
    print(f"\\begin{{table}}")
    print(f"\\begin{{tabular}}{{l|cc|cc}}")
    print(f"&  \\multicolumn{{2}}{{c}}{{human cat.}} & \\multicolumn{{2}}{{c}}{{llm cat.}} \\\\")
    print(f"Category & {group1} & {group2} & {group1} & {group2}\\\\")
    print(f"\\hline")
    print(f"\\hline")
    for cat in cats:
        human_group1 = int(human_df.loc[group1][cat] if cat in human_df.columns else 0)
        human_group2 = int(human_df.loc[group2][cat] if cat in human_df.columns else 0)
        llm_group1 = int(llm_df.loc[group1][cat] if cat in llm_df.columns else 0)
        llm_group2 = int(llm_df.loc[group2][cat] if cat in llm_df.columns else 0)
        print(f"{cat} & {human_group1} & {human_group2} & {llm_group1} & {llm_group2} \\\\")
    print(f"\\end{{tabular}}")
    print(f"\\caption{{Received ads.}}")
    print(f"\\label{{tab:cat_adv_received}}")
    print(f"\\end{{table}}")


def cmp_llm_human_labels():
    llm_df_gender = dataframe_gender_categories("category",max_age=24)
    human_df_gender = dataframe_gender_categories("category_oracle",max_age=24)
    llm_df_age = dataframe_age_categories("category", gender="female")
    human_df_age = dataframe_age_categories("category_oracle", gender="female")
    cats = sorted(set(list(llm_df_gender.columns) + list(human_df_gender.columns) + list(llm_df_age) + list(human_df_age)))

    table_cmp(cats, llm_df_gender, human_df_gender, "male", "female")
    table_cmp(cats, llm_df_age, human_df_age, "18-24", "25+")


def age_proportions():
    with open("ads_db.json", 'r') as f:
        d = json.load(f)

    info = {"female": {}, "male": {}}
    totals = {"female": {}, "male": {}}

    for e in d:
        persona_key = e["persona"][:-1]
        if persona_key not in persona_age or persona_key not in persona_gender:
            continue
        a = persona_age[persona_key]
        g = persona_gender[persona_key]
        c = e["category"]
        if a not in info[g]:
            info[g][a] = {}
            totals[g][a] = 0
        if c not in info[g][a]:
            info[g][a][c] = 0
        info[g][a][c] += 1
        totals[g][a] += 1

    for g, ages in info.items():
        for a, cats in ages.items():
            for cat, count in cats.items():
                info[g][a][cat] /= totals[g][a]

    aggregated = {"female": {"18-24": {}, "25+": {}}, "male": {"18-24": {}, "25+": {}}}
    for g in info:
        for a in info[g]:
            age_group = "18-24" if 18 <= a <= 24 else "25+"
            for cat, val in info[g][a].items():
                if cat not in aggregated[g][age_group]:
                    aggregated[g][age_group][cat] = 0
                aggregated[g][age_group][cat] += val
        # Normalize again within age group
        for ag in aggregated[g]:
            total = sum(aggregated[g][ag].values())
            for cat in aggregated[g][ag]:
                aggregated[g][ag][cat] /= total

    plot_combined_stacked_bar_vertical(aggregated, "gender_age_group_distribution.pdf")


def plot_combined_stacked_bar_horizontal(data, savefile):
    """
    Plots a horizontally stacked bar chart with two bars per gender:
    one for age 18–24 and one for 25+, excluding empty groups.
    """
    labels = set()
    filtered_data = []

    for gender in ['female', 'male']:  # consistent order
        for age_group in ['18-24', '25+']:
            group_data = data[gender][age_group]
            if not group_data:  # Skip if there's no data
                continue
            labels.update(group_data.keys())
            filtered_data.append((f"{gender.capitalize()} {age_group}", group_data))

    labels = sorted(labels)
    y_labels = []
    bar_data = []

    for label, group_data in filtered_data:
        y_labels.append(label)
        bar_data.append([group_data.get(cat, 0) for cat in labels])

    if not bar_data:
        print("No data to plot.")
        return

    y_pos = np.arange(len(y_labels))
    bar_data = np.array(bar_data)

    fig, ax = plt.subplots(figsize=(10, 4))
    left = np.zeros(len(y_labels))

    colors = plt.cm.tab20.colors
    for i, label in enumerate(labels):
        vals = bar_data[:, i]
        ax.barh(y_pos, vals, left=left, label=label, color=colors[i % len(colors)])
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Proportion")
    ax.set_xlim(0, 1)
    ax.set_title("Ad Category Proportions by Gender and Age Group")
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(savefile)


def plot_combined_stacked_bar_vertical(data, savefile):
    """
    Plots a vertically stacked bar chart with one bar per gender-age group,
    excluding groups with no data.
    """
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 15,
    })

    labels = set()
    filtered_data = []

    for gender in ['female', 'male']:  # consistent order
        for age_group in ['18-24', '25+']:
            group_data = data[gender][age_group]
            if not group_data:
                continue
            labels.update(group_data.keys())
            filtered_data.append((f"{gender.capitalize()} {age_group}", group_data))

    labels = sorted(labels)
    x_labels = []
    bar_data = []

    for label, group_data in filtered_data:
        x_labels.append(label)
        bar_data.append([group_data.get(cat, 0) for cat in labels])

    if not bar_data:
        print("No data to plot.")
        return

    x_pos = np.arange(len(x_labels))
    bar_data = np.array(bar_data).T  # Transpose to stack correctly

    fig, ax = plt.subplots(figsize=(10, 10))
    bottom = np.zeros(len(x_labels))

    colors = plt.cm.tab20.colors
    for i, label in enumerate(labels):
        vals = bar_data[i]
        ax.bar(x_pos, vals, bottom=bottom, label=label, color=colors[i % len(colors)])
        bottom += vals

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.set_title("Ad Category Proportions by Gender and Age Group", fontweight='bold')
    #ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.legend(title='Category', loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3)
    # Adjust layout to make space for the top legend
    #plt.subplots_adjust(top=3)
    plt.tight_layout()
    plt.savefig(savefile, dpi=300, bbox_inches='tight')


def plot_combined_stacked_bar_including_empty(data, savefile):
    """
    Plots a horizontally stacked bar chart with two bars per gender:
    one for age 18–24 and one for 25+.
    """
    labels = set()
    for gender in data:
        for age_group in data[gender]:
            labels.update(data[gender][age_group].keys())
    labels = list(sorted(labels))

    y_labels = []
    bar_data = []

    for gender in ['female', 'male']:  # consistent order
        for age_group in ['18-24', '25+']:
            y_labels.append(f"{gender.capitalize()} {age_group}")
            bar_data.append([data[gender][age_group].get(label, 0) for label in labels])

    y_pos = np.arange(len(y_labels))
    bar_data = np.array(bar_data)

    fig, ax = plt.subplots(figsize=(10, 4))
    left = np.zeros(len(y_labels))

    colors = plt.cm.tab20.colors
    for i, label in enumerate(labels):
        vals = bar_data[:, i]
        ax.barh(y_pos, vals, left=left, label=label, color=colors[i % len(colors)])
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Proportion")
    ax.set_xlim(0, 1)
    ax.set_title("Ad Category Proportions by Gender and Age Group")
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(savefile)


def persona_stat(data, cats, tot):
    stats = {}
    for persona in data:
        tot_persona = sum([data[persona][cat] for cat in data[persona]])
        phi = 0
        for cat in cats:
            E = cats[cat]/tot
            O = 0
            if cat in data[persona]:
                O = data[persona][cat]/tot_persona
            phi += ((O-E)**2)/E
        stats[persona] = phi
    return stats


def persona_random_permutation(data):
    d = persona_data_unpack(data)
    y,x = zip(*d)
    x = list(x)
    np.random.shuffle(x)
    d = list(zip(y,x))
    d = persona_data_pack(d)
    return d


def persona_data_unpack(data):
    l = []
    for persona in data:
        for cat in data[persona]:
            for _ in range(data[persona][cat]):
                l.append((persona, cat))
    return l


def persona_data_pack(lst):
    data = {}
    for e in lst:
        persona = e[0]
        cat = e[1]
        if persona not in data:
            data[persona] = {}
        if cat not in data[persona]:
            data[persona][cat] = 0
        data[persona][cat] += 1
    return data


def test_persona():
    with open("ads_db.json", 'r') as f:
        d = json.load(f)
    data = {}
    cats = {}
    tot = 0
    for e in d:
        age = persona_age[e["persona"][:-1]]
        gender = persona_gender[e["persona"][:-1]]
        cat = e["category"]
        if (age, gender) not in data:
            data[(age,gender)] = {}
        if cat not in data[(age,gender)]:
            data[(age,gender)][cat] = 0
        if cat not in cats:
            cats[cat] = 0
        data[(age,gender)][cat] += 1
        cats[cat] += 1
        tot += 1

    values = []
    for _ in range(SAMPLE_NUM):
        d = persona_random_permutation(data)
        v = persona_stat(d, cats, tot)
        values.append(v)

    observed = persona_stat(data, cats, tot)
    for persona in data:
        perm_values = [d[persona] for d in values]
        p,target = estimate_probability(perm_values, observed[persona])
        permutation_test_plot(perm_values, observed[persona], savefile=f"permutation_persona_{persona}.pdf", title=f"{persona}")
        print(f"{persona[0]} & {persona[1]} & {p}")


if __name__ == "__main__":
    test_llm_labels()
    print(f"----------------------------------")
    ads_numbers()
    print(f"----------------------------------")
    test_llm_labels_multi()
    print(f"----------------------------------")
    cmp_llm_human_labels()
    print(f"----------------------------------")
    age_proportions()
    #print(f"----------------------------------")
    #test_persona()
