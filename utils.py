import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


def clean_data_slopegph(df_topics, mycol, is_clean, is_norm_month, topk):
    """
    Starting from expanded dataset, clean it up for slopegraph plot
    """
    year1 = 2025
    df1 = df_topics.copy()
    if is_clean:
        df1 = df1[df1.title != 'not much happened today']
    # filter for the two years        
    dfc = df1[mycol].groupby([df1.year, df1[mycol]]).count().unstack().T
    dfc = dfc.drop(2023, axis=1)
    # Normalize per day or month
    months = df1.month.groupby(df1.year).nunique()
    days = df1.day.groupby(df1.year).nunique()
    # count per month: this allows me to normalize years with different number of months
    if is_norm_month:
        dfc = dfc.div(months, axis=1)
    else:
        dfc = dfc.div(days, axis=1)
        dfc *= 18  # convert back to months to make the two comparable
    dfc = dfc.sort_values(year1, ascending=False)
    # get ranks
    for year in dfc.columns:
        dfc[f'rank_{year}'] = dfc[year].rank(ascending=False, method='first')

    # Normalize line width by average count
    cols_rank = [2024, 2025]
    dfc['mean_count'] = dfc[cols_rank].mean(axis=1)
    max_width = 3  # maximum line width
    dfc['line_width'] = (dfc['mean_count'] / dfc['mean_count'].max()) * max_width
    dfr = dfc.head(topk)
    dfr.columns.name = None
    return dfr



def plot_slope_graph(df_nice, topk, is_rank, **kwargs):
    """
    Create a slope graph of the top K topics for two years
    """
    year0, year1 = 2024, 2025
    tfs = kwargs.get('tfs', 16)
    # Plot
    _, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.get_cmap('Greens') 
    colors = cmap(np.linspace(0.3, 1, topk))[::-1]
    dict_rank = {}
    col_for_sort = 'rank_2025' if is_rank else 2025
    ascending = True if is_rank else False
    df_nice = df_nice.sort_values(col_for_sort, ascending=ascending)
    for ct, (idx, row) in enumerate(df_nice.iterrows()):
        if is_rank:
            y0 = row['rank_2024']
            y1 = row['rank_2025']
            str_title = f'Top {topk} AI topics in 2024 and 2025 (Rank)'
            ylab = 'Rank'
        else:
            y0 = row[2024]
            y1 = row[2025]
            str_title = f'Days with news for top {topk} AI topics in 2024 and 2025 (normalized count)'
            ylab = 'Days / Month'
        ax.plot(
            [0, 1],  # x axis: 0 = year0, 1 = year1
            [y0, y1],  # y axis: ranks
            marker='o',
            linewidth=row['line_width'],
            label=idx, 
            color=colors[ct,:],
            markeredgecolor='0.5',
        )
        rk0 = int(row['rank_2024'])
        rk1 = int(row['rank_2025'])
        # dict_rank[idx] = (int(rk0), int(rk1))
        dict_rank[idx] = r'{rk0} $\rightarrow$ {rk1}'.format(rk0=rk0, rk1=rk1)
    ax.set_xlim([-0.2, 1.2])
    ax.legend()
    _, labels = ax.get_legend_handles_labels()
    newlabs = [f'{lab.replace('-',' ').capitalize()} ({dict_rank[lab]})' for lab in labels]
    ax.legend(newlabs, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=10, title = r'Topic rank (2024 $\rightarrow$ 2025)', title_fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([year0, year1])
    ax.set_title(str_title, fontsize=tfs)
    ax.set_ylabel(ylab, fontsize=14)
    return ax, df_nice


def explore_topic(df_topics, topic):
    """
    Explore a specific topic
    """
    flag_topic = df_topics.topics.apply(lambda x: True if topic in x else False)
    df_res = df_topics[flag_topic].topics.value_counts().reset_index()
    n_cats = df_res.shape[0]
    print(f'Topic: {topic} - {n_cats} categories')
    return df_res


def convert_to_dict(res_df, my_topic):
    """
    Convert the result dataframe to a dictionary
    """
    dict_top = {top:my_topic for top in res_df.topics}
    return dict_top


def is_acronym(test_str):
    # Check if the string is all uppercase and contains only letters
    if test_str.isupper() and test_str.isalpha():
        return True
    else:
        return False
    

# def create_data_per_topic_year(df_topics, mycol, dict_category):
#     """
#     For each dict_category, create a df with counts per year
#     """
#     name_col = list(dict_category.values())[0]
#     predf = df_topics.copy()
#     loc_cat = predf[mycol].isin(dict_category.keys())
#     df_cat = predf[loc_cat]
#     df_cnt = df_cat[mycol].groupby(df_cat.year).count()
#     # rename to name_col
#     df_cnt = df_cnt.rename(name_col)
#     return df_cnt


def create_data_per_topic_year(df_topics, mycol, dict_category, date_granularity='year'):
    """
    For each dict_category, create a df with counts per year
    """
    name_col = list(dict_category.values())[0]
    predf = df_topics.copy()
    loc_cat = predf[mycol].isin(dict_category.keys())
    df_cat = predf[loc_cat]
    df_cnt = df_cat[mycol].groupby(df_cat[date_granularity]).count()
    # rename to name_col
    df_cnt = df_cnt.rename(name_col)
    return df_cnt


def normalize_avg(df):
    df_norm = df.copy()
    df_norm['mean_count'] = df_norm[[2024, 2025]].mean(axis=1)
    max_width = 3  # maximum line width
    df_norm['line_width'] = (df_norm['mean_count'] / df_norm['mean_count'].max()) * max_width
    
    return df_norm

def rank_cols(df):
    df_rank = df.copy()
    for year in [2024, 2025]:
        df_rank[f'rank_{year}'] = df_rank[year].rank(ascending=False, method='first')

    return df_rank


def create_whole_data_per_topic_year(df_topics, mycol, dict_all, topk, date_granularity='year'):
    """
    Concatenate dfs for all categories/dicts
    """
    for d, dict_cat in enumerate(dict_all):
        pre_cnt = create_data_per_topic_year(df_topics, mycol, dict_cat, date_granularity)
        if d == 0:
            df_cats = pre_cnt
        else:
            df_cats = pd.concat([df_cats, pre_cnt], axis=1)
    df_cats = df_cats.fillna(0)
    df_cats = df_cats.T
    df_cats.columns.name = None
    # now normalize: using only days
    df1 = df_topics.copy() 
    days = df1.day.groupby(df1.year).nunique()
    df_cats_norm = df_cats.div(days, axis=1) * 18
    df_cats_norm = df_cats_norm.sort_values(2025, ascending=False)
    # include ranks
    df_cats_norm = rank_cols(df_cats_norm)
    df_cats = rank_cols(df_cats)
    # get topk
    df_cats_norm = df_cats_norm.head(topk)
    df_cats = df_cats.head(topk)
    # normalize for plotting
    df_cats_norm = normalize_avg(df_cats_norm)
    df_cats = normalize_avg(df_cats)
    df_cats = df_cats.drop(2023, axis=1)
    df_cats_norm = df_cats_norm.drop(2023, axis=1)

    return df_cats_norm, df_cats

def create_whole_data_per_topic_year_ts(df_topics, mycol, dict_all, topk, date_granularity):
    """
    Concatenate dfs for all categories/dicts: for time series
    """
    for d, dict_cat in enumerate(dict_all):
        pre_cnt = create_data_per_topic_year(df_topics, mycol, dict_cat, date_granularity)
        if d == 0:
            df_cats = pre_cnt
        else:
            df_cats = pd.concat([df_cats, pre_cnt], axis=1)
    df_cats = df_cats.fillna(0)
    df_cats = df_cats
    df_cats.columns.name = None
    # restrict to topk using avg
    df_avg = df_cats.mean(axis=0).sort_values(ascending=False)
    top_cols = df_avg.head(topk).index
    df_cats = df_cats[top_cols]
    return df_cats

def create_time_series(df_aug, mycol, topk):
    """
    Create time series (months) for the topk categories
    """
    top_50_tops = df_aug[mycol].value_counts().head(topk).index
    loc_top50 = df_aug[mycol].isin(top_50_tops)
    df_top50 = df_aug[loc_top50]
    df_aug50 = df_top50[mycol].groupby([df_top50.month, df_top50[mycol]]).count().unstack()
    df_aug50.fillna(0, inplace=True)
    return df_aug50

def plot_top_k_topics(df_aug_top, topk, single_out_topics, mycol, is_smooth, **kwargs):
    """
    Plot time series of topk categories (applies for topics, models, companies)
    """
    title_var = 'topics' if mycol.find('topic') >= 0 else 'models' if mycol.find('model') >= 0 else 'companies'
    ma_parameter = kwargs.get('ma_parameter', 3)
    alpha_par = kwargs.get('alpha_par', 0.5)
    # plot
    _, ax = plt.subplots(figsize=(18, 6))
    alpha = 1 if len(single_out_topics) == 0 else alpha_par
    pre_cols = df_aug_top.columns
    if is_smooth:
        df_aug_top = df_aug_top.rolling(ma_parameter).mean()
        df_aug_top = df_aug_top.loc[df_aug_top.index[ma_parameter:]]
    cols_for_plot =  pre_cols if len(single_out_topics) == 0 else [col for col in pre_cols if col not in single_out_topics]
    df_aug_top[cols_for_plot].plot(ax=ax, alpha=alpha, color=sns.color_palette("husl", len(df_aug_top.columns)))
    # single out topics with a different palette
    cols_for_so = [col for col in single_out_topics if col in pre_cols]
    if len(cols_for_so) > 0:
        color_so = sns.color_palette("crest", len(cols_for_so))
        df_aug_top[cols_for_so].plot(ax=ax, color=color_so, linewidth=3, alpha = 1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title = f'Top {topk} topics')    
    ax.set_title(f'News count per month (Top {topk} {title_var})')
    ax.set_xlabel('')
    return ax, df_aug_top

def fix_model_name(model_name, my_dict, is_company):
    """
    Fix the model name to map to OpenAI
    """
    if re.search(r'o\d+', model_name):
        if is_company:
            var_to_return =  'OpenAI'
        else:  # otherwise is the family of models
            var_to_return =  'o'
    else:
        var_to_return = my_dict[model_name]
    
    return var_to_return
