# -*- coding: utf-8 -*-
## setup
import os
import pandas as pd
import numpy as np
from config_loader import get_data_dir

data_filepath = get_data_dir()
assessments_filename = os.path.join(data_filepath, 'source', 'assessments.csv')
studentAssessment_filename = os.path.join(data_filepath, 'source', 'studentAssessment.csv')
studentInfo_filename = os.path.join(data_filepath, 'source', 'studentInfo.csv')
studentRegistration_filename = os.path.join(data_filepath, 'source', 'studentRegistration.csv')
studentVle_filename = os.path.join(data_filepath, 'source', 'studentVle.csv')
vle_filename = os.path.join(data_filepath, 'source', 'vle.csv')
train_filename = os.path.join(data_filepath, 'sample', 'train.csv')
test_filename = os.path.join(data_filepath, 'sample', 'test.csv')

## load transform and merge target data
# studentAssessment data
data_studentAssessment = pd.read_csv(studentAssessment_filename, encoding='utf-8', dtype={
    'id_student': str,
    'id_assessment': str,
    'date_submitted': np.int32,
    'is_banked': np.int8,
    'score': np.float32
})
data_studentAssessment = data_studentAssessment[data_studentAssessment['is_banked'] == 0].drop(
    columns=['date_submitted', 'is_banked'])
# assessment data exam
data_assessments = pd.read_csv(assessments_filename, encoding='utf-8', dtype={
    'code_module': str,
    'code_presentation': str,
    'id_assessment': str,
    'assessment_type': str,
    'date': np.float32,
    'weight': np.float32
})
data_assessments = data_assessments.drop(columns=['weight'])
# transform the 'score' column using apply
data_studentAssessment['target'] = data_studentAssessment['score'].apply(lambda x: 'pass' if x >= 40 else 'fail')
# merge target data
data_assessments_exam = data_assessments[data_assessments['assessment_type'] == 'Exam'].drop(
    columns=['assessment_type'])
data_target = pd.merge(data_studentAssessment, data_assessments_exam, on=['id_assessment'], how='inner').drop(
    columns=['id_assessment'])[['code_module', 'code_presentation', 'id_student', 'target']]

## load time-sequence data
# time-sequence data
data_assessments_TMA = data_assessments[data_assessments['assessment_type'] == 'TMA'].drop(columns=['assessment_type'])
data_assessments_CMA = data_assessments[data_assessments['assessment_type'] == 'CMA'].drop(columns=['assessment_type'])
# generate and sort studentAssessment data
data_studentAssessment_TMA = \
    pd.merge(data_studentAssessment, data_assessments_TMA, on=['id_assessment'], how='inner').drop(
        columns=['id_assessment'])[['code_module', 'code_presentation', 'id_student', 'date', 'score']].sort_values(
        by=['code_module', 'code_presentation', 'id_student', 'date'], ascending=True)
data_studentAssessment_CMA = \
    pd.merge(data_studentAssessment, data_assessments_CMA, on=['id_assessment'], how='inner').drop(
        columns=['id_assessment'])[['code_module', 'code_presentation', 'id_student', 'date', 'score']].sort_values(
        by=['code_module', 'code_presentation', 'id_student', 'date'], ascending=True)

# Group by the specified columns, sort by 'date' within each group, and aggregate the 'score' values into a tuple
data_studentAssessment_TMA = data_studentAssessment_TMA.groupby(['code_module', 'code_presentation', 'id_student']
                                                                ).apply(
    lambda x: tuple(x.sort_values('date')['score'])).reset_index(name='TMA_score')
data_studentAssessment_CMA = data_studentAssessment_CMA.groupby(['code_module', 'code_presentation', 'id_student']
                                                                ).apply(
    lambda x: tuple(x.sort_values('date')['score'])).reset_index(name='CMA_score')
## merge dynamic data
data_dynamic_mid = pd.merge(data_studentAssessment_TMA, data_studentAssessment_CMA,
                            on=['code_module', 'code_presentation', 'id_student'], how='outer')
data_target_dynamic = pd.merge(data_target, data_dynamic_mid, on=['code_module', 'code_presentation', 'id_student'],
                               how='left')

## load static data
# studentInfo data
data_studentInfo = pd.read_csv(studentInfo_filename, encoding='utf-8', dtype={
    'code_module': str,
    'code_presentation': str,
    'id_student': str,
    'gender': str,
    'region': str,
    'highest_education': str,
    'imd_band': str,
    'age_band': str,
    'num_of_prev_attempts': np.int32,
    'studied_credits': np.int32,
    'disability': str,
    'final_result': str,
})
data_studentInfo = data_studentInfo[data_studentInfo['final_result'] != 'Withdrawn'].drop(columns=['final_result'])
data_studentInfo['presentation_type'] = data_studentInfo['code_presentation'].apply(lambda x: x[4:])
# studentRegistration data
data_studentRegistration = pd.read_csv(studentRegistration_filename, encoding='utf-8', dtype={
    'code_module': str,
    'code_presentation': str,
    'id_student': str,
    'date_registration': np.float32,
    'date_unregistration': np.float32
})
data_studentRegistration = data_studentRegistration[data_studentRegistration['date_unregistration'].isna()].drop(
    columns=['date_registration', 'date_unregistration'])
# studentVle data
data_studentVle = pd.read_csv(studentVle_filename, encoding='utf-8', dtype={
    'id_student': str,
    'code_module': str,
    'code_presentation': str,
    'id_site': str,
    'date': np.int32,
    'sum_click': np.int32
})
# vle data
data_vle = pd.read_csv(vle_filename, encoding='utf-8', dtype={
    'id_site': str,
    'code_module': str,
    'code_presentation': str,
    'activity_type': str
})
data_vle = data_vle.drop(columns=['week_from', 'week_to'])

## transform static data
# Perform inner join for student static data
data_studentAssessment_exam = pd.merge(data_assessments_exam, data_studentAssessment, on='id_assessment', how='inner')
# Create a crosstab to count the presence of each id_student for each code_module
data_course = pd.crosstab(data_studentRegistration['id_student'], data_studentRegistration['code_module'])
# Convert counts to binary presence indicator (1 for presence, 0 for absence)
data_course = data_course.apply(lambda col: col.apply(lambda x: 1 if x > 0 else 0)).astype(np.int32)
# Add a new column 'num_of_social_sciences_mod' that sums the values of columns 'AAA', 'BBB', and 'GGG'
data_course['num_of_social_sciences_mod'] = data_course[['AAA', 'BBB', 'GGG']].sum(axis=1)
# Add a new column 'num_of_STEM_mod' that sums the values of columns 'CCC', 'DDD', 'EEE', and 'FFF'
data_course['num_of_STEM_mod'] = data_course[['CCC', 'DDD', 'EEE', 'FFF']].sum(axis=1)
# Add a new column 'num_of_mod' that sums the values of columns 'num_of_social_sciences_mod', and 'num_of_STEM_mod'
data_course['num_of_mod'] = data_course[['num_of_social_sciences_mod', 'num_of_STEM_mod']].sum(axis=1)
# Calculate 'prop_of_social_sciences_mod' as the proportion of 'num_of_social_sciences_mod' divided by 'num_of_mod'
data_course['prop_of_social_sciences_mod'] = data_course['num_of_social_sciences_mod'].astype(np.float32) / data_course[
    'num_of_mod'].astype(np.float32)
# Calculate 'prop_of_STEM_mod' as the proportion of 'num_of_STEM_mod' divided by 'num_of_mod'
data_course['prop_of_STEM_mod'] = data_course['num_of_STEM_mod'].astype(np.float32) / data_course['num_of_mod'].astype(
    np.float32)
# Rename columns 'AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG' by prefixing with 'reg_in_mod_'
columns_to_rename = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG']
new_column_names = {col: f'reg_in_mod_{col}' for col in columns_to_rename}
data_course.rename(columns=new_column_names, inplace=True)
## The vle data
data_studentVle = pd.merge(data_studentVle, data_vle, on=['code_module', 'code_presentation', 'id_site'], how='inner')
data_studentVle = data_studentVle.drop(columns=['id_site'])
# Group by 'code_module', 'code_presentation', and 'id_student' and calculate the number of unique dates and total rows
data_studentVle_vle_grouped = data_studentVle.groupby(['code_module', 'code_presentation', 'id_student']).agg(
    num_of_click_vle=('date', 'size'),  # Total number of rows
    num_of_day_vle=('date', 'nunique'),  # Count of unique dates
    avg_of_click_vle=('date', lambda x: x.size / x.nunique() if x.nunique() > 0 else 0)  # Ratio of rows to unique dates
).reset_index()
# Group by 'code_module', 'code_presentation', 'id_student', and 'activity_type'
data_studentVle_activity_grouped_mid = data_studentVle.groupby(
    ['code_module', 'code_presentation', 'id_student', 'activity_type']).agg(
    num_of_click=('date', 'size'),  # Total number of rows
    num_of_day=('date', 'nunique'),  # Count of unique dates
    avg_of_click=('date', lambda x: x.size / x.nunique() if x.nunique() > 0 else 0)  # Ratio of rows to unique dates
).reset_index()
# Pivot the DataFrame to wide format for all activity_types
data_studentVle_activity_grouped = data_studentVle_activity_grouped_mid.pivot(
    index=['code_module', 'code_presentation', 'id_student'],
    columns='activity_type',
    values=['num_of_click', 'num_of_day', 'avg_of_click']
).reset_index()
# Rename the columns to include the activity_type
data_studentVle_activity_grouped.columns = [
    f'{col[0]}_{col[1]}' if col[1] != '' else col[0]
    for col in data_studentVle_activity_grouped.columns
]
data_studentVle_activity_grouped = data_studentVle_activity_grouped.fillna(0)
# merge vle total data and activity type data
data_studentVle_grouped = pd.merge(data_studentVle_vle_grouped, data_studentVle_activity_grouped,
                                   on=['code_module', 'code_presentation', 'id_student'], how='inner')

## merge static data
data_static_mid = pd.merge(data_studentInfo, data_course, on=['id_student'], how='inner')
data_static = pd.merge(data_static_mid, data_studentVle_grouped, on=['code_module', 'code_presentation', 'id_student'],
                       how='left')

## merge target and dynamic and static data
data = pd.merge(data_target_dynamic, data_static, on=['code_module', 'code_presentation', 'id_student'], how='inner')

# Train-test split
data_train = data[data['code_presentation'].isin(['2013B', '2013J', '2014B'])]
data_test = data[data['code_presentation'] == '2014J']

# map 'pass' to 0 and 'fail' to 1 in the 'target' column
data_train.loc[data_train['target'] == 'pass', 'label'] = 0
data_train.loc[data_train['target'] == 'fail', 'label'] = 1
data_test.loc[data_test['target'] == 'pass', 'label'] = 0
data_test.loc[data_test['target'] == 'fail', 'label'] = 1
data_train = data_train.drop(columns=['target'])
data_test = data_test.drop(columns=['target'])
data_train.rename(columns={'label': 'target'}, inplace=True)
data_test.rename(columns={'label': 'target'}, inplace=True)
data_train['target'] = data_train['target'].astype(int)
data_test['target'] = data_test['target'].astype(int)

# Save the data to CSV files
data_train.to_csv(train_filename, index=None, header=True, mode='w')
data_test.to_csv(test_filename, index=None, header=True, mode='w')

# Train-test split
data_target_train = data_target[data_target['code_presentation'].isin(['2013B', '2013J', '2014B'])]
data_target_test = data_target[data_target['code_presentation'] == '2014J']

# Display the shapes of the resulting DataFrames
data_target_train.shape, data_target_test.shape

# Import necessary libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Define the directory to save the plots
save_dir = 'E:/Future/new_paper/script/data/output'

# Create combined figure
palette = sns.color_palette("YlGnBu", n_colors=2)
# Create figure with side-by-side subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
# Common style settings
plt.rcParams.update({'font.size': 12})

# 1. Target Histogram (now matches barplot style)
target_counts = data_target_train['target'].value_counts()
bar1 = sns.barplot(x=target_counts.index, y=target_counts.values,
                   palette=palette, ax=ax1)
ax1.set_xlabel('Target', fontsize=12, labelpad=10)
ax1.set_ylabel('Frequency', fontsize=12, labelpad=10)
ax1.set_title("(a) Target Distribution", fontsize=14, y=-0.35, pad=35)
ax1.yaxis.grid(True, linestyle='--', alpha=0.6)

# Add counts for barplot (a)
for p in bar1.patches:
    ax1.annotate(f'{int(p.get_height())}',
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center',
                 xytext=(0, 5),
                 textcoords='offset points')

# 2. Target Pie Chart
score_counts = data_target_train['target'].value_counts()
ax2.pie(score_counts, labels=score_counts.index,
        autopct='%1.1f%%', startangle=140,
        colors=palette[:len(score_counts)])
ax2.set_title("(b) Target Proportion", fontsize=14, y=-0.35, pad=35)
# Add these settings after creating the subplots (ax1, ax2)
for ax in [ax1, ax2]:
    # Set border properties
    for spine in ax.spines.values():
        spine.set_color('black')  # Border color
        spine.set_linewidth(1)  # Border thickness

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.2, wspace=0.3)

# Save combined plot
plt.savefig(os.path.join(save_dir, 'combined_distributions.png'),
            bbox_inches='tight', dpi=300)
plt.close()
