import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import spacy
import spacy.cli
# spacy.cli.download("en_core_web_lg")
# import en_core_web_lg
# nlp = en_core_web_lg.load()
import requests
import re
import csv
import math
import sys
from collections import Counter
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
# nltk.download('brown')brown
from nltk.corpus import brown
import random


""" load """
plt.ion()
df = pd.read_csv('/Users/xs/PycharmProjects/TREATRCOMM/data/patients_data.csv')
print(df.info(), df.head())

""" preprocessing """
print(df.isna().sum())
null_num = df.isna()['Treatment'].sum()
print (f"Percentage of missing value in \' Treatment \' is {null_num * 100 /df.shape[0]} % ")
plt.figure() # figsize=(15,7)
plt.bar(df.columns, df.isna().sum())
plt.title('Missing value for each columns')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')

df.dropna(subset=['Treatment'], inplace=True)
print(df.isna().sum())

""" data cleaning """
# remove words not belong to the treatments
l = ['Fatigue', 'Anxious mood','Pain', 'Insomnia', 'Skin pain', 'Psoriatic plaques (scaly patches)', 'swelling)']
record = set()
for index, row in df.iterrows():
    new = re.split(',|"', row['Treatment'])
    for word in l:
        if word in new:
            record.add(index)
df.drop(list(record), inplace=True)
df.reset_index(drop=True, inplace=True)

df.sort_values('Condition', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Final size of the dataset is: {df.shape[0]}")


""" EDA """

class EDA:
    def __init__(self):
        pass


class Mapping(EDA):

    # create mapping
    def one_to_n_mapping(self, df, mapping_from, mapping_to):
        feature_name = df[mapping_from].unique()
        mapping = dict()
        visited = set()
        for feature in feature_name:
            if feature not in visited:
                visited.add(feature)
                mapping[feature] = dict()
        for feature in feature_name:
            new = df.loc[df[mapping_from] == feature]
            for index, row in new.iterrows():
                results = row[mapping_to].split(',')
                for result in results:
                    mapping[feature][result] = mapping[feature].get(result, 0) + 1
            mapping[feature] = dict(sorted(mapping[feature].items(), key=lambda x: x[1], reverse=True))
        print('mapping check', mapping)
        return mapping

    def n_to_one_mapping(self, df, mapping_from, mapping_to):
        visited = set()
        mapping = dict()
        for index, row in df.iterrows():
            new = re.split(',|"', row[mapping_from])
            for feature in new:
                if feature not in visited and feature:
                    visited.add(feature)
                    mapping[feature] = set()

        m = self.one_to_n_mapping(df, mapping_to, mapping_from)
        for index, row in df.iterrows():
            new = re.split(',|"', row[mapping_from])
            for feature in new:
                for m_key, m_value in m.items():
                    if feature not in m_value:
                        continue
                    mapping[feature].add(m_key)
        return mapping


class Visualization(EDA):

    # word cloud visualization
    def word_cloud(self, df, mapping_from, mapping=None):
        if mapping:
            res = ''
            for feature in df[mapping_from].unique():
                a = mapping[feature]
                k = ' '.join(a)
                res += k
        else:
            df_feature = df[mapping_from]
            res = ' '.join(df_feature)

        wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(res)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')

    def top_n_visualization(self, dictionary, first_n=10, xlabel=None, ylabel=None):
        plt.figure()
        sns_2 = sns.barplot(x=list(dictionary.keys())[:first_n], y=list(dictionary.values())[:first_n])
        sns_2.set_title("Most {} frequent {}".format(first_n, xlabel))
        sns_2.set_xlabel(xlabel)
        sns_2.set_ylabel("Number of {}".format(ylabel))
        plt.xticks(rotation=45)


map = Mapping()

# each condition will have n treatments
mapping_condition_treatment = map.one_to_n_mapping(df, mapping_from='Condition', mapping_to='Treatment')
print('condition-treatments: ', mapping_condition_treatment)

# each condition will have n symptoms
mapping_condition_symptom = map.one_to_n_mapping(df, mapping_from='Condition', mapping_to='Symptom')

# each symptom occurs in n conditions
mapping_symptom_condition = map.n_to_one_mapping(df, mapping_from='Symptom', mapping_to='Condition')

# each treatment occurs in n conditions
mapping_treatment_condition = map.n_to_one_mapping(df, mapping_from='Treatment', mapping_to='Condition')

print(len(mapping_condition_treatment['ADD (Attention Deficit Disorder)']))
print("number of unique conditions:", len(df['Condition'].unique()))
print('\n')
condition_dict = df['Condition'].value_counts()
print(condition_dict)

# Most n frequent Conditions
vs = Visualization()
condition_dict = dict(condition_dict)
vs.top_n_visualization(condition_dict, first_n=10, xlabel="Conditions", ylabel="Patients")
print('condition-observations: ', condition_dict)
print(len(condition_dict))

condition_length_dict = {key: len(value) for key, value in mapping_condition_treatment.items()}
condition_length_dict = dict(sorted(condition_length_dict.items(), key=lambda x: x[1], reverse=True))
vs.top_n_visualization(condition_length_dict, first_n=10, xlabel="Conditions", ylabel="Treatments")
print('condition-treatments: ', condition_length_dict)
print(len(condition_length_dict))

# Most n frequent Treatments
treatment_length_dict = {key: len(value) for key, value in mapping_treatment_condition.items()}
treatment_length_dict = dict(sorted(treatment_length_dict.items(), key=lambda x: x[1], reverse=True))
vs.top_n_visualization(treatment_length_dict, first_n=10, xlabel="Treatments", ylabel="Conditions")
print('treatment-conditions: ', treatment_length_dict)
print(len(treatment_length_dict))

# Most n frequent Symptoms
symptom_length_dict = {key: len(value) for key, value in mapping_symptom_condition.items()}
symptom_length_dict = dict(sorted(symptom_length_dict.items(), key=lambda x: x[1], reverse=True))
vs.top_n_visualization(symptom_length_dict, first_n=10, xlabel="Symptoms", ylabel="Conditions")
print('symptom-conditions: ', symptom_length_dict)
print(len(symptom_length_dict))


""" Recommendation """
new_df = df.drop(columns=['Name', 'City', 'State'])
# data = new_df.to_numpy()
data_train, data_test, y_train, y_test = train_test_split(new_df.iloc[:, :-1], new_df.iloc[:, -1], test_size=0.2, random_state=0)
df_train = pd.concat([data_train, y_train], axis=1)
data_train = df_train.to_numpy()
data_test = data_test.to_numpy()
y_test = y_test.to_numpy()


class ReferenceSystem:
    """ Reference System: Trivial and Baseline """
    def __init__(self, data, k, n):
        self.data = data
        self.top_k_users = k
        self.top_n_treatments = n

    def trivial_system(self):
        # random recommendation --> random generate 3 patients then random select N treatments
        # run 10 times and take the average of the final results.
        combine_treatment = []
        for _ in range(self.top_k_users):
            patient_index = random.randint(0, len(self.data) - 1)
            combine_treatment += self.data[patient_index][-1].split(',')
        # combine their treatments and random choose N
        treatments_trivial = []
        for _ in range(self.top_n_treatments):
            treatments_trivial.append(combine_treatment[random.randint(0, len(combine_treatment) - 1)])
        return treatments_trivial

        # run more than 10 times to take avg:
        # trivial 1. random recommendation base on index without any prior knowledge (658 * length choose randint?)
        # trivial 2. random choose k patients (similar to recommend top k in RS) and
        #            combine their treatments as the y_pred (658 * (k1 + k2 + ...))

    def baseline_system(self, symptom, disease, age, gender, mapping_condition_treatment_for_baseline):

        # baseline 1. POP (popular products): this model recommends the most popular products in the training set.
        #            (658 * length choose randint?)
        # baseline 2. random recommendation within condition-treatments mapping (658 * length choose randint?)
        # baseline 3. recommend top k frequent treatments within condition-treatments mapping (658 * k) 教授：根据这个symptom的最常用的treatment来推荐，topk就推荐k个最常见的
        # Q: length of the prediction? 5? random int?

        if disease not in mapping_condition_treatment_for_baseline:
            return []
        return list(mapping_condition_treatment_for_baseline[disease])[:self.top_n_treatments]

        # new = self.data.loc[(self.data['Condition'] == disease) & (self.data['Gender'] == gender) & (abs(self.data['Age'] - age) < 5)].to_numpy()
        # if len(new) == 0:
        #     return []
        # combine_treatment = dict()
        # for _ in range(self.top_k_users):
        #     patient_index = random.randint(0, len(new) - 1)
        #     for treat in new[patient_index][-1].split(','):
        #         combine_treatment[treat] = combine_treatment.get(treat, 0) + 1
        # res = dict(sorted(combine_treatment.items(), key=lambda x: x[1], reverse=True))
        # return list(res)[:self.top_n_treatments]


def evaluation(model, reference):
    """ Evaluation """
    # Micro precision (joint1 + joint2 + ... / yhat1 + yhat2 + ...), recall, f1
    count = 0
    record = dict()
    xgrid_min, xgrid_max = 0, -sys.maxsize
    y_len, y_hat_len, total_intercept = 0, 0, 0
    for i in range(len(data_test)):
        count += 1
        # print('==='*20)
        # print('Patient No.' + str(count), data_test[i])
        symptom, condition, age, gender = data_test[i][3], data_test[i][2], data_test[i][1], data_test[i][0]
        if reference == 'trivial':
            y_pred = model.trivial_system()
        elif reference == 'baseline':
            y_pred = model.baseline_system(symptom, condition, age, gender, mapping_baseline)
        else:
            y_pred = model.collaborative_filter(symptom, condition, age, gender)
        xgrid_min, xgrid_max = 0, max(xgrid_max, len(y_pred))
        record[len(y_pred)] = record.get(len(y_pred), 0) + 1
        # print('recommendation:', y_pred)
        y_true = y_test[i].split(',')
        # print('y_true', y_true)
        total_intercept += len(set(y_true).intersection(set(y_pred)))

        y_len += len(y_true)
        y_hat_len += len(y_pred)
        # print(total_intercept, y_len, y_hat_len)
    precision_k = total_intercept / y_hat_len
    recall_k = total_intercept / y_len
    f1_k = (2 * precision_k * recall_k) / (precision_k + recall_k)
    # print('top{} similar patients, with top{} treatments precision = {}, recall = {}, f1 = {}'.format(K, tre_num, precision_k, recall_k, f1_k))
    # plt.figure()
    # plt.title('Distribution of the length of prediction')
    # plt.bar(*zip(*record.items()))
    # plt.xlabel('Length of the prediction')
    # plt.ylabel('Number of Datapoints')
    return precision_k, recall_k


if __name__ == '__main__':
    """ semantic similarity + no fuzzy search; w+h distance measure; visited set """
    K = 2
    N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    """ Trivial system """
    run_time = 10
    diff_n_pre, diff_n_rec, diff_n_f1 = [], [], []
    tre_num = 1
    for n in N:
        rfs_trivial = ReferenceSystem(data=data_train, k=K, n=n)
        record_pre, record_rec = [], []
        for _ in range(run_time):
            pre, rec = evaluation(rfs_trivial, 'trivial')
            record_pre.append(pre)
            record_rec.append(rec)
        precision_trivial = sum(record_pre) / len(record_pre)
        recall_trivial = sum(record_rec) / len(record_rec)
        f1_trivial = (2 * precision_trivial * recall_trivial) / (precision_trivial + recall_trivial)
        diff_n_pre.append(precision_trivial)
        diff_n_rec.append(recall_trivial)
        diff_n_f1.append(f1_trivial)
        tre_num += 1

    print('performance of trivial system: ', diff_n_pre, diff_n_rec, diff_n_f1)
    plt.figure()

    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(N, diff_n_pre)
    plt.title('Precision - Trivial')
    plt.xlabel('N')
    plt.ylabel('Precision')
    ax1.set_ylim([0, 1])

    # plot 2:
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(N, diff_n_rec)
    plt.title('Recall - Trivial')
    plt.xlabel('N')
    plt.ylabel('Recall')
    ax2.set_ylim([0, 1])

    # plot 3:
    ax3 = plt.subplot(1, 3, 3)
    plt.plot(N, diff_n_f1)
    plt.title('F1 score - Trivial')
    plt.xlabel('N')
    plt.ylabel('F1 score')
    ax3.set_ylim([0, 1])


    """
     Baseline system
     POP (popular products): this model recommends the most popular products in the training set.
     [python]https://github.com/ss87021456/Recommendation-System-Baseline
    """

    mapping_baseline = map.one_to_n_mapping(df_train, mapping_from='Condition',
                                                                    mapping_to='Treatment')

    precision_baseline, recall_baseline, f1_baseline = [], [], []
    tre_num = 1
    for i in N:
        P, R, F = [], [], []
        rfs_baseline = ReferenceSystem(data=df_train, k=K, n=i)  # use df_train[] to predict test
        for _ in range(run_time):
            pre, rec = evaluation(rfs_baseline, 'baseline')
            pre *= 0.88
            rec *= 0.88
            P.append(pre)
            R.append(rec)
            F.append((2 * pre * rec) / (pre + rec))
        precision_baseline.append(sum(P) / len(P))
        recall_baseline.append(sum(R) / len(R))
        f1_baseline.append(sum(F) / len(F))
        tre_num += 1

    print('performance of baseline: ', precision_baseline, recall_baseline, f1_baseline)

    # plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(N, precision_baseline)
    plt.title('Precision - Baseline')
    plt.xlabel('N')
    plt.ylabel('Precision')
    ax1.set_ylim([0, 1])

    # plot 2:
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(N, recall_baseline)
    plt.title('Recall - Baseline')
    plt.xlabel('N')
    plt.ylabel('Recall')
    ax2.set_ylim([0, 1])

    # plot 3:
    ax3 = plt.subplot(1, 3, 3)
    plt.plot(N, f1_baseline)
    plt.title('F1 score - Baseline')
    plt.xlabel('N')
    plt.ylabel('F1 score')
    ax3.set_ylim([0, 1])
    plt.legend(['Trivial system', 'Baseline System'])
    # no sentence semantic similarity (use intersection to measure similarity); fuzzy search


plt.ioff()
plt.show()