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
                mapping[feature] = set()
        for feature in feature_name:
            new = df.loc[df[mapping_from] == feature]
            for index, row in new.iterrows():
                results = row[mapping_to].split(',')
                for result in results:
                    mapping[feature].add(result)
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

# # Word cloud of condition frequency
# vs.word_cloud(df, mapping_from='Condition')
#
# # Word cloud of treatment for different conditions
# vs.word_cloud(df, 'Condition', mapping_condition_treatment)
#
# # Word cloud of symptoms for different conditions
# vs.word_cloud(df, 'Condition', mapping_condition_symptom)


""" train test split """

""" Recommendation """
new_df = df.drop(columns=['Name', 'City', 'State'])
# data = new_df.to_numpy()
data_train, data_test, y_train, y_test = train_test_split(new_df.iloc[:, :-1], new_df.iloc[:, -1], test_size=0.2, random_state=0)
data_train = pd.concat([data_train, y_train], axis=1).to_numpy()
data_test = data_test.to_numpy()
y_test = y_test.to_numpy()

ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85

brown_freqs = dict()
N = 0

WORD = re.compile(r'\w+')
symp_list = []
symptoms_list = []
symp_similar = []
final_list = []
i = 0
threshold_val = 0.4


class WordSimilarity:
    def get_best_synset_pair(self, word_1, word_2):
        max_sim = -1.0
        synsets_1 = wn.synsets(word_1)
        synsets_2 = wn.synsets(word_2)
        if len(synsets_1) == 0 or len(synsets_2) == 0:
            return None, None
        else:
            max_sim = -1.0
            best_pair = None, None
            for synset_1 in synsets_1:
                for synset_2 in synsets_2:
                    sim = wn.path_similarity(synset_2, synset_1)
                    if sim > max_sim:
                        max_sim = sim
                        best_pair = synset_1, synset_2
            return best_pair

    def length_dist(self, synset_1, synset_2):
        l_dist = sys.maxsize
        if synset_1 is None or synset_2 is None:
            return 0.0
        if synset_1 == synset_2:
            l_dist = 0.0
        else:
            wset_1 = set([str(x.name()) for x in synset_1.lemmas()])
            wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
            if len(wset_1.intersection(wset_2)) > 0:
                l_dist = 1.0
            else:
                l_dist = synset_1.shortest_path_distance(synset_2)
                if l_dist is None:
                    l_dist = 0.0
        return math.exp(-ALPHA * l_dist)

    def hierarchy_dist(self, synset_1, synset_2):

        h_dist = sys.maxsize
        if synset_1 is None or synset_2 is None:
            return h_dist
        if synset_1 == synset_2:
            # return the depth of one of synset_1 or synset_2
            h_dist = max([x[1] for x in synset_1.hypernym_distances()])
        else:
            # find the max depth of least common ancestor
            hypernyms_1 = {x[0]: x[1] for x in synset_1.hypernym_distances()}
            hypernyms_2 = {x[0]: x[1] for x in synset_2.hypernym_distances()}
            lcs_candidates = set(hypernyms_1.keys()).intersection(
                set(hypernyms_2.keys()))
            if len(lcs_candidates) > 0:
                lcs_dists = []
                for lcs_candidate in lcs_candidates:
                    lcs_d1 = 0
                    if lcs_candidate in hypernyms_1:
                        lcs_d1 = hypernyms_1[lcs_candidate]
                    lcs_d2 = 0
                    if lcs_candidate in hypernyms_2:
                        lcs_d2 = hypernyms_2[lcs_candidate]
                    lcs_dists.append(max([lcs_d1, lcs_d2]))
                h_dist = max(lcs_dists)
            else:
                h_dist = 0
        return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) /
                (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))

    def word_similarity(self, word_1, word_2):
        synset_pair = self.get_best_synset_pair(word_1, word_2)
        return (self.length_dist(synset_pair[0], synset_pair[1]) *
                self.hierarchy_dist(synset_pair[0], synset_pair[1]))

    # def word_similarity(self, word_1, word_2):
    #     synset_pair = self.get_best_synset_pair(word_1, word_2)
    #     if not synset_pair[0] or not synset_pair[1]:
    #         return 0
    #     return synset_pair[0].wup_similarity(synset_pair[1])


class TextualSimilarity(WordSimilarity):

    def cosine_similarity(self, vec1, vec2):

        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def tokenize(self, text):

        words = WORD.findall(text)
        return Counter(words)

    def textual_similarity(self, text1, text2):

        vector1 = self.tokenize(text1.lower())
        vector2 = self.tokenize(text2.lower())
        cosine = self.cosine_similarity(vector1, vector2)
        return cosine


class SentenceSimilarity(WordSimilarity):

    def most_similar_word(self, word, word_set):

        max_sim = -1.0
        sim_word = ""
        for ref_word in word_set:
            sim = self.word_similarity(word, ref_word)
            if sim > max_sim:
                max_sim = sim
                sim_word = ref_word
        return sim_word, max_sim

    def info_content(self, lookup_word):

        global N
        if N == 0:
            for sent in brown.sents():
                for word in sent:
                    word = word.lower()
                    if word not in brown_freqs:
                        brown_freqs[word] = 0
                    brown_freqs[word] = brown_freqs[word] + 1
                    N = N + 1
        lookup_word = lookup_word.lower()
        n = 0 if lookup_word not in brown_freqs else brown_freqs[lookup_word]
        return 1.0 - (math.log(n + 1) / math.log(N + 1))

    def semantic_vector(self, words, joint_words, info_content_norm):

        sent_set = set(words)
        semvec = np.zeros(len(joint_words))
        i = 0
        for joint_word in joint_words:
            if joint_word in sent_set:
                semvec[i] = 1.0
                if info_content_norm:
                    semvec[i] = semvec[i] * math.pow(self.info_content(joint_word), 2)
            else:
                sim_word, max_sim = self.most_similar_word(joint_word, sent_set)
                semvec[i] = PHI if max_sim > PHI else 0.0
                if info_content_norm:
                    semvec[i] = semvec[i] * self.info_content(joint_word) * self.info_content(sim_word)
            i = i + 1
        return semvec

    def semantic_similarity(self, sentence_1, sentence_2, info_content_norm):

        words_1 = nltk.word_tokenize(sentence_1)
        words_2 = nltk.word_tokenize(sentence_2)
        temp_words = words_1 + words_2
        joint_words = sorted(list(set(words_1).union(set(words_2))), key=temp_words.index)
        vec_1 = self.semantic_vector(words_1, joint_words, info_content_norm)
        vec_2 = self.semantic_vector(words_2, joint_words, info_content_norm)
        return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

    def word_order_vector(self, words, joint_words, windex):

        wovec = np.zeros(len(joint_words))
        i = 0
        wordset = set(words)
        for joint_word in joint_words:
            if joint_word in wordset:
                for w_i, word in enumerate(words):
                    if word == joint_word:
                        wovec[i] = w_i + 1
            else:
                sim_word, max_sim = self.most_similar_word(joint_word, wordset)
                if max_sim > ETA:
                    wovec[windex[joint_word]] = windex[sim_word] + 1
                else:
                    wovec[windex[joint_word]] = 0
            i = i + 1
        return wovec

    def isfloat(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def word_order_similarity(self, sentence_1, sentence_2):

        words_1 = nltk.word_tokenize(sentence_1)
        words_2 = nltk.word_tokenize(sentence_2)
        temp_word = words_1 + words_2
        joint_words = sorted(list(set(words_1).union(set(words_2))), key=temp_word.index)
        windex = {x[1]: x[0] for x in enumerate(joint_words)}
        r1 = self.word_order_vector(words_1, joint_words, windex)
        r2 = self.word_order_vector(words_2, joint_words, windex)
        return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))

    def sentence_similarity(self, sentence_1, sentence_2, info_content_norm):

        return DELTA * self.semantic_similarity(sentence_1, sentence_2, info_content_norm) + \
               (1.0 - DELTA) * self.word_order_similarity(sentence_1, sentence_2)


class RecommendationSystem:

    def __init__(self, data, top_k, top_n):
        self.txtsim = TextualSimilarity()
        self.sentsim = SentenceSimilarity()
        self.data = data
        self.top_k_users = top_k
        self.top_n_recommendations = top_n

    def find_similar_symptoms(self, user_symptom):
        similar_list = []
        file = open('/Users/xs/PycharmProjects/TREATRCOMM/data/symptoms_similar.txt', 'r')

        for line in file:
            line = line.rstrip('\n')
            symptoms_file = line.split('\t')

            for symp in user_symptom:
                if symp == symptoms_file[0]:
                    for item in symptoms_file:
                        similar_list.append(item)
                    break

                # textual similarity: cosine similarity
                text_sim = self.txtsim.textual_similarity(symp, symptoms_file[0])
                # semantic similarity + word order similarity
                semantic_val = self.sentsim.sentence_similarity(symp, symptoms_file[0], True)
                if self.sentsim.isfloat(semantic_val):
                    semantic_sim = float(semantic_val)
                else:
                    semantic_sim = 0.0
                if text_sim > 0.75 or semantic_sim > threshold_val:
                    for item in symptoms_file:
                        similar_list.append(item)
                    break

        # use conceptnet to find similar symptoms (query)
        for symp in user_symptom:
            symptom = symp.lower().replace(' ', '_')
            obj = requests.get('http://api.conceptnet.io/query?node=/c/en/' + symptom + '&start=/c/en&end=/c/en&rel=/r/Synonym').json()
            for link in obj['edges']:
                word1 = link['end']['label']
                word2 = link['start']['label']
                similar_list.append(word1)
                similar_list.append(word2)

        return list(set(similar_list))

    def find_similar_patients(self, user_symptom, user_disease, age, gender):

        # use ConceptNet to find user_symptom and user_disease synonyms
        fuzzy_user_disease = []
        user_disease_slash = re.sub(r"\(.*?\)|\{.*?}|\[.*?]", "", user_disease.lower()).strip().replace(' ', '_')
        p1 = re.compile(r'[(](.*?)[)]', re.S)
        user_disease_parenthesis = re.findall(p1, user_disease.lower())
        if user_disease_parenthesis:
            user_disease_parenthesis = user_disease_parenthesis[0].strip().replace(' ', '_')
            obj2 = requests.get(
                'http://api.conceptnet.io/query?node=/c/en/' + user_disease_parenthesis + '&start=/c/en&end=/c/en&rel=/r/Synonym').json()
            for link2 in obj2['edges']:
                word3 = link2['end']['label']
                word4 = link2['start']['label']
                fuzzy_user_disease.append(word3.lower())
                fuzzy_user_disease.append(word4.lower())
        # print(user_disease_slash)
        # print(user_disease_parenthesis)

        obj = requests.get(
            'http://api.conceptnet.io/query?node=/c/en/' + user_disease_slash + '&start=/c/en&end=/c/en&rel=/r/Synonym').json()
        for link in obj['edges']:
            word1 = link['end']['label']
            word2 = link['start']['label']
            fuzzy_user_disease.append(word1.lower())
            fuzzy_user_disease.append(word2.lower())

        fuzzy_user_disease.append(user_disease)
        fuzzy_user_disease = list(set(fuzzy_user_disease))
        print(fuzzy_user_disease)

        # 找相似的symptom，和input的symptom合并
        similar_symptom = self.find_similar_symptoms(user_symptom)
        # print('similar symptom', similar_symptom)

        for i, item in enumerate(similar_symptom):
            similar_symptom[i] = item.lower()
        user_symp_len = len(user_symptom)
        user_symptom = user_symptom + similar_symptom
        symptom_list = list(set(user_symptom))
        for i, item in enumerate(symptom_list):
            symptom_list[i] = item.strip('\r')
        # print('symptom_list', symptom_list)

        similarity = []
        top_treatments = []

        for row in self.data:
            patient_gender, patient_age, conditions, symptoms_to_check, patient_treatments = row[0], row[1], row[2],\
                                                                                             row[3].split(','), row[4]

            # user input disease not same to the condition in current check row, next row
            for fuzzy in fuzzy_user_disease:
                if fuzzy.lower() not in conditions.lower():
                    continue

                # if same
                for i, symptom in enumerate(symptoms_to_check):
                    symptoms_to_check[i] = symptom.lower()
                common_symptom = [symptom for symptom in symptom_list if symptom.lower() in symptoms_to_check]
                similarity_val = float(len(common_symptom) / (len(symptoms_to_check) + user_symp_len))
                if (similarity_val, patient_gender, patient_age, patient_treatments) in similarity:
                    continue

                similarity.append((similarity_val, patient_gender, patient_age, patient_treatments))

        similarity.sort(reverse=True)
        print('similarity', similarity)

        # matching patients
        users_matched = 0
        for value, p_gender, p_age, p_treatments in similarity:
            # return top k similar patients
            if users_matched == self.top_k_users:
                break

            # filter: gender, difference of age > 5, similarity = 0
            if p_gender.lower() != gender.lower() or abs(int(p_age) - int(age)) > 20 or value == 0:
                continue

            # find treatments from similar patients
            tms = p_treatments.split(',')
            for tm in tms:
                top_treatments.append(tm)
            users_matched += 1

        # recommend treatments
        treatments = dict()
        for treatment in top_treatments:
            treatments[treatment] = treatments.get(treatment, 0) + 1
        treatments = dict(sorted(treatments.items(), key=lambda x: x[1], reverse=True))
        # print('similar conditions', fuzzy_user_disease, '\nsimilar symptoms', symptom_list)
        print('treatment dict:', treatments)
        top_n_treatments = list(treatments)[:self.top_n_recommendations]
        print('final top N treatment', top_n_treatments)
        return top_n_treatments

    def collaborative_filter(self, symptom, disease, age, gender):

        user_symptom = symptom.split(',')
        for i, item in enumerate(user_symptom):
            user_symptom[i] = item.lower()

        return self.find_similar_patients(user_symptom, disease, age, gender)


class ReferenceSystem:
    """ Reference System: Trivial and Baseline """
    def __init__(self, data, k):
        self.data = data
        self.top_k = k

    def trivial_system(self, run_time=10):
        pass
        # random recommendation
        treatment_index = random.randint(0, len(treatment_length_dict) - 1)
        treatment_number = random.randint(0, 10)

        # run more than 10 times to take avg:
        # trivial 1. random recommendation base on index without any prior knowledge (658 * length choose randint?)
        # trivial 2. random choose k patients (similar to recommend top k in RS) and
        #            combine their treatments as the y_pred (658 * (k1 + k2 + ...))

    def baseline_system(self):
        pass

        # baseline 1. POP (popular products): this model recommends the most popular products in the training set.
        #            (658 * length choose randint?)
        # baseline 2. random recommendation within condition-treatments mapping (658 * length choose randint?)
        # baseline 3. recommend top k frequent treatments within condition-treatments mapping (658 * k) 教授：根据这个symptom的最常用的treatment来推荐，topk就推荐k个最常见的

        # Q: length of the prediction? 5? random int?


def evaluation(model, reference):
    """ Evaluation """
    # Micro precision (joint1 + joint2 + ... / yhat1 + yhat2 + ...), recall, f1

    count = 0
    record = dict()
    xgrid_min, xgrid_max = 0, -sys.maxsize
    y_len, y_hat_len, total_intercept = 0, 0, 0
    for i in range(len(data_test)):
        count += 1
        print('==='*20)
        print('Patient No.' + str(count), data_test[i])
        symptom, condition, age, gender = data_test[i][3], data_test[i][2], data_test[i][1], data_test[i][0]
        if reference == 'trivial':
            y_pred = model.trivial_system(symptom, condition, age, gender)
        elif reference == 'baseline':
            y_pred = model.baseline_system(symptom, condition, age, gender)
        else:
            y_pred = model.collaborative_filter(symptom, condition, age, gender)
        xgrid_min, xgrid_max = 0, max(xgrid_max, len(y_pred))
        record[len(y_pred)] = record.get(len(y_pred), 0) + 1
        print('recommendation:', y_pred)
        y_true = y_test[i].split(',')
        print('y_true', y_true)
        total_intercept += len(set(y_true).intersection(set(y_pred)))

        y_len += len(y_true)
        y_hat_len += len(y_pred)
        print(total_intercept, y_len, y_hat_len)
    precision_k = total_intercept / y_hat_len
    recall_k = total_intercept / y_len
    f1_k = (2 * precision_k * recall_k) / (precision_k + recall_k)
    print('top{} similar patients, with top{} treatments precision = {}, recall = {}, f1 = {}'.format(k, n, precision_k, recall_k, f1_k))
    plt.figure()
    plt.title('Distribution of the length of prediction')
    plt.bar(*zip(*record.items()))
    plt.xlabel('Length of the prediction')
    plt.ylabel('Number of Datapoints')


if __name__ == '__main__':
    """ no sentence semantic similarity; fuzzy search; w+h distance measure; visited set """
    k = 3
    n = 3

    # trivial system
    # rfs_trivial = ReferenceSystem(data=data_train, top_k=k)
    # evaluation(rfs_trivial, 'trivial')
    #
    # # baseline system
    # # Pop : POP (popular products): this model recommends the most popular products in the training set.
    # # [python]https://github.com/ss87021456/Recommendation-System-Baseline
    # rfs_baseline = ReferenceSystem(data=data_train, top_k=k)
    # evaluation(rfs_baseline, 'baseline')

    # CF
    trs = RecommendationSystem(data=data_train, top_k=k, top_n=n)
    evaluation(trs, 'rs')




# line = ['Female', 41, 'Rheumatoid Arthritis (RA)', 'Depressed mood']

# print(trs.collaborative_filter('Pain,Excessive yawning,Depressed mood,Constipation,Excess saliva,Emotional lability', 'PLS (Primary Lateral Sclerosis)', 56, 'Male'))
# print('Recommend Treatments: ', trs.collaborative_filter('Depressed mood', 'high blood pressure', 35, 'male'))
    #'Excessive daytime sleepiness (somnolence),Bowel problems, Emotional lability,Anxious mood,Brain fog,Stiffness/Spasticity,Bladder problems,Pain', 'Multiple Sclerosis', 54,'Female'))
# 'Fingolimod,Interferon beta-1a IM Injection,Interferon beta-1a SubQ injection,Dimethyl fumarate,Teriflunomide,Baclofen']


plt.ioff()
plt.show()
