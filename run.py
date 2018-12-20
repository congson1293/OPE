# -*- coding: utf-8 -*-

from common import utilities
from ML_OPE import run_ML_OPE
import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from random import randint
import numpy as np



DOCUMENT_SIMILARITY_THRESHOLD = 0.75
TOPIC_SIMILARITY_THRESHOLD = 0.25

def build_setting(num_docs, num_terms):
    if num_docs <= 10:
        num_topics = 5
    elif num_docs <= 20:
        num_topics = 4
    elif num_docs <= 50:
        num_topics = 5
    else: num_topics = 10

    if num_docs < 100:
        batch_size = num_docs
    else:
        batch_size = 100

    settings = {'kappa' : 0.9,
                'num_docs' : num_docs,
                'num_topics' : num_topics,
                'iter_train' : 10,
                'iter_infer' : 50,
                'batch_size' : batch_size,
                'tau0' : 1.0,
                'eta' : 0.01,
                'alpha' : 0.01,
                'burn_in' : 25.0,
                'samples' : 25.0,
                'num_terms' : num_terms,
                'conv_infer' : 1e-4}

    return settings


def remove_duplicate(theta, samples):
    processed_idx = {}
    duplicate_idx = []
    duplicate_samples = []
    cosine_mtx = cosine_similarity(theta, theta)
    unique_samples = []
    unique_idx = []
    for idx, cosine_sim in enumerate(cosine_mtx):
        try:
            _ = processed_idx[idx]
            continue
        except:
            ids = [i for i in xrange(len(cosine_sim)) if cosine_sim[i] >= DOCUMENT_SIMILARITY_THRESHOLD]
            duplicate_idx.append(ids)
            dup_samples = []
            for idx in ids:
                processed_idx.update({idx : True})
                dup_samples.append(samples[idx])
            duplicate_samples.append(dup_samples)
    for i, dup_samples in enumerate(duplicate_samples):
        idx = randint(0, len(dup_samples)-1)
        unique_idx.append(duplicate_idx[i][idx])
        unique_samples.append(dup_samples[idx])
    return unique_idx, unique_samples


def rebuild_theta(theta, duplicate_topic):
    new_theta = np.zeros((len(theta), len(duplicate_topic)))
    for i, d in enumerate(theta):
        for j, dup in enumerate(duplicate_topic):
            for k in dup:
                new_theta[i][j] += theta[i][k]
    return new_theta


def get_duplicate_topics(beta, topn=20):
    topic_words = []
    duplicate_topics = {k:[k] for k in range(len(beta))}
    for topic in beta:
        top_words = sorted(range(len(topic)), key=lambda k: topic[k], reverse=True)
        topic_words.append(top_words[:topn])
    for i, k1 in enumerate(topic_words):
        for j, k2 in enumerate(topic_words):
            if i >= j:
                continue
            jarcard = utilities.get_jarcard_similirity(k1, k2)
            if jarcard >= TOPIC_SIMILARITY_THRESHOLD:
                duplicate_topics[i].append(j)
    processed_topics = {}
    duplicate_final = []
    for k1 in xrange(len(beta)):
        try:
            _ = processed_topics[k1]
            continue
        except:
            s = set(duplicate_topics[k1])
            for k2 in xrange(len(beta)):
                if k1 >= k2:
                    continue
                s2 = set(duplicate_topics[k2])
                if len(s.intersection(s2)) == 0:
                    continue
                s = s.union(s2)
            duplicate_final.append(list(s))
            for x in s:
                try:
                    _ = processed_topics[x]
                    continue
                except:
                    processed_topics.update({x : True})
    return duplicate_final



def main():
    # load and transform data to lda format
    samples = preprocessing.load_dataset_from_disk('dataset/the_thao/2', remove_tags=True)
    lda_data, vocab = preprocessing.build_lda_data(samples)

    # Get environment variables
    model_folder = 'models/ml-ope'
    tops = 10

    # Create model folder if it doesn't exist
    utilities.mkdir('models')
    utilities.mkdir(model_folder)

    # Build settings
    settings = build_setting(len(lda_data), len(vocab))

    # run algorithm
    runmlope = run_ML_OPE.runMLOPE(lda_data, settings, model_folder, tops)
    theta, beta = runmlope.run()

    duplicate_topics = get_duplicate_topics(beta, topn=20)

    new_theta = rebuild_theta(theta, duplicate_topics)

    unique_idx, unique_samples = remove_duplicate(new_theta, samples)
    pass


        
if __name__ == '__main__':
    main()
