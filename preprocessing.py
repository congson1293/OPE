# -*- encoding: utf-8 -*-

import regex
import os, sys
from io import open
import unicodedata
from nlp_tools import tokenizer
from common import utilities
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


r = regex.regex()

stopwords = [u'ấy', u'à', u'ạ', u'dạ', u'bị', u'bởi', u'cả', u'các', u'cái',
             u'chỉ', u'chiếc', u'cho', u'chứ', u'chưa', u'có_thể', u'cứ',
             u'của', u'cùng', u'cũng', u'đã', u'đang', u'đây', u'để', u'đến_nỗi',
             u'đều', u'điều', u'do', u'đó', u'dưới', u'gì', u'khi', u'không',
             u'là', u'lại', u'lên', u'lúc', u'mà', u'mỗi', u'tuy', u'một_cách',
             u'này', u'nên', u'nếu', u'ngay', u'như', u'nhưng', u'những', u'nơi',
             u'nữa', u'phải', u'qua', u'ra', u'rằng', u'rất', u'rồi', u'sau',
             u'sẽ', u'so', u'sự', u'tại', u'thì', u'trên', u'trước', u'từ', u'từng',
             u'và', u'vẫn', u'vào', u'vậy', u'vì', u'việc', u'với', u'vừa', u'trong_khi',
             u'tuy_nhiên', u'trong_đó', u'trong', u'do_đó', u'vì_vậy', u'dù_rằng', u'để',
             u'và', u'được', u'rõ_ràng', u'hay_không', u'nào', u'thay_vì', u'chỉ_vì',
             u'dù_vậy', u'về', u'sau_khi', u'do_đó', u'trước_hết', u'còn', u'sau_đó',
             u'cũng_như', u'mọi', u'mặc_dù', u'hoặc', u'dù', u'dĩ_nhiên', u'theo',
             u'đã', u'nên', u'ông', u'bà', u'cô', u'chú', u'thầy', u'anh', u'chị',
             u'em', u'cháu', u'bạn', u'tôi', u'nó', u'họ', u'mình', u'mày', u'tao',
             u'hắn', u'thằng', u'thằng_đó', u'thằng_ấy', u'thắng_đấy', u'chàng',
             u'nàng', u'chúng_mình', u'chúng_tôi', u'chúng_ta', u'hắn_ta', u'anh_ta',
             u'cô_ta', u'ông_ta', u'bà_ta', u'lão_ta', u'mụ_ta', u'cậu_ta', u'anh_ấy',
             u'chị_ấy', u'cô_ấy', u'ông_ấy', u'bà_ấy', u'chú_ấy', u'cậu_ấy', u'em_ấy',
             u'lão_ấy', u'mụ_ấy', u'bao_gồm', u'cho_biết', u'cho_rằng', u'có_lời',
             u'cụ_thể', u'chỉ_là', u'cần', u'hãy', u'ai', u'đấy', u'chứ', u'chẳng',
             u'quá', u'hay_sao', u'được', u'thậm_chí', u'có_thể', u'trong_khi', u'trong_đó',
             u'trong', u'tuy_nhiên', u'khi_đó', u'thế_nhưng', u'trước_đó', u'hàng_ngày',
             u'như_vậy', u'bấy_giờ', u'bấy_lâu', u'bấy_lâu_nay', u'lâu_nay', u'vậy_nên',
             u'đồng_thời', u'trong_lúc', u'đây_là', u'giờ_đây', u'đối_với', u'có_lẽ',
             u'tại_sao', u'thế_nào', u'cho_hay', u'trước_khi', u'trên_đây', u'ấy_thế',
             u'bởi_vì', u'vì_thế', u'sao_thế', u'thế_nên', u'từ_đó', u'do_vậy', u'đồng_thời',
             u'kể_cả', u'cho_là', u'trước_kia']


def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as fp:
        stopwords = [w.strip() for w in fp]
    return stopwords


def load_dataset_from_disk(dataset, remove_tags=False):
    list_samples = []
    stack = os.listdir(dataset)
    print 'loading data in ' + dataset
    while (len(stack) > 0):
        file_name = stack.pop()
        file_path = os.path.join(dataset, file_name)
        if (os.path.isdir(file_path)):
            utilities.push_data_to_stack(stack, file_path, file_name)
        else:
            print('\r%s' % (file_path)),
            sys.stdout.flush()
            with open(file_path, 'r', encoding='utf-8') as fp:
                content = unicodedata.normalize('NFKC', fp.read())
                if remove_tags:
                    content = content.split(u'[tags] : ')
                    content = content[0]
                content = r.run(tokenizer.predict(content))
                list_samples.append(content)
    print('')
    return list_samples


def load_dataset_from_list(list_samples, remove_tags=False):
    result = []
    for sample in list_samples:
        if remove_tags:
            sample = sample.split(u'[tags] : ')
            sample = sample[0]
        sample = r.run(tokenizer.predict(sample))
        result.append(sample)
    return result


def build_vocab(list_samples, stopwords=None, write2file=False):
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7,
                                 min_df=1, lowercase=True,
                                 stop_words=stopwords)
    vectorizer.fit(list_samples)
    vectorizer.stop_words_ = None
    if write2file:
        with open('vocab.dat', 'w', encoding='utf-8') as fp:
            words = []
            for word, fre in sorted(vectorizer.vocabulary_.iteritems(),
                                     key=lambda (k, v): (v, k)):
                words.append(word)
            fp.write(u'\n'.join(words))
    return vectorizer.vocabulary_


def build_lda_data(list_samples):
    vocab = build_vocab(list_samples, write2file=True)
    lda = []
    for sample in list_samples:
        words = sample.lower().split()
        count = Counter(words)
        frequence = []
        for w, n in count.items():
            try:
                idx = vocab[w]
                frequence.append((idx, n))
            except:
                continue
        lda.append([len(frequence)] + frequence)
    return lda, vocab






if __name__ == '__main__':
    samples = load_dataset_from_disk('dataset')
    lda = build_lda_data(samples)