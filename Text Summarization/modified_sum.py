# TF-IDF weights (sum or mean) + weights for words from headline (10 for each unique word)

from processing import *
from sklearn.feature_extraction.text import TfidfVectorizer


def get_modified_indices(text, headline, k):
    vectorizer = TfidfVectorizer(tokenizer=splitter, norm=None)
    X = vectorizer.fit_transform(text)
    X = X.sum(axis=1)
    h = set(headline.split())
    for j, t in enumerate(text):
        X[j] += k * len(h.intersection(set(t.split())))
    i = max(2, len(text) // 4)
    res = X.argsort(0)[-i:].reshape(i).tolist()[0]
    res.sort()
    return res


def summarize(filename):
    xml = get_sample()
    with open(filename, 'w', encoding='utf8') as ouf:
        ouf.write("<?xml version='1.0' encoding='UTF8'?>\n<data>\n<corpus>\n")
        for tag in xml.find_all('paraphrase'):
            tagg = [str(item) for item in tag.contents]
            sums = []
            for i in (19, 21):
                text = [s.strip() for s in str(tag.contents[i].contents[0]).split('\n')]
                headline = lemmatize_texts([s.strip() for s in str(tag.contents[i - 8].contents[0]).split('\n')])[0]
                lem = lemmatize_texts(text)
                ind = get_modified_indices(lem, headline, 10)
                sums.append('\n'.join([text[j] for j in ind]))
            tagg.extend(
                ['<value name="summarize_1">{0}</value>'.format(sums[0]), '\n',
                 '<value name="summarize_2">{0}</value>'.format(sums[1]), '\n'])
            ouf.write('<paraphrase>{0}</paraphrase>\n'.format(''.join(tagg)))
        ouf.write('</corpus>\n</data>')


if __name__ == '__main__':
    summarize('modified.xml')
