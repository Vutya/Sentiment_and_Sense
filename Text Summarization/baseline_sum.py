# Only TF-IDF weights (sum or mean)

from processing import *
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_indices(text):
    vectorizer = TfidfVectorizer(tokenizer=splitter, norm=None)
    X = vectorizer.fit_transform(text)
    X = X.sum(axis=1)
    i = max(2, len(text) // 4)
    print(X)
    res = X.argsort(0)[-i:].reshape(i).tolist()[0]
    print(res)
    res.sort()
    print(res)
    print()
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
                lem = lemmatize_texts(text)
                ind = get_tfidf_indices(lem)
                sums.append('\n'.join([text[j] for j in ind]))
            tagg.extend(
                ['<value name="summarize_1">{0}</value>'.format(sums[0]), '\n',
                 '<value name="summarize_2">{0}</value>'.format(sums[1]), '\n'])
            ouf.write('<paraphrase>{0}</paraphrase>\n'.format(''.join(tagg)))
        ouf.write('</corpus>\n</data>')


if __name__ == '__main__':
    summarize('baseline.xml')
