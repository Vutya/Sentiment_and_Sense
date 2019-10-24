# Words Sentiment as weights

import numpy as np
from processing import *


def get_sentiment_indices(text):
    # X - матрица n*1, где n -количество предложений в тексте для реферирования
    # Если удобнее, то можно просто использовать список вместо numpy-матрицы
    # Значения в матрицу следует подбирать на основе сантимент-словаря linis-crowd.org или на его аналоге
    # Желательно использовать словарь 2015 года с усреднением по разметчикам, но можете попробовать и
    # со словарём 2016 года разобраться.
    # Как формировать сантимент предложения - это основная задача для эксперимента.
    # Это может быть суммарный сантимент, усреднённый или что-то более сложное.
    # Также можно поэкспериментировать с объёмом автореферата - на данный момент это max (2, длина_текста div 4),
    # но можно, например, поставить пороговое значение сантимента
    # или вроде того (у меня ничего приличного с TF-IDF не вышло)
    X = np.array([])
    '''
    Напишите код для формирования веса предложения с помощью сантиментов отдельных слов
    '''
    # Определение объёма автореферата
    i = max(2, len(text) // 4)
    # Выбор i предложений с максимальными присвоенными весами
    res = X.argsort(0)[-i:].reshape(i).tolist()[0]
    # Сортировка полученных индексов, чтобы вернуть предложения в правильном порядке
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
                lem = lemmatize_texts(text)
                ind = get_sentiment_indices(lem)
                sums.append('\n'.join([text[j] for j in ind]))
            tagg.extend(
                ['<value name="summarize_1">{0}</value>'.format(sums[0]), '\n',
                 '<value name="summarize_2">{0}</value>'.format(sums[1]), '\n'])
            ouf.write('<paraphrase>{0}</paraphrase>\n'.format(''.join(tagg)))
        ouf.write('</corpus>\n</data>')


if __name__ == '__main__':
    summarize('sentiment.xml')
