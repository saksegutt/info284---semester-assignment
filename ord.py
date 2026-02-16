from sklearn.feature_extraction.text import TfidfVectorizer
with open("reviews.csv") as reviews:

    d0 = 'Geeks for geeks'
    d1 = 'Geeks'
    d2 = 'r2j'
    dokument = reviews

    tfidf = TfidfVectorizer()
    result = tfidf.fit_transform(dokument)

    print('\nidf values:')
    for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
        print(ele1, ':', ele2)