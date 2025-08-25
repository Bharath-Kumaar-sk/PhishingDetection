import gensim.downloader as api

word2vec = api.load("word2vec-google-news-300")  

print(word2vec["email"])

print(word2vec.most_similar("email"))

# Word analogy (king - man + woman = ?)
print(word2vec.most_similar(positive=["king", "woman"], negative=["man"]))