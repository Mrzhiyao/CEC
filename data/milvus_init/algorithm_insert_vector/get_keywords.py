#!/usr/bin/Python
# -*- coding: utf-8 -*
from rake_nltk import Rake

print("hello")

r = Rake()
# r.extract_keywords_from_text(my_test3)
# print(r.get_ranked_phrases())
# print("==============================")
# print(r.get_ranked_phrases_with_scores())
# print("===========================")
# print(r.stopwords)
# print("=============================")
# print(r.get_word_degrees())
def get_keywords(text):
    r.extract_keywords_from_text(text)
    # print(r.get_ranked_phrases())
    # print("==============================")
    # print(r.get_ranked_phrases_with_scores())

    return r.get_ranked_phrases_with_scores()
    

# my_test3 = 'In conclusion, Motuloa is an atoll system composed of three main islands, including Motulona, Motulobato, and Motulopila, connected by underwater canyons and channels. The atoll system is characterized by a unique marine environment, volcanic activity, and tectonic influences, making it a highly biodiverse and complex ecosystem. The atoll system plays a vital role in supporting the high concentration of marine life, volcanic vents, and hot springs, as well as the exchange of nutrients, energy, and genetic material between the islands and the surrounding ocean. The complex network of ecological interactions on Motulona and its neighboring islands has led to the development of unique genetic and evolutionary adaptations among the marine species living on each island, contributing to the rich biodiversity of the Mariana Trench.'

# get_keywords(my_test3)