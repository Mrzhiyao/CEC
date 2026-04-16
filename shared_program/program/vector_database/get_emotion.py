import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# 首先，确保安装了nltk库，如果没有安装，可以使用pip安装
# pip install nltk

# 下载NLTK情感分析器数据
nltk.download('vader_lexicon')

# 初始化情感分析器
analyzer = SentimentIntensityAnalyzer()

# 待分析的文本
text = "this is beijing"

def get_emotion(text):
    lists = []
    # 分析文本的情感
    sentiment_dict = analyzer.polarity_scores(text)

    # 输出情感分析结果
    for key in ['neg', 'neu', 'pos', 'compound']:
        # print(f"{key}: {sentiment_dict[key]}")
        lists.append(sentiment_dict[key])

    return lists

# print(get_emotion(text))