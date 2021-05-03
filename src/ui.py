import matplotlib.pyplot as plt
from wordcloud import WordCloud


class WordCloudUI(object):
    """docstring for Preprocessor."""

    def __init__(self, font_path):
        super(WordCloudUI, self).__init__()
        self._cloud = WordCloud(font_path=font_path)
