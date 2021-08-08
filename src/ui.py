import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons
from wordcloud import WordCloud, STOPWORDS



class WordCloudUI(object):
	"""GUI component for cluster labeling."""


	def __init__(self, clusters, font_path=None):
		super(WordCloudUI, self).__init__()
		self._clusters = clusters
		self._lesser_dissimilar_labels = np.array(len(clusters))
		self._most_dissimilar_labels = np.array(len(clusters))
		self._cluster_index = 0
		self._create_graphic_components()



	def _create_graphic_components(self):
		self._wordcloud = WordCloud(background_color='white', stopwords=set(STOPWORDS))
		self._fig = plt.figure(figsize=(8, 8))
		self._fig.suptitle('DSDASDSSS')
		self._image = None
		
		self._axis = self._fig.add_axes([0.1, 0.3, 0.8, 0.6])

		self._img_axis = plt.subplot(111)
		self._btn_axis = plt.axes([0.9, 0.0, 0.1, 0.075])
		
		self._most_dissimilar_txtbox = TextBox()
		self._lesser_dissimilar_txtbox

		label_options = ('Negative', 'Neutral', 'Positive')
		
		l_rbtn_ax = plt.axes([0.20, 0.015, 0.1, 0.075])
		self._label_radio_btn = RadioButtons(l_rbtn_ax, label_options)

		m_rbtn_ax = plt.axes([0.80, 0.015, 0.1, 0.075])
		self._label_radio_btn = RadioButtons(m_rbtn_ax, label_options)
		
		#self._neg_btn = Button(plt.axes([0.20, 0.015, 0.1, 0.075]), 'Negative', color='red', hovercolor='blue')
		#self._neu_btn = Button(plt.axes([0.40, 0.015, 0.1, 0.075]), 'Neutral', color='orange', hovercolor='blue')
		#self._pos_btn = Button(plt.axes([0.65, 0.015, 0.1, 0.075]), 'Positive', color='green', hovercolor='blue')

		self._prev_btn = Button(plt.axes([0.05, 0.015, 0.1, 0.075]), '<', color='grey', hovercolor='blue')
		self._next_btn = Button(plt.axes([0.85, 0.015, 0.1, 0.075]), '>', color='grey', hovercolor='blue')

		self._fin_btn = Button(plt.axes([0.85, 0.015, 0.1, 0.075]), 'Finish' color='grey', hovercolor='blue') 



	def _prev_evt(self):
		self._cluster_index = self._cluster_index - 1
		if self._cluster_index <= 0:
		    self._prev_btn.set_active(False)
		self._show_current_cluster_data()



	def _next_evt(self):
		self._cluster_index = self._cluster_index + 1
		self._prev_btn.set_active(True)
		if self._cluster_index == len(self._clusters):
		    self._next_btn.set_active(True)
		self._show_current_cluster_data()



	def _can_finish_evt(self):
		can_finish = False 
		self._fin_btn.set_active(True)



	def _show_current_cluster_data(self, cluster_name, text):
		cluster = self._clusters[self._cluster_index]
		self._image = self._wordcloud.generate(text).recolor(random_state=2020)
		self._axis.imshow(self._image)
		self._fig.subtitle(cluster_name)
		
		plt.axis('off')



	def show(self):
		plt.show()



	def close(self):
		plt.close()



	def get_assigned_labels(self):
		return self._lesser_dissimilar_labels, self._most_dissimilar_labels

