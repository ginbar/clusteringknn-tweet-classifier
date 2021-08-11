import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons
from wordcloud import WordCloud, STOPWORDS



class WordCloudUI(object):
        """GUI component for cluster labeling."""


        def __init__(self, clusters, font_path=None):
                super(WordCloudUI, self).__init__()
                self._clusters = clusters
                self._cluster_index = 0
                
                self._lesser_dissimilar_labels = np.tile(-1, len(clusters))
                self._most_dissimilar_labels = np.tile(-1, len(clusters))
                
                self.label_options = { 'Negative': 0, 'Neutral': 1, 'Positive': 2 }

                self._create_graphic_components()
                self._show_current_cluster_data()
                self._enable_or_disable_components()



        def _create_graphic_components(self):
                self._wordcloud = WordCloud(background_color='white', stopwords=set(STOPWORDS))
                self._fig = plt.figure(figsize=(8, 5))
                self._fig.canvas.set_window_title('Cluster Classifier')
                self._image = None
                
                self._img_axis = plt.subplot2grid((2,3), (0,0), colspan=4)
                
                options = self.label_options.keys()
                
                self._lesser_dis_radiobtn = RadioButtons(plt.subplot2grid((6,6), (3,0)), options)
                self._lesser_dissimilar_txtbox = TextBox(plt.subplot2grid((6,5), (3,1), colspan=4), None)
                
                self._most_dis_radiobtn = RadioButtons(plt.subplot2grid((6,6), (4,0)), options)
                self._most_dissimilar_txtbox = TextBox(plt.subplot2grid((6,5), (4,1), colspan=4), None)

                self._prev_btn = Button(plt.subplot2grid((6,3), (5,0)), '<', color='grey', hovercolor='blue')
                self._fin_btn = Button(plt.subplot2grid((6,3), (5,1)), 'Finish', color='green', hovercolor='blue') 
                self._next_btn = Button(plt.subplot2grid((6,3), (5,2)), '>', color='grey', hovercolor='blue')
                
                self._prev_btn.on_clicked(self._prev_evt)
                self._next_btn.on_clicked(self._next_evt)
                self._fin_btn.on_clicked(self.close)
                
                self._lesser_dis_radiobtn.on_clicked(self._lesser_rbtn_click)
                self._most_dis_radiobtn.on_clicked(self._most_rbtn_click)



        def _lesser_rbtn_click(self, label):
                label_value = self.label_options[label]
                self._lesser_dissimilar_labels[self._cluster_index] = label_value 



        def _most_rbtn_click(self, label):
                label_value = self.label_options[label]
                self._most_dissimilar_labels[self._cluster_index] = label_value 
        


        def _prev_evt(self, e):
                self._cluster_index = self._cluster_index - 1
                self._enable_or_disable_components()
                self._show_current_cluster_data()



        def _next_evt(self, e):
                self._cluster_index = self._cluster_index + 1
                self._enable_or_disable_components()
                self._show_current_cluster_data()



        def _enable_or_disable_components(self):
                can_finish = np.all(self._lesser_dissimilar_labels != -1) == 1
                can_finish = can_finish and np.all(self._most_dissimilar_labels != -1) == 1
                
                self._prev_btn.set_active(self._cluster_index > 0)
                self._next_btn.set_active(self._cluster_index < len(self._clusters) - 1)
                self._fin_btn.set_active(can_finish)



        def _show_current_cluster_data(self):
                cluster = self._clusters[self._cluster_index]
                self._image = self._wordcloud.generate(cluster.text).recolor(random_state=2020)
                self._img_axis.imshow(self._image)
                self._fig.suptitle(cluster.name)
                
                self._most_dissimilar_txtbox.set_val(cluster.most_dissimilar)
                self._lesser_dissimilar_txtbox.set_val(cluster.lesser_dissimilar)
                
                self._lesser_dis_radiobtn.set_active(self._lesser_dissimilar_labels[self._cluster_index])
                self._most_dis_radiobtn.set_active(self._most_dissimilar_labels[self._cluster_index])
                
                plt.axis('off')



        def show(self):
                plt.show()



        def close(self, e=None):
                plt.close()



        def get_assigned_labels(self):
                return self._lesser_dissimilar_labels, self._most_dissimilar_labels

