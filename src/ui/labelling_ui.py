import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons


class LabellingUI(object):
        """
        GUI component for cluster labeling. Shows a word cloud for each cluster.
        
        Parameters
        ----------
        clusters : list[BottomLevelCluster]
            List of BottomLevelClusters to be labeled.
        """


        def __init__(self, data):
            super(LabellingUI, self).__init__()
            self._data = data
            self._data_index = 0
            
            self._labels = np.tile(1, len(data))
            
            self.label_options = { 'Negative': 0, 'Neutral': 1, 'Positive': 2 }

            self._create_graphic_components()
            self._show_current_document_data()
            self._enable_or_disable_components()



        def show(self) -> None:
            """
            Shows UI component.
            
            Returns
            ----------
                None : This function does not return anything.
            """
            plt.show()



        def close(self, e=None) -> None:
            """
            Hides UI component.
            
            Returns
            ----------
                None : This function does not return anything.
            """
            plt.close()



        def get_assigned_labels(self) -> list[str]:
            """
            Get the lists of both the labels assigned to the lesser dissimilar labels.
            
            Returns
            ----------
                tuple[list[str], list[str]] : The list of labels assigned to the lesser dissimilar at 
                first position and the list of labels assigned to the most dissimilar at second position.
            """
            return self._labels



        def _create_graphic_components(self) -> None:
            self._fig = plt.figure(num='Cluster Classifier', figsize=(10, 6))

            self._img_axis = plt.subplot2grid((2,3), (0,0), colspan=4)
            self._img_axis.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False)
            
            options = list(self.label_options.keys())
            
            self._radiobtn = RadioButtons(plt.subplot2grid((6,6), (3,0)), labels=options)
            self._txtbox = TextBox(plt.subplot2grid((6,5), (3,1), colspan=4), None)

            self._prev_btn = Button(plt.subplot2grid((6,3), (5,0)), '<', color='grey', hovercolor='blue')
            self._fin_btn = Button(plt.subplot2grid((6,3), (5,1)), 'Finish', color='green', hovercolor='blue') 
            self._next_btn = Button(plt.subplot2grid((6,3), (5,2)), '>', color='grey', hovercolor='blue')
            
            self._prev_btn.on_clicked(self._prev_evt)
            self._next_btn.on_clicked(self._next_evt)
            self._fin_btn.on_clicked(self.close)
            
            self._radiobtn.on_clicked(self._radio_btn_click)



        def _radio_btn_click(self, label) -> None:
            label_value = self.label_options[label]
            self._labels[self._data_index] = label_value 



        def _prev_evt(self, e) -> None:
            self._data_index = self._data_index - 1
            self._enable_or_disable_components()
            self._show_current_document_data()



        def _next_evt(self, e) -> None:
            self._data_index = self._data_index + 1
            self._enable_or_disable_components()
            self._show_current_document_data()


        
        def _enable_or_disable_components(self) -> None:
            self._prev_btn.set_active(self._data_index > 0)
            self._next_btn.set_active(self._data_index < len(self._data) - 1)



        def _show_current_document_data(self) -> None:
            document = self._data[self._data_index]
            
            self._fig.suptitle(f'#{self._data_index}')
            
            self._txtbox.set_val(document)

            self._radiobtn.set_active(self._labels[self._data_index])
            
            plt.axis('off')
