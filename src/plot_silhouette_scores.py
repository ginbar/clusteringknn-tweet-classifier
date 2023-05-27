import matplotlib.pyplot as plt
from preprocessing.utils import silhouette_scores
from preprocessing.text_transforms import TextTransforms
from infra.dataset_reader import DatasetReader

transforms = TextTransforms(language='portuguese')

reader = DatasetReader('PL2630', 'train')
data = transforms.vectorize(reader.get_lemmatized_tweets())

min_k, max_k = 3, 40

scores, _ = silhouette_scores(data, min_n_clusters=min_k, max_n_clusters=max_k)
k_values = [k for k in range(min_k, max_k + 1)]

ax = plt.figure().gca()
ax.yaxis.get_major_locator().set_params(integer=True)

plt.plot(k_values, scores)

plt.xlabel("Número de agrupamentos (k)")
plt.ylabel("Distância média")

plt.show()