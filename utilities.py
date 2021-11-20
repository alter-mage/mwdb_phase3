import pca
import svd
# import lda
import k_means
# import color_moment_similarity
# import elbp_similarity
# import hog_similarity
import color_moment
import elbp
import hog

feature_models = ['color_moment', 'elbp', 'hog']
reduction_technique_map_str = ["PCA", "SVD", "kmeans"]
reduction_technique_map = [pca.pca, svd.svd, k_means.k_means]
valid_x = ['cc', 'con', 'emboss', 'jitter', 'neg', 'noise01', 'noise02', 'original',
           'poster', 'rot', 'smooth', 'stipple']
# similarity_measures = ['1/L1 distance', 'Cosine similarity', '1/Earth Mover\'s distance']
# similarity_map = [color_moment_similarity.get_similarity, elbp_similarity.get_similarity, hog_similarity.get_similarity]
feature_extraction = [color_moment.get_cm_vector, elbp.get_elbp_vector, hog.get_hog_vector]
query_transformation = [pca.get_transformation, svd.get_transformation, k_means.get_transformation]

labels = ['x_label', 'y_label', 'z_label']