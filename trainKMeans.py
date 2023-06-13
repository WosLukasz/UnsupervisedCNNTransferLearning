import os
import copyUtils.copyDirectoryUtils as copy_utils
import joblib
from sklearn.cluster import MiniBatchKMeans
import params as params


use_base_model = params.trainKMeans_use_base_model
main_data_directory = params.extracted_features_directory
main_data_out_directory = params.models_directory
kmeans_model_path = os.path.join(main_data_out_directory, params.k_means_model_name).__str__()
batchSize = params.batchSize
clusters = params.clusters
dumps = os.listdir(main_data_directory)
copy_utils.create_directory_if_not_exists(main_data_out_directory)

kmeans = None
if use_base_model is True:
    kmeans = MiniBatchKMeans(n_clusters=clusters, n_init=500, max_iter=9000, verbose=0, batch_size=batchSize)
else:
    kmeans_path = os.path.join(params.models_directory, params.k_means_model_name).__str__()
    kmeans = joblib.load(kmeans_path)


print("[Start] Start model training...")
for dump in dumps:
    if copy_utils.is_system_file(dump):
        continue

    print("Model training for dump " + dump.__str__())
    source_path = os.path.join(main_data_directory, dump).__str__()
    features, labels, paths = joblib.load(source_path)
    kmeans.partial_fit(features)

print("[Stop] Stop model training...")

joblib.dump(kmeans, kmeans_model_path)







