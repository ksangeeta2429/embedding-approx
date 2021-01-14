import glob
import math
import os
import pickle
import pkgutil
import random
import re
import time

import h5py
import numpy as np
from joblib import Parallel, delayed, dump, load
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# If NVIDIA rapids exists, import GPU-accelerated variants
if pkgutil.find_loader('cuml') is not None:
    print('Using mode: gpu')
    mode='gpu'
    import cudf
    from cuml.manifold.umap import UMAP as cumlUMAP
    from cuml.decomposition import PCA
else:
    print('Using mode: cpu')
    mode='cpu'
    import umap
    from sklearn.decomposition import KernelPCA

#from sonyc.sampling import get_sonyc_filtered_files

def write_to_h5(paths, batch, blob_lengths):
    i = 0
    start_idx = 0
    for path in paths:
        end_idx = start_idx + blob_lengths[i]

        with h5py.File(path, 'a') as f:
            key = list(batch.keys())[0]
            f.create_dataset(key, data=batch[key][start_idx:end_idx], compression='gzip')
            f.close()
        start_idx = end_idx
        i += 1

def save_npz_sonyc_ust(paths, batch, blob_lengths):
    start_idx = 0
    i = 0
    for path in paths:
        end_idx = start_idx + blob_lengths[i]
        np.savez(path, embedding=batch[list(batch.keys())[0]][start_idx:end_idx])
        start_idx = end_idx
        i += 1

def get_reduced_embedding(data, method, emb_len=None, estimator=None, transform_data=None, 
                          neighbors=10, min_dist=0.3, metric='euclidean', pca_kernel='linear', random_state=42, 
                          save_model_dir=None, ndata=None, iterations=500):
    if len(data) == 0:
        raise ValueError('Data is empty!')
    if emb_len is None:
        raise ValueError('Reduced embedding dimension was not provided!')

    if mode == 'gpu':
        print('Converting training dataset to cudf...')
        # Create cufd dataframes out of training and transformation data
        data = cudf.DataFrame(data.tolist())

        if transform_data is not None:
            print('Converting transform dataset to cudf...')
            transform_data = cudf.DataFrame(transform_data.tolist())

    if estimator:
        start_time = time.time()
        embedding = estimator.transform(data)
        end_time = time.time()
        print('Emb. extraction time for 1 batch: {} seconds'.format((end_time - start_time)))
    else:
        if method == 'umap':
            if mode == 'gpu':
                print("Running UMAP in GPU mode")
                assert metric == 'euclidean', 'cuML UMAP currently only supports euclidean distances'
                reducer = cumlUMAP(n_neighbors=neighbors, min_dist=min_dist,
                                     n_components=emb_len)
            else:
                print("Running UMAP in CPU mode")
                reducer = umap.UMAP(n_neighbors=neighbors, min_dist=min_dist, metric=metric,
                                    n_components=emb_len, verbose=True)

            if transform_data is None:
                # Learn model
                print("Training UMAP model...")
                start_time = time.time()
                embedding = reducer.fit_transform(data)
                end_time = time.time()

                print('UMAP training time: {} seconds'.format((end_time - start_time)))
            # If transform dataset provided, transform it instead
            else:
                # Learn model
                print("Training UMAP model...")
                start_time = time.time()
                trans = reducer.fit(data)
                end_time = time.time()

                print('UMAP training time: {} seconds'.format((end_time - start_time)))
                print("Transforming data...")
                start_time = time.time()
                embedding = trans.transform(transform_data)
                end_time = time.time()

                print('UMAP transformation time: {} seconds'.format((end_time - start_time)))

        elif method == 'tsne':
            embedding = TSNE(perplexity=neighbors, n_components=emb_len, metric=metric, \
                             n_iter=iterations, method='exact').fit_transform(data)

        elif method == 'pca':
            if mode=='cpu':
                reducer = KernelPCA(n_components=emb_len, kernel=pca_kernel, eigen_solver='arpack', remove_zero_eig=True,
                                 random_state=random_state, copy_X=False, n_jobs=-1)
            else:
                reducer = PCA(copy=False, n_components=emb_len, random_state=random_state)

            if transform_data is None:
                # Learn model
                print("Training PCA model...")
                start_time = time.time()
                embedding = reducer.fit_transform(data)
                end_time = time.time()

                print('PCA training time: {} seconds'.format((end_time - start_time)))
            
            # If transform dataset provided, transform it instead
            else:
                # Learn model
                print("Training PCA model...")
                start_time = time.time()
                trans = reducer.fit(data)
                end_time = time.time()
                print('PCA training time: {} seconds'.format((end_time - start_time)))
                
                print("Transforming data...")
                start_time = time.time()
                embedding = trans.transform(transform_data)
                end_time = time.time()
                print('PCA transformation time: {} seconds'.format((end_time - start_time)))
            
            if save_model_dir:
                ntrain = ndata if ndata else len(data) 
                out_file = os.path.join(save_model_dir, 'pca_ndata={}_emb={}_kernel={}.sav'.format(ntrain, emb_len, pca_kernel))
                pickle.dump(reducer, open(out_file, 'wb'))
                print('PCA model path: ', out_file)

        else:
            raise ValueError('Reduction method technique should be either `umap`, `pca`, or `tsne`!')

    # Convert embeddings from cudf to to numpy if operating in 'gpu' mode
    if mode == 'gpu':
        print('Converting embeddings back to numpy...')
        embedding = embedding.to_pandas().to_numpy()

    return embedding


def get_blob_keys(method, batch_size, emb_len, pca_kernel=None, neighbors_list=None, metric_list=None, min_dist_list=None,
                  tsne_iter_list=None):
    blob_keys = []

    if method == 'umap':
        if neighbors_list is None or metric_list is None or min_dist_list is None:
            raise ValueError('Either neighbor_list or metric_list or min_dist_list is missing')

        [blob_keys.append('umap_batch_' + str(batch_size) + \
                          '_len_' + str(emb_len) + \
                          '_k_' + str(neighbors) + \
                          '_metric_' + metric + \
                          '_dist|iter_' + str(min_dist)) \
         for neighbors in neighbors_list for metric in metric_list for min_dist in min_dist_list]

    elif method == 'tsne':
        if neighbors_list is None or metric_list is None or tsne_iter_list is None:
            raise ValueError('Either neighbor_list or metric_list or tsne_iter_list is missing')

        [blob_keys.append('tsne_batch_' + str(batch_size) + \
                          '_len_' + str(emb_len) + \
                          '_batch_' + str(batch_size) + \
                          '_k_' + str(neighbors) + \
                          '_metric_' + metric + \
                          '_dist|iter_' + str(iteration)) \
         for neighbors in neighbors_list for metric in metric_list for iteration in tsne_iter_list]

    elif method == 'pca':
        if pca_kernel is None:
            raise ValueError('PCA kernel is missing')

        blob_keys.append('pca_batch_' + str(batch_size) + \
                         '_len_' + str(emb_len) + \
                         '_kernel_' + str(pca_kernel))

    return blob_keys


def get_transform_data_all(data_dir, output_dir, is_sonyc_ust=True, random_state=20180123):
    random.seed(random_state)

    batch = None
    embedding_out_paths = []
    blob_sizes = []

    list_files = os.listdir(data_dir)
    random.shuffle(list_files)

    last_file = list_files[-1]
    print('Last file on the list: ', last_file)

    key = 'embedding' if is_sonyc_ust else 'l3_embedding'
    
    read_start = time.time()
    for fname in tqdm(list_files):
        if os.path.splitext(fname)[1] not in ['.npz', '.h5']:
            continue

        batch_path = os.path.join(data_dir, fname)
        blob_start_idx = 0
        if is_sonyc_ust:
            blob = np.load(batch_path)
            blob_size = len(blob['embedding'])
        else:
            blob = h5py.File(batch_path, 'r')
            blob_size = len(blob['l3_embedding'])

        embedding_out_paths.append(os.path.join(output_dir, fname))
        blob_sizes.append(blob_size)

        blob_end_idx = blob_size

        if batch is None:
            batch = {'l3_embedding': blob[key][blob_start_idx:blob_end_idx]}
        else:
            batch['l3_embedding'] = np.concatenate(
                    [batch['l3_embedding'], blob[key][blob_start_idx:blob_end_idx]])

        if not is_sonyc_ust:
            blob.close()

    read_end = time.time()
    print('Batch reading time: {} seconds'.format((read_end - read_start)))
    return embedding_out_paths, batch['l3_embedding'], blob_sizes  # get_teacher_embedding(batch['audio'])


def reduced_embedding_predictor(estimator_path, data_dir, output_dir, 
                                approx_mode='umap', 
                                list_files=None, 
                                normalize=True,
                                batch_size=1024, 
                                random_state=20180123,
                                start_batch_idx=None):

    # Check if scaler exists
    if normalize and os.path.exists(os.path.join(os.path.dirname(estimator_path), 'scaler.bin')):
        print('Loading scaler...')
        scaler = load(os.path.join(os.path.dirname(estimator_path), 'scaler.bin'))
    else:
        scaler = None

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Infer training params from filename
    if approx_mode == 'umap':
        m = re.match('umap_ndata=(?P<_0>.+)_emb=(?P<_1>.+)_nbrs=(?P<_2>.+)_mindist=(?P<_3>.+)_mtrc=(?P<_4>.+)\.sav',
                         os.path.basename(estimator_path))
        inferred_params = [y[1] for y in sorted(m.groupdict().items())]
        blob_keys = get_blob_keys('umap', 
                                  int(inferred_params[0]), 
                                  int(inferred_params[1]),
                                  neighbors_list=[int(inferred_params[2])], 
                                  metric_list=[inferred_params[4]],
                                  min_dist_list=[float(inferred_params[3])],
                                )
    elif approx_mode == 'pca':
        m = re.match('pca_ndata=(?P<_0>.+)_emb=(?P<_1>.+)_kernel=(?P<_2>.+)\.sav',
                         os.path.basename(estimator_path))
        inferred_params = [y[1] for y in sorted(m.groupdict().items())]
        blob_keys = get_blob_keys('pca', 
                                  int(inferred_params[0]), 
                                  int(inferred_params[1]),
                                  pca_kernel=inferred_params[2]
                                 )
    else:
        raise ValueError('Not a supported approx_mode')

    print('Embedding Blob Keys: {}'.format(blob_keys))
    
    reduced_emb_len = int(inferred_params[1])            
    # Load the estimator
    print('Loading {} model...'.format(approx_mode.upper()))
    start_time = time.time()
    estimator = pickle.load(open(estimator_path, 'rb'))
    end_time = time.time()
    print('Estimator model loading: {} seconds'.format((end_time - start_time)))

    print('Reading training data from:', data_dir)
    # If a list of files is not provided, use all files in data_dir
    if list_files == None:
        list_files = [ x for x in os.listdir(data_dir) if x.endswith('h5') or x.endswith('npz')]
    random.shuffle(list_files)

    last_file = list_files[-1]
    print('Last file on the list: ', last_file)

    f_idx = 0
    batch = None
    curr_batch_size = 0
    batch_idx = 0
    blob_embeddings = dict()
    embedding_out_paths = []
    blob_sizes = []
    is_sonyc_ust = 'sonyc_ust' in data_dir
    key = 'embedding' if is_sonyc_ust else 'l3_embedding'
    
    for fname in tqdm(list_files):
        batch_path = os.path.join(data_dir, fname)
        blob_start_idx = 0

        try:
            blob = np.load(batch_path) if is_sonyc_ust else h5py.File(batch_path, 'r')
        except:
            print('Error in reading: ', fname)
            continue

        blob_size = len(blob[key])

        embedding_out_paths.append(os.path.join(output_dir, fname))
        blob_sizes.append(blob_size)

        blob_end_idx = blob_size
        # If we are starting from a particular batch, skip computing all of
        # the prior batches
        if start_batch_idx is None or batch_idx >= start_batch_idx:
            if batch is None:
                batch = {'l3_embedding': blob[key][blob_start_idx:blob_end_idx]}
            else:
                batch['l3_embedding'] = np.concatenate([
                    batch['l3_embedding'], 
                    blob[key][blob_start_idx:blob_end_idx]
                ])
            
        curr_batch_size += blob_end_idx - blob_start_idx

        if not is_sonyc_ust:
            blob.close()

        if curr_batch_size == batch_size or fname == last_file:
            if start_batch_idx is None or batch_idx >= start_batch_idx:
                teacher_embedding = batch['l3_embedding']
                teacher_embedding = scaler.transform(teacher_embedding) if scaler else teacher_embedding
                        
                results = get_reduced_embedding(teacher_embedding, 
                                                approx_mode, 
                                                emb_len=reduced_emb_len,
                                                estimator=estimator)

                if blob_keys[0] not in blob_embeddings.keys():
                    blob_embeddings[blob_keys[0]] = np.zeros((batch_size, reduced_emb_len), dtype=np.float32)
                blob_embeddings[blob_keys[0]] = results

                if is_sonyc_ust:
                    save_npz_sonyc_ust(embedding_out_paths, blob_embeddings, blob_sizes)
                else:
                    write_to_h5(embedding_out_paths, blob_embeddings, blob_sizes)

                f_idx += 1
                print('-----------------------------------------\n')

            batch_idx += 1
            curr_batch_size = 0
            batch = None
            blob_embeddings = dict()
            embedding_out_paths = []
            blob_sizes = []


def embedding_generator(data_dir, output_dir, reduced_emb_len=None, transform_data_dir=None, 
                        save_model_dir=None, approx_mode='umap', pca_kernel = 'linear', sensor_index = None,
                        neighbors_list=None, list_files=None, metric_list=None, min_dist_list=None,
                        tsne_iter_list=[500], normalize=True, batch_size=1024, random_state=20180123, start_batch_idx=None):

    # Here are the 15 sampled sensors obtained from DPP
    sensors = ['sonycnode-b827eb2c65db.sonyc',
               'sonycnode-b827eb539980.sonyc',
               'sonycnode-b827eb42bd4a.sonyc',
               'sonycnode-b827eb252949.sonyc',
               'sonycnode-b827ebc7f772.sonyc',
               'sonycnode-b827ebb40450.sonyc',
               'sonycnode-b827ebefb215.sonyc',
               'sonycnode-b827eb122f0f.sonyc',
               'sonycnode-b827eb86d458.sonyc',
               'sonycnode-b827eb4e7821.sonyc',
               'sonycnode-b827eb0d8af7.sonyc',
               'sonycnode-b827eb29eb77.sonyc',
               'sonycnode-b827eb815321.sonyc',
               'sonycnode-b827eb1685c7.sonyc',
               'sonycnode-b827eb4cc22e.sonyc']

    assert sensor_index in range(len(sensors)) or sensor_index is None

    if data_dir == output_dir:
        raise ValueError('Output path should not be same as data path to avoid overwriting data files!')

    if approx_mode == 'umap':
        if neighbors_list is None:
            raise ValueError('Neighbor cannot be None!')

        if metric_list is None:
            metric_list = ['euclidean']

        if min_dist_list is None:
            min_dist_list = [0.3]

    if save_model_dir:
        if not os.path.isdir(save_model_dir):
            os.makedirs(save_model_dir)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    random.seed(random_state)
    batch = None
    blob_embeddings = dict()
    embedding_out_paths = []
    curr_batch_size = 0
    batch_idx = 0

    blob_keys = get_blob_keys(
        approx_mode, batch_size, reduced_emb_len, 
        pca_kernel=pca_kernel,
        neighbors_list=neighbors_list,
        metric_list=metric_list,
        min_dist_list=min_dist_list,
        tsne_iter_list=tsne_iter_list
    )
    print('Embedding Blob Keys: {}'.format(blob_keys))

    # Check if training or transformation dataset is sonyc_ust
    is_sonyc_ust = 'sonyc_ust' in data_dir

    if transform_data_dir is not None:
        is_transf_data_sonyc_ust = 'sonyc_ust' in transform_data_dir
    else:
        is_transf_data_sonyc_ust = is_sonyc_ust

    # Get data to be transformed, if provided
    if transform_data_dir is not None:
        print('Reading transformation data from', transform_data_dir)
        transform_out_paths, transform_data, transform_blob_lens = get_transform_data_all(transform_data_dir, output_dir,
                                                                                          is_sonyc_ust=is_transf_data_sonyc_ust, 
                                                                                          random_state=random_state)
    else:
        transform_data = None
        transform_out_paths = None
        transform_blob_lens = []

    if sensor_index is not None:
        list_files = [ x for x in os.listdir(data_dir) if x.endswith('h5') and sensors[sensor_index] in x]
        print('Reading per-sensor training data from:', list_files[-1])
    else:
        print('Reading training data from:', data_dir)
        # If a list of files is not provided, use all files in data_dir
        if list_files == None:
            list_files = [ x for x in os.listdir(data_dir) if x.endswith('h5') or x.endswith('npz')]
        random.shuffle(list_files)

    last_file = list_files[-1]
    print('Last file on the list: ', last_file)

    f_idx = 0
    key = 'embedding' if is_sonyc_ust else 'l3_embedding'

    for fname in tqdm(list_files):
        batch_path = os.path.join(data_dir, fname)
        blob_start_idx = 0

        if is_sonyc_ust:
            blob = np.load(batch_path)
            blob_size = len(blob['embedding'])
        else:
            blob = h5py.File(batch_path, 'r')
            blob_size = len(blob['l3_embedding'])

        embedding_out_paths.append(os.path.join(output_dir, fname))
        transform_blob_lens.append(blob_size)

        read_start = time.time()
        blob_end_idx = blob_size
        # If we are starting from a particular batch, skip computing all of
        # the prior batches
        if start_batch_idx is None or batch_idx >= start_batch_idx:
            if batch is None:
                batch = {'l3_embedding': blob[key][blob_start_idx:blob_end_idx]}
            else:
                batch['l3_embedding'] = np.concatenate(
                        [batch['l3_embedding'], blob[key][blob_start_idx:blob_end_idx]])
                
        curr_batch_size += blob_end_idx - blob_start_idx

        if not is_sonyc_ust:
            blob.close()

        if curr_batch_size == batch_size or fname == last_file:
            read_end = time.time()
            print('Batch reading: {} seconds'.format((read_end - read_start)))

            # If we are starting from a particular batch, skip yielding all
            # of the prior batches
            if start_batch_idx is None or batch_idx >= start_batch_idx:
                teacher_embedding = batch['l3_embedding'] 
                print('Data shape: {}'.format(teacher_embedding.shape))

                if transform_data is not None:
                    assert teacher_embedding.shape[-1]==transform_data.shape[-1], \
                        'Dimension mismatch: Training and transformation data must have the same dimensions'

                if normalize:
                    print('Normalizing training data...')
                    stdscaler = StandardScaler()
                    teacher_embedding = stdscaler.fit_transform(teacher_embedding)

                    if transform_data is not None:
                        print('Normalizing transform data')
                        transform_data = stdscaler.transform(transform_data)
                                                                                        
                    if save_model_dir:
                        dump(stdscaler, os.path.join(save_model_dir, 'scaler.bin'), compress=True)
                        print('Standard Scaler saved....')

                print('Batch size:', curr_batch_size)
                if approx_mode == 'umap':
                    results = get_reduced_embedding(teacher_embedding, 'umap',
                                                    emb_len=reduced_emb_len, 
                                                    transform_data=transform_data,
                                                    neighbors=neighbors_list[0],
                                                    metric=metric_list[0],
                                                    min_dist=min_dist_list[0])
                        
                elif approx_mode == 'tsne':
                    n_process = len(neighbors_list) * len(metric_list) * len(tsne_iter_list)

                    results = Parallel(n_jobs=n_process)(delayed(get_reduced_embedding) \
                                                             (teacher_embedding, 'tsne',
                                                              emb_len=reduced_emb_len,
                                                              neighbors=neighbors,
                                                              metric=metric,
                                                              iterations=iterations) \
                                                             for neighbors in neighbors_list for metric in metric_list
                                                             for iterations in tsne_iter_list)
                elif approx_mode == 'pca':
                    results = get_reduced_embedding(teacher_embedding, 'pca',
                                                    emb_len=reduced_emb_len, 
                                                    pca_kernel=pca_kernel,
                                                    transform_data=transform_data, 
                                                    random_state=random_state,
                                                    save_model_dir=save_model_dir,
                                                    ndata=batch_size)
                else:
                    raise ValueError('Supported modes: umap, pca, tsne')

                # Due to multi-threading for tsne, we need to accumulate the results
                if approx_mode=='tsne':
                    assert len(results) == n_process

                    for idx in range(len(results)):
                        if blob_keys[idx] not in blob_embeddings.keys():
                            blob_embeddings[blob_keys[idx]] = np.zeros((batch_size, reduced_emb_len),
                                                                       dtype=np.float32)
                            blob_embeddings[blob_keys[idx]] = results[idx]
                        else:
                            blob_embeddings[blob_keys[idx]] = results[idx]
                else:
                    if blob_keys[0] not in blob_embeddings.keys():
                        blob_embeddings[blob_keys[0]] = np.zeros((batch_size, reduced_emb_len), dtype=np.float32)
                        blob_embeddings[blob_keys[0]] = results
                    else:
                        blob_embeddings[blob_keys[0]] = results

                if transform_data_dir is not None:
                    embedding_out_paths = transform_out_paths

                write_start = time.time()
                if is_transf_data_sonyc_ust:
                    save_npz_sonyc_ust(embedding_out_paths, blob_embeddings, transform_blob_lens)
                else:
                    write_to_h5(embedding_out_paths, blob_embeddings, batch_size)
                write_end = time.time()
                f_idx += 1
                print('All files saved! Write took {} seconds'.format((write_end - write_start)))
                print('-----------------------------------------\n')

            batch_idx += 1
            curr_batch_size = 0
            batch = None
            blob_embeddings = dict()
            embedding_out_paths = []
            read_start = time.time()


def generate_trained_embeddings_driver(estimator_path, data_dir, output_dir, 
                                        continue_extraction=False, 
                                        partition_to_run=None,
                                        num_partitions=15, 
                                        **kwargs
                                    ):
    # def divide_chunks(l, n):
    #     # looping till length l
    #     for i in range(0, len(l), n):
    #         yield l[i:i + n]

    if partition_to_run is None:
        # Compute remaining list of files
        if continue_extraction:
            list_files = list(set(os.listdir(data_dir)) - set(os.listdir(output_dir)))
            print('Continuing to extract for {} remaining files'.format(len(list_files)))
        else:
            list_files = None
    else:
        # Run specified partition out of given num_partitions
        # Get list of files to run
        print('Partition to run: {} out of {} partitions'.format(partition_to_run, num_partitions))
        # list_files = all_files[partition_to_run]
        list_files = [
            f for f in sorted(os.listdir(data_dir)) if re.search(
                          rf'sonyc_ndata=2500000_part={partition_to_run}_(.*).h5', 
                          os.path.basename(f)
                      )]

    # Call embedding generator with appropriate list of files
    reduced_embedding_predictor(estimator_path,
                                data_dir,
                                output_dir, list_files=list_files, **kwargs)


def sanity_check_downsampled_l3_dataset(data_dir, verbose=True):
    list_files = glob.glob(os.path.join(data_dir, '*.h5'))

    num_points = 0
    for file in list_files:
        if verbose:
            print('Processing {}'.format(file))

        f = h5py.File(file, 'r')

        if "dataset_index" in list(f.keys()):
            flag = 'sonyc'
        else:
            flag = 'music'

        # Verify that one-to-one correspondence exists
        assert len(f["filename"]) == len(f["l3_embedding"]) == len(f["feature_index"])
        if flag == 'sonyc':
            assert len(f["filename"]) == len(f["dataset_index"])

        num_points += len(f["l3_embedding"])

        orig_data_paths = list(f["filename"])
        for i in range(len(orig_data_paths)):
            downsampled_data = f["l3_embedding"][i]
            orig_f = h5py.File(orig_data_paths[i], 'r')

            if flag == 'music':
                if verbose:
                    print('\tFeature {} drawn from file {}, feature_index {}'.format(i, orig_data_paths[i],
                                                                                     f["feature_index"][i]))
                orig_data = orig_f[list(orig_f.keys())[0]][f["feature_index"][i]]
            else:
                print(
                    '\tFeature {} drawn from file {}, dataset_index {}, feature_index {}'.format(i, orig_data_paths[i],
                                                                                                 f["dataset_index"][i],
                                                                                                 f["feature_index"][i]))
                orig_data = orig_f[list(orig_f.keys())[0]][f["dataset_index"][i]][1][f["feature_index"][i]]

            assert np.array_equal(downsampled_data, orig_data)

    print('All done!')
    print('Total number of points is file: {}'.format(num_points))


def train_reduced_embedding(data_dir, output_dir, reduced_emb_len, neighbors=5,
                         metric='euclidean', min_dist=0.3, batch_size=1024, random_state=20180123, normalize = True,
                         start_batch_idx=None):
    if data_dir == output_dir:
        raise ValueError('Output path should not be same as data path to avoid overwriting data files!')

    random.seed(random_state)

    out_file_name = os.path.join(output_dir,
                                 'umap_ndata={}_emb={}_nbrs={}_mindist={}_mtrc={}.sav'.
                                 format(batch_size, reduced_emb_len, neighbors, min_dist, metric))

    print('Data directory:', data_dir)
    print('Model out filename:', out_file_name)

    batch = None
    curr_batch_size = 0
    batch_idx = 0

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    list_files = [ x for x in os.listdir(data_dir) if x.endswith('h5')]
    last_file = list_files[-1]
    print('Last file on the list: ', last_file)

    for fname in list_files:
        batch_path = os.path.join(data_dir, fname)
        blob_start_idx = 0

        blob = h5py.File(batch_path, 'r')
        blob_size = len(blob['l3_embedding'])

        blob_end_idx = blob_size

        # If we are starting from a particular batch, skip computing all of
        # the prior batches
        if start_batch_idx is None or batch_idx >= start_batch_idx:
            if batch is None:
                batch = {'l3_embedding': blob['l3_embedding'][blob_start_idx:blob_end_idx]}
            else:
                batch['l3_embedding'] = np.concatenate(
                    [batch['l3_embedding'], blob['l3_embedding'][blob_start_idx:blob_end_idx]])

        curr_batch_size += blob_end_idx - blob_start_idx

        if blob_end_idx == blob_size:
            blob.close()

        # Use only the first full batch for training
        if curr_batch_size == batch_size or fname == last_file:
            # If we are starting from a particular batch, skip yielding all
            # of the prior batches
            if start_batch_idx is None or batch_idx >= start_batch_idx:
                teacher_embedding = batch['l3_embedding']  # get_teacher_embedding(batch['audio'])
                print('Collected data; shape: {}'.format(teacher_embedding.shape))
                np.random.shuffle(teacher_embedding)
                print('Size in mem: {} GB'.format(teacher_embedding.nbytes / 1e9))

                # Normalize
                if normalize:
                    scaler = StandardScaler()
                    teacher_embedding = scaler.fit_transform(teacher_embedding)
                    # Save scaler
                    dump(scaler, os.path.join(output_dir, 'scaler.bin'), compress=True)


                reducer = umap.UMAP(n_neighbors=neighbors, min_dist=min_dist,
                                    metric=metric, n_components=reduced_emb_len, verbose=True)

                print('Starting UMAP training: sample_size={}, num_neighbors={},'
                      'min_dist={}, metric={}, reduced_emb_len={}'.format(curr_batch_size, neighbors,
                                                                          min_dist, metric, reduced_emb_len))

                start_time = time.time()
                trans = reducer.fit(teacher_embedding)
                end_time = time.time()

                print('UMAP training finished: took {} hours'.format((end_time - start_time) / 3600))

                # Diagnostic
                #print('Train embedding shape: ', embedding.shape)

                # Save pickled model
                pickle.dump(reducer, open(out_file_name, 'wb'))
                #dump(reducer, out_file_name)
                print('UMAP model saved at ', out_file_name)

                return
