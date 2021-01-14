import sys
import csv
import glob
import h5py
import warnings
import io
import os
import math
import pescador
import multiprocessing
import time
import numpy as np
import random
import pickle
from joblib import Parallel, delayed, dump, load
from decrypt import read_encrypted_tar_audio_file
from tqdm import tqdm
from datetime import datetime
from dppy.finite_dpps import FiniteDPP
from sklearn import preprocessing


def create_feature_file_partitions(feature_dir, output_dir, all_sensors=False, num_partitions=15, extra_files=False, random_state=20180123):
    def divide_chunks(l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    random.seed(random_state)

    if all_sensors:
        list_files = glob.glob(os.path.join(feature_dir, '*.h5'))
    elif extra_files: 
        extra_files = [
            'sonycnode-b827ebc6dcc6.sonyc_features_openl3.h5',
            'sonycnode-b827ebba613d.sonyc_features_openl3.h5',
            'sonycnode-b827ebad073b.sonyc_features_openl3.h5',
            'sonycnode-b827eb0fedda.sonyc_features_openl3.h5',
            'sonycnode-b827eb44506f.sonyc_features_openl3.h5',
            'sonycnode-b827eb132382.sonyc_features_openl3.h5',
            'sonycnode-b827eb2a1bce.sonyc_features_openl3.h5',
        ]
        list_files = [os.path.join(feature_dir, f) for f in extra_files]
    else:
        feature_files = [
            'sonycnode-b827eb2c65db.sonyc_features_openl3.h5',
            'sonycnode-b827eb539980.sonyc_features_openl3.h5',
            'sonycnode-b827eb42bd4a.sonyc_features_openl3.h5',
            'sonycnode-b827eb252949.sonyc_features_openl3.h5',
            'sonycnode-b827ebc7f772.sonyc_features_openl3.h5',
            'sonycnode-b827ebb40450.sonyc_features_openl3.h5',
            'sonycnode-b827ebefb215.sonyc_features_openl3.h5',
            'sonycnode-b827eb122f0f.sonyc_features_openl3.h5',
            'sonycnode-b827eb86d458.sonyc_features_openl3.h5',
            'sonycnode-b827eb4e7821.sonyc_features_openl3.h5',
            'sonycnode-b827eb0d8af7.sonyc_features_openl3.h5',
            'sonycnode-b827eb29eb77.sonyc_features_openl3.h5',
            'sonycnode-b827eb815321.sonyc_features_openl3.h5',
            'sonycnode-b827eb1685c7.sonyc_features_openl3.h5',
            'sonycnode-b827eb4cc22e.sonyc_features_openl3.h5'
        ]
        list_files = [os.path.join(feature_dir, f) for f in feature_files]
        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    partition_length = math.ceil(len(list_files) / num_partitions)
    list_partitions = list(divide_chunks(list_files, partition_length))

    for i, partition in enumerate(list_partitions):
        with open(os.path.join(output_dir, str(i) + '.csv'), 'w') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(partition)


def downsample_sonyc(feature_partitions_dir, indices_dir, output_dir, sample_size, partition_num,
                     audio_dir='/beegfs/work/sonyc', audio_samp_rate=8000, random_state=20180123,
                     embeddings_per_file=1024):
    # Streamer weights are proportional to the number of datasets in the corresponding file
    def generate_pescador_stream_weights(list_files):
        num_datasets = []
        for file in list_files:
            f = h5py.File(file, 'r')
            num_datasets.append(f[list(f.keys())[0]].shape[0])

        num_datasets = np.array(num_datasets)
        return num_datasets.astype(float) / num_datasets.sum()

    @pescador.streamable
    def random_feature_generator(h5_path):
        f = h5py.File(h5_path, 'r')
        num_datasets = f[list(f.keys())[0]].shape[0]
        while True:
            dataset_index = np.random.randint(0, num_datasets)
            num_features = f[list(f.keys())[0]][dataset_index]['openl3'].shape[0]
            # audio_file_name, row = big_dict[f[list(f.keys())[0]][dataset_index]['filename'].decode()]
            index = h5py.File(
                os.path.join(indices_dir, os.path.basename(h5_path).split('.')[0] + '.sonyc_recording_index.h5'), 'r')
            audio_file_name = os.path.join(audio_dir,
                                           index[list(index.keys())[0]][dataset_index]['day_hdf5_path'].decode())
            row = index[list(index.keys())[0]][dataset_index]['day_h5_index']
            audio_file = h5py.File(audio_file_name, 'r')
            tar_data = io.BytesIO(audio_file['recordings'][row]['data'])
            # Read encrypted audio
            raw_audio = get_raw_windows_from_encrypted_audio(audio_file_name, tar_data,
                                                             sample_rate=audio_samp_rate)
            if raw_audio is None:
                continue
            feature_index = np.random.randint(0, num_features)
            yield f[list(f.keys())[0]][dataset_index]['openl3'][feature_index], raw_audio[feature_index]

    assert sample_size > 0

    random.seed(random_state)

    with open(os.path.join(feature_partitions_dir, str(partition_num) + '.csv'), 'r') as f:
        rdr = csv.reader(f, quoting=csv.QUOTE_ALL)
        for row in rdr:
            list_files = row

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    streams = [random_feature_generator(x) for x in list_files]
    rate = math.ceil(sample_size / len(streams))
    print('Num. of pescador streams: {}; Rate: {}'.format(len(streams), rate))

    mux = pescador.StochasticMux(streams, weights=generate_pescador_stream_weights(list_files), n_active=50,
                                 rate=rate, mode='exhaustive')

    num_files = sample_size // embeddings_per_file

    accumulator = []
    rawlist = []
    splitindex = 1
    start_time = time.time()
    print('sample size: ', sample_size)
    for data, raw in mux(max_iter=sample_size):
        accumulator += [data]
        rawlist += [raw]
        if len(accumulator) == embeddings_per_file:
            o_file = os.path.join(output_dir, 'sonyc_ndata={}_part={}_split={}.h5'.format(sample_size, 
                                                                                          partition_num,
                                                                                          splitindex)
            )
            if os.path.exists(o_file):
               continue

            outfile = h5py.File(o_file, 'w')
            outfile.create_dataset('audio', data=np.array(rawlist), chunks=True)
            outfile.create_dataset('l3_embedding', data=np.array(accumulator), chunks=True)
            end_time = time.time()
            print('Wrote {}/{} files, processing time: {} s'.format(splitindex, num_files,
                                                                    (end_time - start_time)))
            accumulator = []
            rawlist = []
            splitindex += 1
            start_time = time.time()


def sample_sonyc(sensor, sample_size, output_dir, mode, timescale='day', ind_path='/beegfs/work/sonyc/indices/2017',
                 rel_path='/beegfs/sk7898/sonyc/relevance/2017',
                 emb_path='/beegfs/work/sonyc/features/openl3/2017',
                 ind_suffix='_recording_index.h5', rel_suffix='_relevance_2hr.h5',
                 emb_suffix='_features_openl3.h5',
                 random_state=20180123):
    # Sampling function
    @pescador.streamable
    def feature_generator(arr_indices, frame):
        if mode == 'random':
            # Initialize matrix of ones (every embedding is equi-probable)
            P = np.ones((len(arr_indices), embs_per_row))
            # Convert to a probability distribution
            P = P / P.sum()
        elif mode == 'spl':
            # Find relative rank sub-matrix corresponding to arr_indices (rows)
            P = rel_2hr_ranks[arr_indices]
            # Convert to a probability distribution
            P = P / P.sum()
        elif 'dpp' in mode:
            embs_in_frame = emb_file[list(emb_file.keys())[0]]['openl3'][arr_indices]
            feats = embs_in_frame.reshape(-1, emb_dim)
            feats_normalized = preprocessing.normalize(feats, norm='l1')

            if mode == 'dpp_div': # Diversity only
                L = feats_normalized.dot(feats_normalized.T) + 1e-1 * np.eye(feats.shape[0]) # add an identity matrix to make L full rank
            elif mode == 'dpp': # Quality and diversity
                q = rel_2hr_ranks[arr_indices].reshape(-1, 1)
                L = q.T * feats_normalized.dot(feats_normalized.T) * q + 1e-1 * np.eye(feats.shape[0]) # add an identity matrix to make L full rank

            DPP = FiniteDPP('likelihood', **{'L': L})
            P = np.ones((len(arr_indices), embs_per_row)) # Only needed to reshape selected flat indices for DPP

        # Deactivate stream if a timeframe doesn't contain enough samples
        if samples_per_stream > P.size:
            return

        print('Samples to pick: {}, number of embeddings: {}, timeframe index: {}'.format(samples_per_stream, P.size,
                                                                                          frame))

        while True:
            # Sample samples_per_stream embeddings
            if 'dpp' not in mode:
                ij_vec = np.random.choice(range(P.size), size=samples_per_stream, p=P.ravel(), replace=False)
            else:
                # Initializing. See: https://github.com/guilgautier/DPPy/issues/58
                rng = np.random.RandomState(413121)
                S0 = rng.choice(feats.shape[0], samples_per_stream, replace=False)

                print('det L_S0 = ', np.linalg.det(L[np.ix_(S0, S0)]))

                DPP.flush_samples()
                DPP.sample_mcmc_k_dpp(size=samples_per_stream, s_init=S0, nb_iter=3)
                ij_vec = DPP.list_of_samples[0][-1]

            emb_list = []
            for k, ij in enumerate(ij_vec):
                i, j = np.unravel_index(ij, P.shape)
                print('{}th index selected:({},{})'.format(k, i, j))

                # Get corresponding embedding (adding offset arr_indices[0])
                emb_list.append(emb_file[list(emb_file.keys())[0]][arr_indices[0] + i]['openl3'][j])

            yield frame, np.array(emb_list)

    assert mode in ['random', 'spl', 'dpp_div', 'dpp']
    assert sample_size > 0
    random.seed(random_state)
    np.random.seed(random_state)

    os.makedirs(output_dir, exist_ok=True)

    index_file = h5py.File(os.path.join(ind_path, sensor + ind_suffix))
    rel_file = h5py.File(os.path.join(rel_path, sensor + rel_suffix))
    emb_file = h5py.File(os.path.join(emb_path, sensor + emb_suffix))

    emb_dim = emb_file[list(emb_file.keys())[0]][0]['openl3'][0].shape[-1]
    print('Embedding dimension: {}'.format(emb_dim))

    embs_per_row = emb_file[list(emb_file.keys())[0]][0]['openl3'].shape[0]
    print('Number of embeddings per row: {}'.format(embs_per_row))

    # Find day/week of the year corresponding to UNIX timestamps
    spl_ts_array = index_file[list(index_file.keys())[0]]['spl_timestamp']

    if timescale == 'day':
        fmt = "%j"
    elif timescale == 'week':
        fmt = '%V'
    elif timescale == 'month':
        fmt = '%m'

    timeframe_arr = []
    for ts in spl_ts_array:
        timeframe_arr.append(int(datetime.fromtimestamp(ts).strftime(fmt)))

    timeframe_arr = np.array(timeframe_arr)

    arr_indices = []
    unique_frames = []
    for unique_day in np.unique(timeframe_arr):
        unique_frames.append(unique_day)
        arr_indices.append(np.where(timeframe_arr == unique_day)[0])

    if mode in ['spl', 'dpp']:
        # Compute relative 2 hour rank matrix
        rel_2hr_ranks = rel_file[list(rel_file.keys())[0]]['relevance_2hr_vector']

    # Init streams
    streams = [feature_generator(x, f) for x, f in zip(arr_indices, unique_frames)]
    samples_per_stream = math.ceil(sample_size / len(streams))
    print('Num. of pescador streams: {}; Samples per stream: {}'.format(len(streams), samples_per_stream))

    # Rate set to 1 here; would be sampling samples_per_stream number of embeddings in generator function
    if 'dpp' not in mode:
        mux = pescador.StochasticMux(streams, n_active=500, rate=1, mode='exhaustive')
    else:
        mux = pescador.StochasticMux(streams, n_active=1, rate=1, mode='exhaustive')

    # Sample
    accumulator = []
    unique_frames_sampled = []
    for frame, data in mux(max_iter=len(streams)):  # Since pulling samples_per_stream embeddings from each stream
        accumulator.append(data)
        unique_frames_sampled.append(frame)

    print(
        'Number of unique frames sampled from: {}/{} ({}%)'.format(len(np.unique(unique_frames_sampled)), len(streams),
                                                                   len(np.unique(unique_frames_sampled)) / len(
                                                                       streams) * 100))
    print('Unique frames sampled from: {}'.format(np.unique(unique_frames_sampled)))

    accumulator = np.array(accumulator).reshape(-1, emb_dim)
    print('Sampling complete; data shape: {}'.format(accumulator.shape))

    # Write to output file
    outfile_name = os.path.join(output_dir,
                                'sonyc-{}_sensor={}_ndata={}_timescale={}.h5'.format(mode, sensor, accumulator.shape[0],
                                                                                     timescale))
    outfile = h5py.File(outfile_name, 'w')
    outfile.create_dataset('l3_embedding', data=np.array(accumulator), chunks=True)
    print('Samples saved to:', outfile_name)


def get_raw_windows_from_encrypted_audio(audio_path, tar_data, sample_rate=8000, clip_duration=10,
                                         decrypt_url='https://decrypt-sonyc.engineering.nyu.edu/decrypt',
                                         cacert_path='/home/jtc440/sonyc/decrypt/CA.pem',
                                         cert_path='/home/jtc440/sonyc/decrypt/jason_data.pem',
                                         key_path='/home/jtc440/sonyc/decrypt/sonyc_key.pem'):
    audio = read_encrypted_tar_audio_file(audio_path,
                                          enc_tar_filebuf=tar_data,
                                          sample_rate=sample_rate,
                                          url=decrypt_url,
                                          cacert=cacert_path,
                                          cert=cert_path,
                                          key=key_path)[0]

    if audio is None:
        return None

    audio_len = int(sample_rate * clip_duration)

    # Make sure audio is all consistent length (10 seconds)
    if len(audio) > audio_len:
        audio = audio[:audio_len]
    elif len(audio) < audio_len:
        pad_len = audio_len - len(audio)
        audio = np.pad(audio, (0, pad_len), mode='constant')

    # Return raw windows
    return get_audio_windows(audio, sr=sample_rate)


def get_audio_windows(audio, sr=8000, center=True, hop_size=0.5):
    """
    Similar to openl3.get_embedding(...)
    """

    def _center_audio(audio, frame_len):
        """Center audio so that first sample will occur in the middle of the first frame"""
        return np.pad(audio, (int(frame_len / 2.0), 0), mode='constant', constant_values=0)

    def _pad_audio(audio, frame_len, hop_len):
        """Pad audio if necessary so that all samples are processed"""
        audio_len = audio.size
        if audio_len < frame_len:
            pad_length = frame_len - audio_len
        else:
            pad_length = int(np.ceil((audio_len - frame_len) / float(hop_len))) * hop_len \
                         - (audio_len - frame_len)

        if pad_length > 0:
            audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)

        return audio

    # Check audio array dimension
    if audio.ndim > 2:
        raise AssertionError('Audio array can only be be 1D or 2D')
    elif audio.ndim == 2:
        # Downmix if multichannel
        audio = np.mean(audio, axis=1)

    audio_len = audio.size
    frame_len = sr
    hop_len = int(hop_size * sr)

    if audio_len < frame_len:
        warnings.warn('Duration of provided audio is shorter than window size (1 second). Audio will be padded.')

    if center:
        # Center audio
        audio = _center_audio(audio, frame_len)

    # Pad if necessary to ensure that we process all samples
    audio = _pad_audio(audio, frame_len, hop_len)

    # Split audio into frames, copied from librosa.util.frame
    n_frames = 1 + int((len(audio) - frame_len) / float(hop_len))
    x = np.lib.stride_tricks.as_strided(audio, shape=(frame_len, n_frames),
                                        strides=(audio.itemsize, hop_len * audio.itemsize)).T

    # Add a channel dimension
    # x = x.reshape((x.shape[0], 1, x.shape[-1]))

    return x


def check_sonyc_openl3_points(feature_dir, indices_dir, out_path=None, verbose=False,
                              min_num_datasets_per_file=0, max_num_datasets_per_file=math.inf):
    num_pts = 0
    num_sets = 0
    files = glob.glob(os.path.join(feature_dir, '*.h5'))
    valid_files = []
    for fname in tqdm(files):
        f = h5py.File(fname, 'r')
        if min_num_datasets_per_file < f[list(f.keys())[0]].shape[0] < max_num_datasets_per_file:
            valid_files += [fname]
            num_sets += f[list(f.keys())[0]].shape[0]
            if verbose:
                print('File: {} Num. of datasets: {}'.format(fname, f[list(f.keys())[0]].shape[0]))
            num_pts += f[list(f.keys())[0]].shape[0] * f[list(f.keys())[0]][0]['openl3'].shape[0]

    print('SONYC features')
    print('--------------')
    print('Num files:', len(valid_files))
    print('Num points:', num_pts)
    print('Num datasets:', num_sets)

    num_sets_indices = 0
    files = glob.glob(os.path.join(indices_dir, '*.h5'))
    for fname in tqdm(files):
        f = h5py.File(fname, 'r')
        if min_num_datasets_per_file < f[list(f.keys())[0]].shape[0] < max_num_datasets_per_file:
            num_sets_indices += f[list(f.keys())[0]].shape[0]
            if verbose:
                print('File: {} Num. of datasets: {}'.format(fname, f[list(f.keys())[0]].shape[0]))

    print('SONYC indices')
    print('--------------')
    print('Num files:', len(valid_files))
    print('Num datasets:', num_sets_indices)

    assert num_sets == num_sets_indices

    # Write valid files to csv
    if out_path is not None:
        csvwrite = csv.writer(open(out_path, 'w', newline=''), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwrite.writerow(valid_files)


# Driver for sample_sonyc functions
def sample_sonyc_driver(sensor_index, total_sample_size, aggregation='combined', **kwargs):
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

    assert sensor_index in range(len(sensors))
    assert kwargs['timescale'] in ['day', 'week', 'month']
    assert aggregation in ['combined', 'per_sensor']

    print('INFO (sample_sonyc_driver): Sampling in mode {} from sensor index {} ({})'.format(kwargs['mode'],
                                                                                             sensor_index,
                                                                                             sensors[sensor_index]))

    if aggregation == 'combined':
        samples_per_sensor = math.ceil(total_sample_size / len(sensors))
    elif aggregation == 'per_sensor':
        samples_per_sensor = total_sample_size

    print('INFO (sample_sonyc_driver): Sampling {} embeddings from sensor {}'.format(samples_per_sensor,
                                                                                     sensors[sensor_index]))
    sample_sonyc(sensor=sensors[sensor_index], sample_size=samples_per_sensor, **kwargs)


if __name__ == '__main__':
    if sys.argv[1] == 'downsample_sonyc_points':
        downsample_sonyc('/beegfs/sk7898/sonyc_feature_partitions',
                         '/beegfs/work/sonyc/indices/2017',
                         '/scratch/sk7898/sonyc_30mil/', 
                         int(sys.argv[2]), 
                         int(sys.argv[3]), 
                         audio_samp_rate=8000)
    elif sys.argv[1] == 'create_feature_file_partitions':
        create_feature_file_partitions('/beegfs/work/sonyc/features/openl3/2017',
                                       '/beegfs/sk7898/sonyc_feature_partitions', 
                                       num_partitions=int(sys.argv[2]))
    elif sys.argv[1] == 'check_sonyc_openl3_points':
        check_sonyc_openl3_points('/beegfs/work/sonyc/features/openl3/2017',
                                  '/beegfs/work/sonyc/new_indices_validated_with_sensor_fault/2017')
    elif sys.argv[1] == 'sample_sonyc_spl_based':
        sample_sonyc_driver(int(sys.argv[4]), int(sys.argv[2]),
                            output_dir='/scratch/dr2915/sonyc_samples/spl/ndata={}/timescale={}'.format(sys.argv[2],
                                                                                                        sys.argv[3]),
                            mode='spl', timescale=sys.argv[3],
                            ind_path='/beegfs/work/sonyc/indices/2017', rel_path='/beegfs/sk7898/sonyc/relevance/2017',
                            emb_path='/beegfs/work/sonyc/features/openl3/2017')
    elif sys.argv[1] == 'sample_sonyc_spl_based_per_sensor':
        sample_sonyc_driver(int(sys.argv[4]), int(sys.argv[2]),
                            output_dir='/scratch/dr2915/sonyc_samples/per_sensor/'
                                       'spl/ndata={}/timescale={}'.format(sys.argv[2], sys.argv[3]),
                            mode='spl', aggregation='per_sensor', timescale=sys.argv[3],
                            ind_path='/beegfs/work/sonyc/indices/2017', rel_path='/beegfs/sk7898/sonyc/relevance/2017',
                            emb_path='/beegfs/work/sonyc/features/openl3/2017')
    elif sys.argv[1] == 'sample_sonyc_random':
        sample_sonyc_driver(int(sys.argv[4]), int(sys.argv[2]),
                            output_dir='/scratch/dr2915/sonyc_samples/random/ndata={}/timescale={}'.format(sys.argv[2],
                                                                                                        sys.argv[3]),
                            mode='random', timescale=sys.argv[3],
                            ind_path='/beegfs/work/sonyc/indices/2017',
                            emb_path='/beegfs/work/sonyc/features/openl3/2017')
    elif sys.argv[1] == 'sample_sonyc_random_per_sensor':
        sample_sonyc_driver(int(sys.argv[4]), int(sys.argv[2]),
                            output_dir='/scratch/dr2915/sonyc_samples/per_sensor/'
                                       'random/ndata={}/timescale={}'.format(sys.argv[2], sys.argv[3]),
                            mode='random', aggregation='per_sensor', timescale=sys.argv[3],
                            ind_path='/beegfs/work/sonyc/indices/2017',
                            emb_path='/beegfs/work/sonyc/features/openl3/2017')
    elif sys.argv[1] == 'sample_sonyc_dpp_div':
        sample_sonyc_driver(int(sys.argv[4]), int(sys.argv[2]),
                            output_dir='/scratch/dr2915/sonyc_samples/dpp_div/ndata={}/timescale={}'.format(sys.argv[2],
                                                                                                        sys.argv[3]),
                            mode='dpp_div', timescale=sys.argv[3],
                            ind_path='/beegfs/work/sonyc/indices/2017',
                            emb_path='/beegfs/work/sonyc/features/openl3/2017')
    elif sys.argv[1] == 'sample_sonyc_dpp_div_per_sensor':
        sample_sonyc_driver(int(sys.argv[4]), int(sys.argv[2]),
                            output_dir='/scratch/dr2915/sonyc_samples/per_sensor/'
                                       'dpp_div/ndata={}/timescale={}'.format(sys.argv[2], sys.argv[3]),
                            mode='dpp_div', aggregation='per_sensor', timescale=sys.argv[3],
                            ind_path='/beegfs/work/sonyc/indices/2017',
                            emb_path='/beegfs/work/sonyc/features/openl3/2017')
    elif sys.argv[1] == 'sample_sonyc_dpp':
        sample_sonyc_driver(int(sys.argv[4]), int(sys.argv[2]),
                            output_dir='/scratch/{}/sonyc_samples/dpp/ndata={}/timescale={}'.format(sys.argv[5],
                                                                                                    sys.argv[2],
                                                                                                    sys.argv[3]),
                            mode='dpp', timescale=sys.argv[3],
                            ind_path='/beegfs/work/sonyc/indices/2017', 
                            rel_path='/beegfs/sk7898/sonyc/relevance/2017',
                            emb_path='/beegfs/work/sonyc/features/openl3/2017')
    elif sys.argv[1] == 'sample_sonyc_dpp_per_sensor':
        sample_sonyc_driver(int(sys.argv[4]), int(sys.argv[2]),
                            output_dir='/scratch/dr2915/sonyc_samples/per_sensor/'
                                       'dpp/ndata={}/timescale={}'.format(sys.argv[2], sys.argv[3]),
                            mode='dpp', aggregation='per_sensor', timescale=sys.argv[3],
                            ind_path='/beegfs/work/sonyc/indices/2017', rel_path='/beegfs/sk7898/sonyc/relevance/2017',
                            emb_path='/beegfs/work/sonyc/features/openl3/2017')