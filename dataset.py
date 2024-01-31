import numpy as np
import os
import cv2
import pickle
import yaml
from PedestrianActionBenchmark.jaad_data import JAAD
from PedestrianActionBenchmark.pie_data import PIE
from torch.utils.data import Dataset, random_split
import torch
os.chdir('/workspace')

class prepare_data(object):

    def __init__(self, dataset='PIE', cache = True):
        """
        Initializes the data preparation class
        Args:
            dataset: Name of the dataset
            opts: Options for preparing data
        """

        self._dataset = dataset
        self._cache = cache
        self._generator = None
        configs_default ='PedestrianActionBenchmark/config_files/configs_default.yaml'
        with open(configs_default, 'r') as f:
            configs = yaml.safe_load(f)
        tte = configs['model_opts']['time_to_event'] if isinstance(configs['model_opts']['time_to_event'], int) else \
            configs['model_opts']['time_to_event'][1]
        configs['data_opts']['min_track_size'] = configs['model_opts']['obs_length'] + tte
        configs['model_opts']['obs_input_type'] = ['box','speed','center','image']
        configs['model_opts']['dataset'] = self._dataset.lower()
        configs['model_opts']['generator'] = False
        configs['data_opts']['sample_type'] = "all"
        configs['model_opts']['overlap'] = 0.5001
        self.configs = configs
        try: 
            if self._dataset == 'JAAD':
                with open("data/JAAD_data/beh_seq_train.pkl", 'rb') as f:
                    self.beh_seq_train = pickle.load(f)
                with open("data/JAAD_data/beh_seq_val.pkl", 'rb') as f:
                    self.beh_seq_val = pickle.load(f)
                with open("data/JAAD_data/beh_seq_test.pkl", 'rb') as f:
                    self.beh_seq_test = pickle.load(f)
            elif self._dataset == 'PIE':
                with open("data/PIE_data/beh_seq_train.pkl", 'rb') as f:
                    self.beh_seq_train = pickle.load(f)
                with open("data/PIE_data/beh_seq_val.pkl", 'rb') as f:
                    self.beh_seq_val = pickle.load(f)
                with open("data/PIE_data/beh_seq_test.pkl", 'rb') as f:
                    self.beh_seq_test = pickle.load(f)
        except:
            if self._dataset == 'JAAD':
                data_path = "JAAD"
                self._data_raw = JAAD(data_path=data_path)
                imdb = JAAD(data_path=data_path)
            elif self._dataset == 'PIE':
                data_path = "PIE"
                self._data_raw = PIE(data_path=data_path)
                imdb = PIE(data_path=data_path)

            self.beh_seq_train = imdb.generate_data_trajectory_sequence('train', **configs['data_opts'])
            self.beh_seq_val = imdb.generate_data_trajectory_sequence('val', **configs['data_opts'])
            self.beh_seq_test = imdb.generate_data_trajectory_sequence('test', **configs['data_opts'])
            

            if cache:
                if not os.path.exists('data'):
                    os.makedirs('data')


                if self._dataset == 'JAAD':
                    if not os.path.exists('data/JAAD_data'):
                        os.makedirs('data/JAAD_data')
                    with open("data/JAAD_data/beh_seq_train.pkl", 'wb') as f:
                        pickle.dump(self.beh_seq_train, f)
                    with open("data/JAAD_data/beh_seq_val.pkl", 'wb') as f:
                        pickle.dump(self.beh_seq_val, f)
                    with open("data/JAAD_data/beh_seq_test.pkl", 'wb') as f:
                        pickle.dump(self.beh_seq_test, f)
                elif self._dataset == 'PIE':
                    if not os.path.exists('data/PIE_data'):
                        os.makedirs('data/PIE_data')
                    with open("data/PIE_data/beh_seq_train.pkl", 'wb') as f:
                        pickle.dump(self.beh_seq_train, f)
                    with open("data/PIE_data/beh_seq_val.pkl", 'wb') as f:
                        pickle.dump(self.beh_seq_val, f)
                    with open("data/PIE_data/beh_seq_test.pkl", 'wb') as f:
                        pickle.dump(self.beh_seq_test, f)
        self.train_data = self.get_data('train',self.beh_seq_train,self.configs['model_opts'])
        self.val_data = self.get_data('val',self.beh_seq_val,self.configs['model_opts'])
        self.test_data = self.get_data('test',self.beh_seq_test,self.configs['model_opts'])

    def get_data_sequence(self, data_type, data_raw, opts):
            """
            Generates raw sequences from a given dataset
            Args:
                data_type: Split type of data, whether it is train, test or val
                data_raw: Raw tracks from the dataset
                opts:  Options for generating data samples
            Returns:
                A list of data samples extracted from raw data
                Positive and negative data counts
            """
            print('\n#####################################')
            print('Generating raw data')
            print('#####################################')
            d = {'center': data_raw['center'].copy(),
                'box': data_raw['bbox'].copy(),
                'ped_id': data_raw['pid'].copy(),
                'crossing': data_raw['activities'].copy(),
                'image': data_raw['image'].copy()}

            balance = opts['balance_data'] if data_type == 'train' else False
            obs_length = opts['obs_length']
            time_to_event = opts['time_to_event']
            normalize = opts['normalize_boxes']

            try:
                d['speed'] = data_raw['obd_speed'].copy()
            except KeyError:
                d['speed'] = data_raw['vehicle_act'].copy()
                print('Jaad dataset does not have speed information')
                print('Vehicle actions are used instead')
            if balance:
                self.balance_data_samples(d, data_raw['image_dimension'][0])
            d['box_org'] = d['box'].copy()
            d['tte'] = []

            if isinstance(time_to_event, int):
                for k in d.keys():
                    for i in range(len(d[k])):
                        d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
                d['tte'] = [[time_to_event]]*len(data_raw['bbox'])
            else:
                overlap = opts['overlap'] # if data_type == 'train' else 0.0
                olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
                olap_res = 1 if olap_res < 1 else olap_res
                for k in d.keys():
                    seqs = []
                    for seq in d[k]:
                        start_idx = len(seq) - obs_length - time_to_event[1]
                        end_idx = len(seq) - obs_length - time_to_event[0]
                        seqs.extend([seq[i:i + obs_length] for i in
                                    range(start_idx, end_idx + 1, olap_res)])
                    d[k] = seqs

                for seq in data_raw['bbox']:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                    range(start_idx, end_idx + 1, olap_res)])
            if normalize:
                for k in d.keys():
                    if k != 'tte':
                        if k != 'box' and k != 'center':
                            for i in range(len(d[k])):
                                d[k][i] = d[k][i][1:]
                        else:
                            for i in range(len(d[k])):
                                d[k][i] = np.subtract(d[k][i][1:], d[k][i][0]).tolist()
                    d[k] = np.array(d[k])
            else:
                for k in d.keys():
                    d[k] = np.array(d[k])

            d['crossing'] = np.array(d['crossing'])[:, 0, :]
            pos_count = np.count_nonzero(d['crossing'])
            neg_count = len(d['crossing']) - pos_count
            print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

            return d, neg_count, pos_count

    def balance_data_samples(self, d, img_width, balance_tag='crossing'):
        """
        Balances the ratio of positive and negative data samples. The less represented
        data type is augmented by flipping the sequences
        Args:
            d: Sequence of data samples
            img_width: Width of the images
            balance_tag: The tag to balance the data based on
        """
        print("Balancing with respect to {} tag".format(balance_tag))
        gt_labels = [gt[0] for gt in d[balance_tag]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                gt_augment = 1
            else:
                gt_augment = 0

            num_samples = len(d[balance_tag])
            for i in range(num_samples):
                if d[balance_tag][i][0][0] == gt_augment:
                    for k in d:
                        if k == 'center':
                            flipped = d[k][i].copy()
                            flipped = [[img_width - c[0], c[1]]
                                       for c in flipped]
                            d[k].append(flipped)
                        if k == 'box':
                            flipped = d[k][i].copy()
                            flipped = [np.array([img_width - b[2], b[1], img_width - b[0], b[3]])
                                       for b in flipped]
                            d[k].append(flipped)
                        if k == 'image':
                            flipped = d[k][i].copy()
                            flipped = [im.replace('.png', '_flip.png') for im in flipped]
                            d[k].append(flipped)
                        if k in ['speed', 'ped_id', 'crossing', 'walking', 'looking']:
                            d[k].append(d[k][i].copy())

            gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(gt_labels))
            num_neg_samples = len(gt_labels) - num_pos_samples
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(42)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]

            # update the data
            for k in d:
                seq_data_k = d[k]
                d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(d[balance_tag]) - num_pos_samples))

    
    def get_data(self, data_type, data_raw, model_opts,cache=True):
        """
        Generates data train/test/val data
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            model_opts: Model options for generating data
        Returns:
            A dictionary containing, data, data parameters used for model generation,
            effective dimension of data (the number of rgb images to be used calculated accorfing
            to the length of optical flow window) and negative and positive sample counts
        """
        try: 
            if self._dataset == 'JAAD':
                with open("data/JAAD_data/JAAD_{}.pkl".format(data_type), 'rb') as f:
                    dd = pickle.load(f)
            elif self._dataset == 'PIE':
                with open("data/PIE_data/PIE_{}.pkl".format(data_type), 'rb') as f:
                    dd = pickle.load(f)
        except:
            
            self._generator = model_opts.get('generator', False)
            data_type_sizes_dict = {}
            process = model_opts.get('process', True)
            dataset = model_opts['dataset']
            data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

            data_type_sizes_dict['box'] = data['box'].shape[1:]
            if 'speed' in data.keys():
                data_type_sizes_dict['speed'] = data['speed'].shape[1:]

            # Store the type and size of each image
            _data = []
            data_sizes = []
            data_types = []

            for d_type in model_opts['obs_input_type']:
                if 'local' in d_type or 'context' in d_type:
                    features, feat_shape = self.get_context_data(model_opts, data, data_type, d_type)
                elif 'pose' in d_type:
                    path_to_pose, _ = get_path(save_folder='poses',
                                            dataset=dataset,
                                            save_root_folder='data/features')
                    features = get_pose(data['image'],
                                        data['ped_id'],
                                        data_type=data_type,
                                        file_path=path_to_pose,
                                        dataset=model_opts['dataset'])
                    feat_shape = features.shape[1:]
                else:
                    features = data[d_type]
                    feat_shape = features.shape[1:]
                _data.append(features)
                data_sizes.append(feat_shape)
                data_types.append(d_type)

            # create the final data file to be returned
            if self._generator:
                _data = (DataGenerator(data=_data,
                                    labels=data['crossing'],
                                    data_sizes=data_sizes,
                                    process=process,
                                    global_pooling=self._global_pooling,
                                    input_type_list=model_opts['obs_input_type'],
                                    batch_size=model_opts['batch_size'],
                                    shuffle=data_type != 'test',
                                    to_fit=data_type != 'test'), data['crossing']) # set y to None
            else:
                _data = (_data, data['crossing'])
            dd = {'data': _data,
                    'ped_id': data['ped_id'],
                    'image': data['image'],
                    'tte': data['tte'],
                    'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                    'count': {'neg_count': neg_count, 'pos_count': pos_count}}
            if cache:
                if not os.path.exists('data'):
                    os.makedirs('data')
                if self._dataset == 'JAAD':
                    if not os.path.exists('data/JAAD_data'):
                        os.makedirs('data/JAAD_data')
                    with open("data/JAAD_data/JAAD_{}.pkl".format(data_type), 'wb') as f:
                        pickle.dump(dd, f)
                elif self._dataset == 'PIE':
                    if not os.path.exists('data/PIE_data'):
                        os.makedirs('data/PIE_data')
                    with open("data/PIE_data/PIE_{}.pkl".format(data_type), 'wb') as f:
                        pickle.dump(dd, f)


        return dd


class tabular_transformer(Dataset):
    # load the dataset
    def __init__(self,set_data,transform ="transform_1"):
        y = set_data['data'][1]
        self.y_mat = np.zeros((len(y),2))
        
        self.y_mat[np.where(y==1)[0],1] = 1
        self.y_mat[np.where(y==0)[0],0] = 1
        self.y = self.y_mat
        self.bbox = set_data['data'][0][0]
        self.speed = set_data['data'][0][1]
        self.center = set_data['data'][0][2]
        if transform == "transform_1":
            self.transform_1()
    def transform_1(self):
        self.X = np.concatenate((self.bbox,self.speed,self.center),axis=2)

    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
