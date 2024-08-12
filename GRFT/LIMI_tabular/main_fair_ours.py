import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import sys, os
import copy
import logging
import itertools
from data.census import census_predict_data
from data.bank import bank_predict_data
from data.credit import credit_predict_data
from data.compas import compas_predict_data
from data.lsac import lsac_predict_data
# from data.meps import meps_predict_data
# from utils.utils_tf import model_prediction, model_argmax, predict_prob
from table_model.dnn_models import dnn
from utils.config import census, credit, bank, compas,lsac
from utils.logger import setup_logger
from utils import utils_base
import argparse
from tqdm import tqdm
import torch
from ctgan.synthesizers.ctgan import CTGANSynthesizer
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
EXP_DIR = "./exp"

# store the result of fairness testing

class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, dense_len, min=-1, max=1, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        tf.keras.constraints.MinMaxNorm()
        self.scale = K.variable([[1. for x in range(dense_len)]], name='ffff',
                                constraint=lambda t: tf.clip_by_value(t, min, max))
        self.dense_len = dense_len
    def call(self, inputs, **kwargs):
        m = inputs * self.scale
        return m
    def get_config(self):
        config = {'dense_len': self.dense_len}
        base_config = super(ScaleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



tot_inputs = set()
global_disc_inputs = set()
global_disc_inputs_list = []
local_disc_inputs = set()
local_disc_inputs_list = []
value_list = []
suc_idx = []


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="main_fair")
    parser.add_argument("--exp_name", type=str, default="_")
    parser.add_argument("--dataset", type=str, default="census")
    parser.add_argument('--dataset_path', type=str, default='_')
    parser.add_argument('--model_path', type=str, default='_')
    parser.add_argument(
        "--sens_param",
        type=str,
        default="sens_param",
        help='sensitive index, index start from 1, 9 for gender, 8 for race',
    )
    parser.add_argument('--cluster_num', type=int, default=4)
    parser.add_argument('--max_global', type=int, default=1000000)

    parser.add_argument('--latent_file', type=str, default='_')
    parser.add_argument('--boundary_file', type=str, default='_')
    parser.add_argument('--svm_file', type=str, default='_')
    parser.add_argument('--gan_file', type=str, default='_')
    parser.add_argument('--step', type=float, default=0.3)

    parser.add_argument('--global_samples_file', type=str, default='_')
    parser.add_argument('--local_samples_file', type=str, default='_')
    parser.add_argument('--suc_idx_file', type=str, default='_')
    parser.add_argument('--disc_value_file', type=str, default='_')

    parser.add_argument("--random_seed", type=int, default=2333)
    opt = vars(parser.parse_args())

    opt["expdir"] = os.path.join(
        EXP_DIR,
        opt["experiment"],
        'ours' + '_' + str(opt['step']),
        opt["exp_name"],
    )
    utils_base.make_dir(opt["expdir"])

    if opt['global_samples_file'] == '_':
        opt['global_samples_file'] = 'global_samples.npy'
        opt['local_samples_file'] = 'local_samples.npy'
        opt['suc_idx_file'] = 'suc_idx.npy'
        opt['disc_value_file'] = 'disc_value.npy'

    opt['global_samples_file'] = os.path.join(opt['expdir'], opt['global_samples_file'])
    opt['local_samples_file'] = os.path.join(opt['expdir'], opt['local_samples_file'])
    opt['suc_idx_file'] = os.path.join(opt['expdir'], opt['suc_idx_file'])
    opt['disc_value_file'] = os.path.join(opt['expdir'], opt['disc_value_file'])
    print(opt["model_path"].replace("/","_")+opt["exp_name"]+".txt")
    logger = setup_logger(opt["expdir"],logfile_name=opt["model_path"].replace("/","_")+opt["exp_name"]+".txt", logger_name="logger", mode="w")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opt["random_seed"])
    np.random.seed(opt["random_seed"])
    logger.info(f"the random_seed is set as {opt['random_seed']}\n")
    logger.info(f"setting {opt}\n")

    return opt, logger


def numpy2log(arr, output_path):
    f = open(output_path, 'w')
    for item in arr:
        if isinstance(item, list):
            line = ','.join(list(map(str, item)))
        else:
            line = str(item)
        f.write(line + '\n')
    return


def sign(value):
    if value >= 0:
        return 1
    else:
        return -1


class FairTest:
    def __init__(self, opt, logger) -> None:
        self.opt = opt
        self.logger = logger
        self.n_value = 0
        self.global_nvalue = []

        # self.build_tf_env()
        self.build_data_and_model()
        self.build_ctgan_bd_clf()
        self.start_time = time.time()
        self.count = [1]

    # def build_tf_env(self):
    #     tf.set_random_seed(self.opt["random_seed"])
    #     config = tf.ConfigProto()
    #     config.gpu_options.per_process_gpu_memory_fraction = 0.5
    #     config.gpu_options.allow_growth = True
    #     self.sess = tf.Session(config=config)

    def build_data_and_model(self):
        with open(self.opt['latent_file'], 'rb') as f:
            self.latent_codes = pickle.load(f)

        data = {
            "census": census_predict_data,
            "credit": credit_predict_data,
            "bank": bank_predict_data,
            "compas":compas_predict_data,
            "lsac":lsac_predict_data
            # "meps": meps_predict_data,
        }
        data_config_set = {
            "census": census,
            "credit": credit,
            "bank": bank,
            "compas":compas,
            "lsac":lsac
            # "meps": meps,
        }
        # prepare the testing data and model

        X, Y, input_shape, nb_classes = data[self.opt['dataset']](
            self.opt['dataset_path']
        )
        # x = tf.placeholder(tf.float32, shape=input_shape)
        # y = tf.placeholder(tf.float32, shape=(None, nb_classes))
        self.X, self.Y = X, Y
        # self.input_shape, self.nb_classes = input_shape, nb_classes
        # self.x, self.y = x, y
        self.data_config = data_config_set[self.opt['dataset']]

        # self.model = dnn(input_shape, nb_classes)
        # self.preds = self.model(x)
        # saver = tf.train.Saver()
        # model_path = os.path.join(self.opt['model_path'], "dnn", "best.model")
        # saver.restore(self.sess, model_path)
        model_path = self.opt['model_path']
        self.model=keras.models.load_model(model_path,custom_objects={'ScaleLayer': ScaleLayer})
        # print(self.model(np.array([[1,2,3,4,5,6,7,8,9,10,11,12]])))
        self.logger.info(f"load model from {model_path}")

    def load_boundary(self, path: str):
        with open(path, 'rb') as f:
            bd = pickle.load(f)
        ans = bd
        return ans

    def build_ctgan_bd_clf(self):
        self.ctgan_syn: CTGANSynthesizer
        self.ctgan_syn = CTGANSynthesizer.load(self.opt['gan_file'])
        self.ctgan_syn.set_mode()
        
        self.boundary = self.load_boundary(self.opt['boundary_file'])
        self.line_svm = self.load_boundary(self.opt['svm_file'])


        def calculate_dis(X, latent_codes, boundary):
            dis_samples = []
            for index in range(0, len(X)):
                latent_sample = latent_codes[index : index + 1]
                dis_sample = (
                    np.sum(boundary['coef_'] * latent_sample) + boundary['intercept_']
                )
                dis_samples.append(dis_sample[0])
            return np.array(dis_samples)

        dis_samples = calculate_dis(self.X, self.latent_codes, self.boundary)
        self.dis_samples = dis_samples

    def clip(self, input):
        """
        Clip the generating instance with each feature to make sure it is valid
        :param input: generating instance
        :param conf: the configuration of dataset
        :return: a valid generating instance
        """
        for i in range(len(input)):
            input[i] = max(input[i], self.data_config.input_bounds[i][0])
            input[i] = min(input[i], self.data_config.input_bounds[i][1])
        return input

    def check_for_error_condition(self, t):
        """
        Check whether the test case is an individual discriminatory instance
        :param conf: the configuration of dataset
        :param sess: TF session
        :param x: input placeholder
        :param preds: the model's symbolic output
        :param t: test case
        :param sens: the index of sensitive feature
        :return: whether it is an individual discriminatory instance
        """
        t = t.astype("int")
        sens = np.array(self.opt['sens_param'].split("_")).astype('int')
        # label = model_argmax(self.sess, self.x, self.preds, np.array([t]))
        # print(t)
        # print(self.model(np.array([t])))
        label = (self.model(np.array([t]))>0.5).numpy().astype(int).flatten()[0]
        protected_domain = []
        sens_new = sens-1
        for i in sens_new:
            protected_domain = protected_domain + [list(range(int(self.data_config.input_bounds[i][0]), int(self.data_config.input_bounds[i][1])+1))]
        all_combs = np.array(list(itertools.product(*protected_domain)))
        for comb in all_combs:
            tnew = copy.deepcopy(t)
            for a, c in zip(sens_new, comb):
                tnew[a] = c
            label_new = (self.model(np.array([tnew]))>0.5).numpy().astype(int).flatten()[0]
            if label_new != label:
                self.n_value = comb
                return True
        return False

    def evaluate_global(self, inp):
        inp = self.clip(inp).astype("int")
        result, real_result = False, False
        temp = copy.deepcopy(inp.astype("int").tolist())
        # temp = temp[: self.opt['sens_param'] - 1] + temp[self.opt['sens_param'] :]
        sens = np.array(self.opt['sens_param'].split("_")).astype('int')
        for a in sens:
            temp[a - 1]=0
        if tuple(temp) not in tot_inputs:
            tot_inputs.add(tuple(temp))
            result = self.check_for_error_condition(inp)


        if result and (tuple(temp) not in global_disc_inputs):
            global_disc_inputs.add(tuple(temp))
            global_disc_inputs_list.append(copy.deepcopy(inp.astype("int").tolist()))
            real_result = True
        return real_result

    def latent_to_data(self, latents):
        fake_data = np.concatenate(latents, axis=0)  
        num_samples = len(fake_data)
        batch_size = self.ctgan_syn._batch_size
        N = int(num_samples / batch_size)
        if num_samples % batch_size != 0:
            N += 1
        data = []

        for ind in tqdm(range(N)):
            fakez = fake_data[ind * batch_size : (ind + 1) * batch_size]
            fakez = torch.Tensor(fakez).cuda()
            fake = self.ctgan_syn._generator(fakez)
            fakeact = self.ctgan_syn._apply_activate(fake, phase="generate")
            data.append(fakeact.detach().cpu().numpy())
        data = np.concatenate(data, axis=0)
        raw_data = self.ctgan_syn._transformer.inverse_transform(data).to_numpy()
        return raw_data

    def _count(self, data: np.ndarray):
        a_set = set()
        for item in data.tolist():
            # print(item)
            a_set.add(tuple(item))
        self.logger.info(
            f"item in data is {len(a_set)}, while len(data) is {len(data)}"
        )

    def global_phase_search(self):
        self.logger.info(
            f"iself.opt['max_global'] is {self.opt['max_global']}, len(self.X) is {len(self.X)}"
        )
        
        inputs = range(min(self.opt['max_global'], len(self.X)))
        global_latent_list = []

        # 0 1 -1
        self.cnt_candidate = {0: 0, 1: 0, -1: 0}
        candidate_dict = {0: [], 1: [], -1: []}
        step = opt['step']  # 0.3
        for num in tqdm(range(len(inputs))):
            index = inputs[num]
            latent_sample = self.latent_codes[index : index + 1]
            dis_record = self.dis_samples[index]

            sample_zero = latent_sample - dis_record * self.boundary['coef_']
            candidate_dict[0].append(sample_zero)

            # just perturb once
            latent_edit = sample_zero + step * self.boundary['coef_']
            candidate_dict[1].append(latent_edit)
            latent_edit = sample_zero - step * self.boundary['coef_']
            candidate_dict[-1].append(latent_edit)

        candidate_dict_data = {0: [], 1: [], -1: []}
        for perturb in [0, 1, -1]:
            tg_latent_edit_list = candidate_dict[perturb]
            raw_data = self.latent_to_data(latents=tg_latent_edit_list)
            self._count(raw_data)
            candidate_dict_data[perturb] = raw_data

        for ind in tqdm(range(0, len(candidate_dict_data[0]))):
            for perturb in [0, 1, -1]:
                if len(tot_inputs) >= self.opt['max_global']:
                    break
                g_inp_flag = self.evaluate_global(candidate_dict_data[perturb][ind])
                if g_inp_flag:
                    suc_idx.append((ind, perturb))
                    global_latent_list.append(candidate_dict[perturb][ind])
                    self.global_nvalue.append(self.n_value)
                    self.cnt_candidate[perturb] += 1
                    break
            _end = time.time()
            use_time = _end - self.start_time
            sec = len(self.count) * 300
            if use_time >= sec:
                self.logger.info(
                    "Percentage discriminatory inputs - "
                    + str(
                        float(len(global_disc_inputs_list))
                        / float(len(tot_inputs))
                        * 100
                    )
                )
                self.logger.info(
                    "Number of discriminatory inputs are "
                    + str(len(global_disc_inputs_list))
                )
                self.logger.info("Total Inputs are " + str(len(tot_inputs)))

                self.logger.info('use time:' + str(use_time) + "\n")
                self.count.append(1)
            if use_time >= 3600:
                break

        self.logger.info(f"the end ind is {ind}")
        self.tot_global_search = len(tot_inputs)
        numpy2log(global_disc_inputs_list, self.opt['global_samples_file'])
        numpy2log(self.global_nvalue, self.opt['disc_value_file'])

        self.logger.info(
            f"len(global_disc_inputs_list) is {len(global_disc_inputs_list)} in {self.tot_global_search} rate is {len(global_disc_inputs_list)/self.tot_global_search * 100:.2f}"
        )
        self.logger.info(f"self.cnt_candidate is {self.cnt_candidate}")

    def run(self):
        t1=time.time()
        self.global_phase_search()
        # self.logger.info("Total Inputs are " + str(len(tot_inputs)))
        # self.logger.info(
        #     f"Total discriminatory inputs of global search- {len(global_disc_inputs)} in {self.opt['max_global']}"
        # )
        t2=time.time()
        numpy2log(suc_idx, self.opt['suc_idx_file'])
        with open("quant_LIMI_result.csv","a+") as file:
            file.write("{},{},{},{}\n".format(self.opt['model_path'],self.opt['exp_name'],len(global_disc_inputs),t2-t1))



if __name__ == '__main__':
    opt, logger = _parse_args()

    fairtest = FairTest(opt, logger)
    start = time.time()
    fairtest.run()
    end = time.time()
    logger.info(f"Total time is {end-start}")
