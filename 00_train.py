"""
 @file   00_train.py
 @brief  Script for training
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
import gc
import time
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy as np
# from import
from tqdm import tqdm
# original lib
import common as com
import pytorch_model
from Dataset import MelDataLoader
import random
from torch.utils.tensorboard import SummaryWriter
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################


########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################


def list_to_vector_array(file_list,
                        cls_label,
                        cls_num,
                        msg="calc...",
                        n_mels=64,
                        frames=5,
                        n_fft=1024,
                        hop_length=512,
                        power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
    #for idx in range(len(file_list)):
        vector_array = com.file_to_vector_array(file_list[idx],
                                                n_mels=n_mels,
                                                frames=frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)

        data = vector_array.flatten()
        mean = np.mean(data, dtype=np.float32)
        std = np.std(data, dtype=np.float32)

        vector_array = (vector_array - mean) / std
        #print(vector_array)

        if idx == 0:
            features = np.zeros((vector_array.shape[0] * len(file_list), dims), dtype=np.float32)
        features[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

        del vector_array
        gc.collect()

    #m_label = np.empty((features.shape[0], cls_num), dtype=np.float32)
    #nm_label = np.empty((features.shape[0], cls_num), dtype=np.float32)
    m_label = np.zeros((features.shape[0], cls_num), dtype=np.float32)
    nm_label = np.zeros((features.shape[0], cls_num), dtype=np.float32)
    
    #m_label.fill(-1)
    #nm_label.fill(-1)

    for i in range(len(m_label)):
        nm_indices = []
        for j in range(cls_num):
            if j == cls_label:
                m_label[i][j] = 1
            else:
                nm_indices.append(j)

        nm_idx = random.choice(nm_indices)
        nm_label[i][nm_idx] = 1

    #print(label)
    dataset = [[features[i], m_label[i], nm_label[i]] for i in range(len(m_label))]
    return dataset
        
'''
def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list for training
    """
    com.logger.info("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")

    file_num = int(len(files) / 2) if len(files) > 2000 else len(files)
    random.shuffle(files)
    files = files[:file_num]
    com.logger.info("train_file num : {num}".format(num=len(files)))
    return files
'''

def file_list_generator(target_dir,
                             id_name,
                             dir_name="train",
                             prefix_normal="normal",
                             ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    ext : str (default="wav")
        file extension of audio files

    """
    com.logger.info("target_dir : {}".format(target_dir+"_"+id_name))

    # development
    #if mode:
    files = sorted(
        glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                dir_name=dir_name,
                                                                                prefix_normal=prefix_normal,
                                                                                id_name=id_name,
                                                                                ext=ext)))

    # files = numpy.concatenate((normal_files, anomaly_files), axis=0)
    # labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
    random.shuffle(files)
    files = files[:600]
    #files = files[:100]
    com.logger.info("train_file  num : {num}".format(num=len(files)))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")
    print("\n========================================")

    return files
########################################################################


########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)
        
    # make output directory
    os.makedirs(param["model_directory"]["idcae"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        '''
        model_file_path change to .pt
        '''
        
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.pt".format(model=param["model_directory"]["idcae"],
                                                                     machine_type=machine_type)
        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory"]["idcae"],
                                                                  machine_type=machine_type)

        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue
        
        writer = SummaryWriter()
        # generate dataset
        print("============== DATASET_GENERATOR ==============")

        machine_id_list = com.get_machine_id_list(target_dir, dir_name="train")

        for i in range(len(machine_id_list)):
            files = file_list_generator(target_dir, machine_id_list[i])

            sub_dataset = list_to_vector_array(files,
                                        cls_label=i,
                                        cls_num=len(machine_id_list),
                                        msg="generate train_dataset",
                                        n_mels=param["feature"]["idcae"]["n_mels"],
                                        frames=param["feature"]["idcae"]["frames"],
                                        n_fft=param["feature"]["idcae"]["n_fft"],
                                        hop_length=param["feature"]["idcae"]["hop_length"],
                                        power=param["feature"]["idcae"]["power"])


            if i == 0:
                dataset = sub_dataset
                #del sub_dataset
            else:
                dataset = np.concatenate((dataset, sub_dataset), axis=0)
                #del sub_dataset
        
        # train model
        print("============== MODEL TRAINING ==============")
        ########################################################################################
        # pytorch
        import torch.nn as nn
        import torch
        from torch.utils.data import random_split, DataLoader
        from pytorch_model import Net, CustomLoss
        ########################################################################################

        paramF = param["feature"]["idcae"]["frames"]
        paramM = param["feature"]["idcae"]["n_mels"]
        dim = paramF * paramM

        model = Net(paramF=paramF, paramM=paramM, classNum=len(machine_id_list))
        model.float()
        
        '''
        1. Dataset input to model
        2. Define optimizer and loss
        3. Validation
        '''  
        epochs = int(param["fit"]["idcae"]["epochs"])
        batch_size = int(param["fit"]["idcae"]["batch_size"])

        loss_function = CustomLoss(alpha=0.8, C=5, dim=dim, batch_size=batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

        
        val_split = param["fit"]["idcae"]["validation_split"]
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        train_dataset, valid_dataset = random_split(dataset, [train_size, val_size])
        train_batches = DataLoader(dataset=MelDataLoader(train_dataset), batch_size=batch_size, shuffle=True)
        val_batches = DataLoader(dataset=MelDataLoader(valid_dataset), batch_size=batch_size, shuffle=True)

        train_loss_list = []
        val_loss_list = []

        device = torch.device('cuda')
        model = model.to(device=device, dtype=torch.float32)

        # FIXME: encoder condition decoder training
        for epoch in range(1, epochs+1):
            train_loss = 0.0
            mls = 0.0
            nmls = 0.0
            val_loss = 0.0
            print("Epoch: {}".format(epoch))

            model.train()

            for feature_batch, label_batch, nm_label_batch in tqdm(train_batches):
                optimizer.zero_grad()
                #print(type(feature_batch))
                feature_batch = feature_batch.to(device, non_blocking=True, dtype=torch.float32)
                label_batch = label_batch.to(device, non_blocking=True, dtype=torch.float32)
                nm_label_batch = nm_label_batch.to(device, non_blocking=True, dtype=torch.float32)
                
                m_output, nm_output = model(feature_batch, label_batch, nm_label_batch)
                m_output = m_output.to(device, non_blocking=True, dtype=torch.float32)
                nm_output = nm_output.to(device, non_blocking=True, dtype=torch.float32)

                loss, _ , _ = loss_function(m_output, nm_output, feature_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                del m_output, nm_output, feature_batch, label_batch, nm_label_batch
                gc.collect()
                
            train_loss /= len(train_batches)
            train_loss_list.append(train_loss)

            model.eval()

            for feature_batch, label_batch, nm_label_batch in tqdm(val_batches):
                
                feature_batch = feature_batch.to(device, non_blocking=True, dtype=torch.float32)
                label_batch = label_batch.to(device, non_blocking=True, dtype=torch.float32)
                nm_label_batch = nm_label_batch.to(device, non_blocking=True, dtype=torch.float32)
                
                m_output, nm_output = model(feature_batch, label_batch, nm_label_batch)
                m_output = m_output.to(device, non_blocking=True, dtype=torch.float32)
                nm_output = nm_output.to(device, non_blocking=True, dtype=torch.float32)

                loss, ml, nml = loss_function(m_output, nm_output, feature_batch)
                val_loss += loss.item()
                mls += ml.item()
                nmls += nml.item()

                del m_output, nm_output, feature_batch, label_batch, nm_label_batch
                gc.collect()

            val_loss /= len(val_batches)
            mls /= len(val_batches)
            nmls /= len(val_batches)

            val_loss_list.append(val_loss)
            
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalars('comp/loss', {'train': train_loss, 'validation': val_loss}, epoch)
            writer.add_scalar('match loss', mls, epoch)
            writer.add_scalar('non match loss', nmls, epoch)

        visualizer.loss_plot(train_loss_list, val_loss_list)
        visualizer.save_figure(history_img)
        torch.save(model.state_dict(), model_file_path)
        com.logger.info("save_model -> {}".format(model_file_path))

        del dataset, train_batches, val_batches
        gc.collect()

        time.sleep(30)