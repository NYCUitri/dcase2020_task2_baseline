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
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
# from import
from tqdm import tqdm
# original lib
import common as com
import pytorch_model
import random
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
    #for idx in tqdm(range(len(file_list)), desc=msg):
    for idx in range(len(file_list)):
        vector_array = com.file_to_vector_array(file_list[idx],
                                                n_mels=n_mels,
                                                frames=frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset


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
    os.makedirs(param["model_directory"], exist_ok=True)

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
        model_file_path = "{model}/model_{machine_type}.pt".format(model=param["model_directory"],
                                                                     machine_type=machine_type)
        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory"],
                                                                  machine_type=machine_type)

        # if os.path.exists(model_file_path):
        #     com.logger.info("model exists")
        #     continue

        # generate dataset
        print("============== DATASET_GENERATOR ==============")
        files = file_list_generator(target_dir)
        train_data = list_to_vector_array(files,
                                          msg="generate train_dataset",
                                          n_mels=param["feature"]["n_mels"],
                                          frames=param["feature"]["frames"],
                                          n_fft=param["feature"]["n_fft"],
                                          hop_length=param["feature"]["hop_length"],
                                          power=param["feature"]["power"])

        # train model
        print("============== MODEL TRAINING ==============")
        ########################################################################################
        # pytorch
        import torch.nn as nn
        import torch
        from torch.utils.data import DataLoader, random_split
        from pytorch_model import Net
        ########################################################################################
        inputDim = param["feature"]["n_mels"] * param["feature"]["frames"]
        paramF = param["feature"]["frames"]
        paramM = param["feature"]["n_mels"]

        model = Net(paramF, paramM)
        model.double()
        '''
        1. Dataset input to model
        2. Define optimizer and loss
        3. Validation
        '''
        
        loss_function = nn.CrossEntropyLoss()
        # loss_function = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        epochs = int(param["fit"]["epochs"])
        batch_size = int(param["fit"]["batch_size"])

        #data_size = len(train_data) if len(train_data) < 1500 * batch_size else 1500 * batch_size
        #train_data = train_data[:data_size]
        val_split = param["fit"]["validation_split"]
        val_size = int(len(train_data) * val_split)
        train_size = len(train_data) - val_size

        train_dataset, valid_dataset = random_split(train_data, [train_size, val_size])
        train_batches = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=param["fit"]["shuffle"])
        val_batches = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=param["fit"]["shuffle"])

        train_loss_list = []
        val_loss_list = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device=device, dtype=torch.double, non_blocking=True)

        for epoch in range(1, epochs+1):
            train_loss = 0.0
            val_loss = 0.0
            print("Epoch: {}".format(epoch))

            model.train()

            # FIXME: calculate loss
            for batch in tqdm(train_batches):
                optimizer.zero_grad()
                batch = batch.to(device, non_blocking=True)
                reconstructed = model(batch).to(device, non_blocking=True)

                loss = loss_function(reconstructed, )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_batches)
            train_loss_list.append(train_loss)

            model.eval()
            for batch in tqdm(val_batches):
                batch = batch.to(device, non_blocking=True)
                output = model(batch).to(device, non_blocking=True)
                loss = loss_function(output, batch)
                val_loss += loss.item()

            val_loss /= len(val_batches)
            val_loss_list.append(val_loss)
        
        visualizer.loss_plot(train_loss_list, val_loss_list)
        visualizer.save_figure(history_img)
        torch.save(model.state_dict(), model_file_path)
        com.logger.info("save_model -> {}".format(model_file_path))

        del train_data, train_batches, val_batches
        import gc
        gc.collect()