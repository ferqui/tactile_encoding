import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from utils.models import MN_neuron
from utils.utils import check_cuda, train_test_validation_split

def main():
    # Settings for the SNN
    global use_trainable_out
    use_trainable_out = False
    global use_trainable_tc
    use_trainable_tc = False
    global use_dropout
    use_dropout = False
    global batch_size
    batch_size = 512  # 128
    global lr
    lr = 0.0001

    # set u CUDA device
    device = check_cuda()

    def run_snn(inputs, layers):

        if use_trainable_out and use_trainable_tc:
            w1, w2, v1, alpha1, beta1, alpha2, beta2, out_scale, out_offset = layers
        elif use_trainable_tc:
            w1, w2, v1, alpha1, beta1, alpha2, beta2 = layers
        elif use_trainable_out:
            w1, w2, v1, out_scale, out_offset = layers
        else:
            w1, w2, v1 = layers
        if use_dropout:
            # using dropout on (n in %)/100 of spikes
            dropout = nn.Dropout(p=0.25)
        if use_trainable_tc:
            alpha1, beta1 = torch.abs(alpha1), torch.abs(beta1)
            alpha2, beta2 = torch.abs(alpha2), torch.abs(beta2)

        bs = inputs.shape[0]

        h1 = torch.einsum(
            "abc,cd->abd", (inputs.tile((nb_input_copies,)), w1))
        # h1 = torch.einsum(
        #     "abc,cd->abd", (inputs, w1))
        if use_dropout:
            h1 = dropout(h1)
        if use_trainable_tc:
            spk_rec, mem_rec = recurrent_layer.compute_activity_tc(
                bs, nb_hidden, h1, v1, alpha1, beta1, nb_steps)
        else:
            spk_rec, mem_rec = recurrent_layer.compute_activity(
                bs, nb_hidden, h1, v1, nb_steps)

        # Readout layer
        h2 = torch.einsum("abc,cd->abd", (spk_rec, w2))
        if use_dropout:
            h2 = dropout(h2)
        if use_trainable_tc:
            s_out_rec, out_rec = feedforward_layer.compute_activity_tc(
                bs, nb_outputs, h2, alpha2, beta2, nb_steps)
        else:
            s_out_rec, out_rec = feedforward_layer.compute_activity(
                bs, nb_outputs, h2, nb_steps)

        if use_trainable_out:
            # trainable output spike scaling
            # mean_firing_rate = torch.div(torch.sum(s_out_rec,1), s_out_rec.shape[1]) # mean firing rate
            # s_out_rec = mean_firing_rate*layers[5] + layers[6]
            s_out_rec = torch.sum(s_out_rec, 1)*out_scale + \
                out_offset  # sum spikes

        other_recs = [mem_rec, spk_rec, out_rec]
        layers_update = layers

        return s_out_rec, other_recs, layers_update

    def train(dataset, lr=0.0015, nb_epochs=300, opt_parameters=None, layers=None, dataset_test=None, break_early=False, patience=None):

        if (opt_parameters != None) & (layers != None):
            parameters = opt_parameters  # The paramters we want to optimize
            layers = layers
        elif (opt_parameters != None) & (layers == None):
            parameters = opt_parameters
            if use_trainable_out and use_trainable_tc:
                layers = [w1, w2, v1, alpha1, beta1,
                          alpha2, out_scale, out_offset]
            elif use_trainable_out:
                layers = [w1, w2, v1, out_scale, out_offset]
            elif use_trainable_tc:
                layers = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
            else:
                layers = [w1, w2, v1]
        elif (opt_parameters == None) & (layers != None):
            if use_trainable_out and use_trainable_tc:
                layers = [w1, w2, v1, alpha1, beta1, alpha2,
                          beta2, out_scale, out_offset]
            elif use_trainable_out:
                layers = [w1, w2, v1, out_scale, out_offset]
            elif use_trainable_tc:
                layers = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
            else:
                layers = [w1, w2, v1]
            layers = layers
        elif (opt_parameters == None) & (layers == None):
            if use_trainable_out and use_trainable_tc:
                parameters = [w1, w2, v1, alpha1, beta1, alpha2,
                              beta2, out_scale, out_offset]
                layers = [w1, w2, v1, alpha1, beta1, alpha2,
                          beta2, out_scale, out_offset]
            elif use_trainable_out:
                parameters = [w1, w2, v1, out_scale, out_offset]
                layers = [w1, w2, v1, out_scale, out_offset]
            elif use_trainable_tc:
                parameters = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
                layers = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
            else:
                parameters = [w1, w2, v1]
                layers = [w1, w2, v1]

        # The log softmax function across output units
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

        g = torch.Generator()
        g.manual_seed(seed)
        # generator = DataLoader(dataset, batch_size=batch_size, shuffle=True,
        #                     num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        # windows only works wth num_workers=0
        generator = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)

        # The optimization loop
        loss_hist = [[], []]
        accs_hist = [[], []]
        for e in range(nb_epochs):
            # learning rate decreases over epochs
            optimizer = torch.optim.Adamax(
                parameters, lr=lr, betas=(0.9, 0.995))
            # if e > nb_epochs*25:
            #     lr = lr * 0.98
            local_loss = []
            # accs: mean training accuracies for each batch
            accs = []
            for x_local, y_local in generator:
                x_local, y_local = x_local.to(
                    device, non_blocking=True), y_local.to(device, non_blocking=True)
                spks_out, recs, layers_update = run_snn(x_local, layers)
                # [mem_rec, spk_rec, out_rec]
                _, spk_rec, _ = recs

                # with output spikes
                if use_trainable_out:
                    m = spks_out
                else:
                    m = torch.sum(spks_out, 1)  # sum over time
                # cross entropy loss on the active read-out layer
                log_p_y = log_softmax_fn(m)

                # # L1 loss on total number of spikes (hidden layer)
                # reg_loss = 1e-3**torch.mean(torch.sum(spk_rec, 1))
                # # L2 loss on spikes per neuron (hidden layer)
                # reg_loss += 1e-6* \
                #     torch.mean(torch.sum(torch.sum(spk_rec, dim=0), dim=0)**2)
                # L1 loss on total number of spikes (hidden layer)
                reg_loss = 1e-4**torch.mean(torch.sum(spk_rec, 1))
                # L2 loss on spikes per neuron (hidden layer)
                reg_loss = reg_loss + 1e-8 * \
                    torch.mean(torch.sum(torch.sum(spk_rec, dim=0), dim=0)**2)

                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(log_p_y, y_local) + reg_loss

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                local_loss.append(loss_val.item())

                # compare to labels
                _, am = torch.max(m, 1)  # argmax over output units
                tmp = np.mean((y_local == am).detach().cpu().numpy())
                accs.append(tmp)

            mean_loss = np.mean(local_loss)
            loss_hist[0].append(mean_loss)

            # mean_accs: mean training accuracy of current epoch (average over all batches)
            mean_accs = np.mean(accs)
            accs_hist[0].append(mean_accs)

            # Calculate test accuracy in each epoch
            if dataset_test is not None:
                test_acc, test_loss = compute_classification_accuracy(
                    dataset_test,
                    layers=layers_update
                )
                accs_hist[1].append(test_acc)  # only safe best test
                loss_hist[1].append(test_loss)  # only safe loss of best test

            if dataset_test is None:
                # save best training
                if mean_accs >= np.max(accs_hist[0]):
                    best_acc_layers = []
                    for ii in layers_update:
                        best_acc_layers.append(ii.detach().clone())
            else:
                # save best test
                if np.max(test_acc) >= np.max(accs_hist[1]):
                    best_acc_layers = []
                    for ii in layers_update:
                        best_acc_layers.append(ii.detach().clone())

            print("Epoch {}/{} done. Train accuracy (loss): {:.2f}% ({:.5f}), Test accuracy (loss): {:.2f}% ({:.5f}).".format(
                e + 1, nb_epochs, accs_hist[0][-1]*100, loss_hist[0][-1], accs_hist[1][-1]*100, loss_hist[1][-1]))

            # check for early break
            if break_early:
                if e >= patience-1:
                    # mean acc drops
                    if np.mean(np.diff(accs_hist[1][-patience:]))*100 < -1.0:
                        print("\nmean(delta_test_acc): {:.2f} delta_test_acc: {}" .format(
                            np.mean(np.diff(accs_hist[1][-patience:]))*100, np.diff(accs_hist[1][-patience:])*100))
                        print("\nmean(delta_test_loss): {:.2f} delta_test_loss: {}" .format(
                            np.mean(np.diff(loss_hist[1][-patience:])*-1), np.diff(loss_hist[1][-patience:])*-1))
                        print(
                            f'\nBreaking the training early at episode {e+1}, test acc dropped.')
                        break
                    # mean acc static
                    elif abs(np.mean(np.diff(accs_hist[1][-patience:])))*100 < 1.0:
                        print("\nmean(delta_test_acc): {:.2f} delta_test_acc: {}" .format(
                            np.mean(np.diff(accs_hist[1][-patience:]))*100, np.diff(accs_hist[1][-patience:])*100))
                        print("\nmean(delta_test_loss): {:.2f} delta_test_loss: {}" .format(
                            np.mean(np.diff(loss_hist[1][-patience:])*-1), np.diff(loss_hist[1][-patience:])*-1))
                        print(
                            f'\nBreaking the training early at episode {e+1}, test acc static.')
                        break
                    # mean loss increases
                    elif np.mean(np.diff(loss_hist[1][-patience:])*-1) < 0.0:
                        print("\nmean(delta_test_acc): {:.2f} delta_test_acc: {}" .format(
                            np.mean(np.diff(accs_hist[1][-patience:]))*100, np.diff(accs_hist[1][-patience:])*100))
                        print("\nmean(delta_test_loss): {:.2f} delta_test_loss: {}" .format(
                            np.mean(np.diff(loss_hist[1][-patience:])*-1), np.diff(loss_hist[1][-patience:])*-1))
                        print(
                            f'\nBreaking the training early at episode {e+1}, test loss increasing.')
                        break
                    # mean loss static
                    elif abs(np.mean(np.diff(loss_hist[1][-patience:])*-1)) < 1.0:
                        print("\nmean(delta_test_acc): {:.2f} delta_test_acc: {}" .format(
                            np.mean(np.diff(loss_hist[1][-patience:])*-1)*100, np.diff(loss_hist[1][-patience:])*-1*100))
                        print("\nmean(delta_test_loss): {:.2f} delta_test_loss: {}" .format(
                            np.mean(np.diff(loss_hist[1][-patience:])*-1), np.diff(loss_hist[1][-patience:])*-1))
                        print(
                            f'\nBreaking the training early at episode {e+1}, test loss static.')
                        break

        return loss_hist, accs_hist, best_acc_layers

    def build_and_train(data_steps, ds_train, ds_test, epochs=300, break_early=False, patience=None):

        global nb_input_copies
        # Num of spiking neurons used to encode each channel
        nb_input_copies = 1  # params['nb_input_copies']

        # Network parameters
        global nb_inputs
        nb_inputs = 24*nb_input_copies
        global nb_outputs
        nb_outputs = len(np.unique(labels))
        global nb_hidden
        nb_hidden = 450
        global nb_steps
        nb_steps = data_steps

        tau_mem = 20e-3  # ms
        tau_syn = tau_mem/2

        if not use_trainable_tc:
            global alpha
            global beta
        alpha = torch.as_tensor(float(np.exp(-dt/tau_syn)))
        beta = torch.as_tensor(float(np.exp(-dt/tau_mem)))

        fwd_weight_scale = 3.0
        rec_weight_scale = 1e-2*fwd_weight_scale

        # Spiking network
        layers = []

        # recurrent layer
        w1, v1 = recurrent_layer.create_layer(
            nb_inputs, nb_hidden, fwd_weight_scale, rec_weight_scale)

        # readout layer
        w2 = feedforward_layer.create_layer(
            nb_hidden, nb_outputs, fwd_weight_scale)

        if use_trainable_tc:
            # time constants
            alpha1, beta1 = trainable_time_constants.create_time_constants(
                nb_hidden, alpha, beta, use_trainable_tc)

            alpha2, beta2 = trainable_time_constants.create_time_constants(
                nb_outputs, alpha, beta, use_trainable_tc)

        layers.append(w1), layers.append(w2), layers.append(v1)
        if use_trainable_tc:
            layers.append(alpha1), layers.append(
                beta1), layers.append(alpha2), layers.append(beta2)

        if use_trainable_out:
            # include trainable output for readout layer (linear: y = out_scale * x + out_offset)
            out_scale = torch.empty(
                (nb_outputs),  device=device, dtype=torch.float, requires_grad=True)
            torch.nn.init.ones_(out_scale)
            layers.append(out_scale)
            out_offset = torch.empty(
                (nb_outputs),  device=device, dtype=torch.float, requires_grad=True)
            torch.nn.init.zeros_(out_offset)
            layers.append(out_offset)

        layers_init = []
        for ii in layers:
            layers_init.append(ii.detach().clone())

        if use_trainable_out and use_trainable_tc:
            opt_parameters = [w1, w2, v1, alpha1, beta1,
                              alpha2, beta2, out_scale, out_offset]
        elif use_trainable_tc:
            opt_parameters = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
        elif use_trainable_out:
            opt_parameters = [w1, w2, v1, out_scale, out_offset]
        else:
            opt_parameters = [w1, w2, v1]

        # a fixed learning rate is already defined within the train function, that's why here it is omitted
        loss_hist, accs_hist, best_layers = train(
            ds_train, lr=lr, nb_epochs=epochs, opt_parameters=opt_parameters, layers=layers, dataset_test=ds_test, break_early=break_early, patience=patience)

        # best training and test at best training
        acc_best_train = np.max(accs_hist[0])  # returns max value
        acc_best_train = acc_best_train*100
        idx_best_train = np.argmax(accs_hist[0])  # returns index of max value
        acc_test_at_best_train = accs_hist[1][idx_best_train]*100

        # best test and training at best test
        acc_best_test = np.max(accs_hist[1])
        acc_best_test = acc_best_test*100
        idx_best_test = np.argmax(accs_hist[1])
        acc_train_at_best_test = accs_hist[0][idx_best_test]*100

        print(
            "\n------------------------------------------------------------------------------------")
        print("Final results: ")
        print("Best training accuracy: {:.2f}% and according test accuracy: {:.2f}% at epoch: {}".format(
            acc_best_train, acc_test_at_best_train, idx_best_train+1))
        print("Best test accuracy: {:.2f}% and according train accuracy: {:.2f}% at epoch: {}".format(
            acc_best_test, acc_train_at_best_test, idx_best_test+1))
        print(
            "------------------------------------------------------------------------------------")
        print(
            "------------------------------------------------------------------------------------\n")
        return loss_hist, accs_hist, best_layers

    def compute_classification_accuracy(dataset, layers=None):
        """ Computes classification accuracy on supplied data in batches. """

        # generator = DataLoader(dataset, batch_size=batch_size,
        #                     shuffle=False, num_workers=4, pin_memory=True)
        generator = DataLoader(dataset, batch_size=batch_size,
                               shuffle=False, num_workers=0, pin_memory=True)
        accs = []
        losss = []
        # The log softmax function across output units
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

        for x_local, y_local in generator:
            x_local, y_local = x_local.to(
                device, non_blocking=True), y_local.to(device, non_blocking=True)
            if layers == None:
                if use_trainable_out and use_trainable_tc:
                    layers = [w1, w2, v1, alpha1, beta1, alpha2,
                              beta2, out_scale, out_offset]
                elif use_trainable_out:
                    layers = [w1, w2, v1, out_scale, out_offset]
                elif use_trainable_tc:
                    layers = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
                else:
                    layers = [w1, w2, v1]
                spks_out, _, _ = run_snn(x_local, layers)
            else:
                spks_out, _, _ = run_snn(x_local, layers)
            # with output spikes
            if use_trainable_out:
                m = spks_out
            else:
                m = torch.sum(spks_out, 1)  # sum over time
            _, am = torch.max(m, 1)     # argmax over output units
            # compute test loss
            log_p_y = log_softmax_fn(m)
            loss_val = loss_fn(log_p_y, y_local).detach().cpu().numpy()
            losss.append(loss_val)
            # compute acc
            tmp = np.mean((y_local == am).detach().cpu().numpy())
            accs.append(tmp)

        return np.mean(accs), np.mean(losss)

    def ConfusionMatrix(dataset, save, layers=None, labels=None):

        g = torch.Generator()
        g.manual_seed(seed)
        # generator = DataLoader(dataset, batch_size=batch_size, shuffle=True,
        #                     num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        generator = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        accs = []
        trues = []
        preds = []
        for x_local, y_local in generator:
            x_local, y_local = x_local.to(
                device, non_blocking=True), y_local.to(device, non_blocking=True)
            if layers == None:
                if use_trainable_out and use_trainable_tc:
                    layers = [w1, w2, v1, alpha1, beta1, alpha2,
                              beta2, out_scale, out_offset]
                elif use_trainable_out:
                    layers = [w1, w2, v1, out_scale, out_offset]
                elif use_trainable_tc:
                    layers = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
                else:
                    layers = [w1, w2, v1]
                spks_out, _, _ = run_snn(x_local, layers)
            else:
                spks_out, _, _ = run_snn(x_local, layers)
            # with output spikes
            if use_trainable_out:
                m = spks_out
            else:
                m = torch.sum(spks_out, 1)  # sum over time
            _, am = torch.max(m, 1)     # argmax over output units
            # compare to labels
            tmp = np.mean((y_local == am).detach().cpu().numpy())
            accs.append(tmp)
            trues.extend(y_local.detach().cpu().numpy())
            preds.extend(am.detach().cpu().numpy())

        print("Accuracy from Confusion Matrix: {:.2f}% +- {:.2f}%".format(np.mean(accs)
                                                                                 * 100, np.std(accs)*100))

        cm = confusion_matrix(trues, preds, normalize='true')
        cm_df = pd.DataFrame(cm, index=[ii for ii in labels], columns=[
            jj for jj in labels])
        plt.figure("cm", figsize=(12, 9))
        sn.heatmap(cm_df,
                   annot=True,
                   fmt='.1g',
                   cbar=False,
                   square=False,
                   cmap="YlGnBu")
        plt.xlabel('\nPredicted')
        plt.ylabel('True\n')
        plt.xticks(rotation=0)
        if save:
            path_to_save_fig = f'{path_for_plots}/generation_{generation+1}_individual_{best_individual+1}'
            if use_trainable_tc:
                path_to_save_fig = f'{path_to_save_fig}_train_tc'
            if use_trainable_out:
                path_to_save_fig = f'{path_to_save_fig}_train_out'
            if use_dropout:
                path_to_save_fig = f'{path_to_save_fig}_dropout'
            path_to_save_fig = f'{path_to_save_fig}_cm.png'
            plt.savefig(path_to_save_fig, dpi=300)
            plt.close()
        else:
            plt.show()

    def NetworkActivity(dataset, save, layers=None, labels=None):

        g = torch.Generator()
        g.manual_seed(seed)
        # generator = DataLoader(dataset, batch_size=batch_size, shuffle=True,
        #                     num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        generator = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)

        accs = []
        trues = []
        preds = []
        for x_local, y_local in generator:
            x_local, y_local = x_local.to(
                device, non_blocking=True), y_local.to(device, non_blocking=True)
            if layers == None:
                if use_trainable_out and use_trainable_tc:
                    layers = [w1, w2, v1, alpha1, beta1, alpha2,
                              beta2, out_scale, out_offset]
                elif use_trainable_out:
                    layers = [w1, w2, v1, out_scale, out_offset]
                elif use_trainable_tc:
                    layers = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
                else:
                    layers = [w1, w2, v1]
                spks_out, recs, _ = run_snn(x_local, layers)
            else:
                spks_out, recs, _ = run_snn(x_local, layers)

            # [mem_rec, spk_rec, out_rec]
            _, spk_rec, _ = recs

        nb_plt = 4
        gs = GridSpec(1, nb_plt)

        # hidden layer
        fig = plt.figure("hidden layer", figsize=(8, 6), dpi=300)
        plt.title("Hidden layer 1")
        for i in range(nb_plt):
            plt.subplot(gs[i])
            plt.imshow(spk_rec[i].detach().cpu().numpy().T,
                       cmap=plt.cm.gray_r, origin="lower")
            if i == 0:
                plt.xlabel("Time")
                plt.ylabel("Units")
            sn.despine()
        if save:
            path_to_save_fig = f'{path_for_plots}/generation_{generation+1}_individual_{best_individual+1}'
            if use_trainable_tc:
                path_to_save_fig = f'{path_to_save_fig}_train_tc'
            if use_trainable_out:
                path_to_save_fig = f'{path_to_save_fig}_train_out'
            if use_dropout:
                path_to_save_fig = f'{path_to_save_fig}_dropout'
            path_to_save_fig = f'{path_to_save_fig}_hidden_layer.png'
            plt.savefig(path_to_save_fig, dpi=300)
            plt.close()

        # output layer
        fig = plt.figure("output layer", figsize=(8, 6), dpi=300)
        plt.title("Output layer")
        for i in range(nb_plt):
            plt.subplot(gs[i])
            plt.imshow(spks_out[i].detach().cpu().numpy().T,
                       cmap=plt.cm.gray_r, origin="lower")
            if i == 0:
                plt.xlabel("Time")
                plt.ylabel("Units")
            sn.despine()
        if save:
            path_to_save_fig = f'{path_for_plots}/generation_{generation+1}_individual_{best_individual+1}'
            if use_trainable_tc:
                path_to_save_fig = f'{path_to_save_fig}_train_tc'
            if use_trainable_out:
                path_to_save_fig = f'{path_to_save_fig}_train_out'
            if use_dropout:
                path_to_save_fig = f'{path_to_save_fig}_dropout'
            path_to_save_fig = f'{path_to_save_fig}_output_layer.png'
            plt.savefig(path_to_save_fig, dpi=300)
            plt.close()
        else:
            plt.show()

    class SurrGradSpike(torch.autograd.Function):
        """
        Here we implement our spiking nonlinearity which also implements 
        the surrogate gradient. By subclassing torch.autograd.Function, 
        we will be able to use all of PyTorch's autograd functionality.
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """

        scale = 10

        @staticmethod
        def forward(ctx, input):
            """
            In the forward pass we compute a step function of the input Tensor
            and return it. ctx is a context object that we use to stash information which 
            we need to later backpropagate our error signals. To achieve this we use the 
            ctx.save_for_backward method.
            """
            ctx.save_for_backward(input)
            out = torch.zeros_like(input)
            out[input > 0] = 1.0
            return out

        @staticmethod
        def backward(ctx, grad_output):
            """
            In the backward pass we receive a Tensor we need to compute the 
            surrogate gradient of the loss with respect to the input. 
            Here we use the normalized negative part of a fast sigmoid 
            as this was done in Zenke & Ganguli (2018).
            """
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
            return grad

    spike_fn = SurrGradSpike.apply

    class feedforward_layer:
        '''
        class to initialize and compute spiking feedforward layer
        '''
        def create_layer(nb_inputs, nb_outputs, scale):
            ff_layer = torch.empty(
                (nb_inputs, nb_outputs),  device=device, dtype=torch.float, requires_grad=True)
            torch.nn.init.normal_(ff_layer, mean=0.0,
                                  std=scale/np.sqrt(nb_inputs))
            return ff_layer

        def compute_activity(nb_input, nb_neurons, input_activity, nb_steps):
            syn = torch.zeros((nb_input, nb_neurons),
                              device=device, dtype=torch.float)
            mem = torch.zeros((nb_input, nb_neurons),
                              device=device, dtype=torch.float)
            out = torch.zeros((nb_input, nb_neurons),
                              device=device, dtype=torch.float)
            mem_rec = []
            spk_rec = []

            # Compute feedforward layer activity
            for t in range(nb_steps):
                mthr = mem-1.0
                out = spike_fn(mthr)
                rst_out = out.detach()

                new_syn = alpha*syn + input_activity[:, t]
                new_mem = (beta*mem + syn)*(1.0-rst_out)

                mem_rec.append(mem)
                spk_rec.append(out)

                mem = new_mem
                syn = new_syn

            # Now we merge the recorded membrane potentials into a single tensor
            mem_rec = torch.stack(mem_rec, dim=1)
            spk_rec = torch.stack(spk_rec, dim=1)
            return spk_rec, mem_rec

        def compute_activity_tc(nb_input, nb_neurons, input_activity, alpha, beta, nb_steps):
            syn = torch.zeros((nb_input, nb_neurons),
                              device=device, dtype=torch.float)
            mem = torch.zeros((nb_input, nb_neurons),
                              device=device, dtype=torch.float)
            out = torch.zeros((nb_input, nb_neurons),
                              device=device, dtype=torch.float)
            mem_rec = []
            spk_rec = []

            # Compute feedforward layer activity
            for t in range(nb_steps):
                mthr = mem-1.0
                out = spike_fn(mthr)
                rst_out = out.detach()

                new_syn = torch.abs(alpha)*syn + input_activity[:, t]
                new_mem = (torch.abs(beta)*mem + syn)*(1.0-rst_out)

                mem_rec.append(mem)
                spk_rec.append(out)

                mem = new_mem
                syn = new_syn

            # Now we merge the recorded membrane potentials into a single tensor
            mem_rec = torch.stack(mem_rec, dim=1)
            spk_rec = torch.stack(spk_rec, dim=1)
            return spk_rec, mem_rec

    class recurrent_layer:
        '''
        class to initialize and compute spiking recurrent layer
        '''
        def create_layer(nb_inputs, nb_outputs, fwd_scale, rec_scale):
            ff_layer = torch.empty(
                (nb_inputs, nb_outputs),  device=device, dtype=torch.float, requires_grad=True)
            torch.nn.init.normal_(ff_layer, mean=0.0,
                                  std=fwd_scale/np.sqrt(nb_inputs))

            rec_layer = torch.empty(
                (nb_outputs, nb_outputs),  device=device, dtype=torch.float, requires_grad=True)
            torch.nn.init.normal_(rec_layer, mean=0.0,
                                  std=rec_scale/np.sqrt(nb_inputs))
            return ff_layer,  rec_layer

        def compute_activity(nb_input, nb_neurons, input_activity, layer, nb_steps):
            syn = torch.zeros((nb_input, nb_neurons),
                              device=device, dtype=torch.float)
            mem = torch.zeros((nb_input, nb_neurons),
                              device=device, dtype=torch.float)
            out = torch.zeros((nb_input, nb_neurons),
                              device=device, dtype=torch.float)
            mem_rec = []
            spk_rec = []

            # Compute recurrent layer activity
            for t in range(nb_steps):
                # input activity plus last step output activity
                h1 = input_activity[:, t] + \
                    torch.einsum("ab,bc->ac", (out, layer))
                mthr = mem-1.0
                out = spike_fn(mthr)
                rst = out.detach()  # We do not want to backprop through the reset

                new_syn = alpha*syn + h1
                new_mem = (beta*mem + syn)*(1.0-rst)

                mem_rec.append(mem)
                spk_rec.append(out)

                mem = new_mem
                syn = new_syn

            # Now we merge the recorded membrane potentials into a single tensor
            mem_rec = torch.stack(mem_rec, dim=1)
            spk_rec = torch.stack(spk_rec, dim=1)
            return spk_rec, mem_rec

        def compute_activity_tc(nb_input, nb_neurons, input_activity, layer, alpha, beta, nb_steps):
            syn = torch.zeros((nb_input, nb_neurons),
                              device=device, dtype=torch.float)
            mem = torch.zeros((nb_input, nb_neurons),
                              device=device, dtype=torch.float)
            out = torch.zeros((nb_input, nb_neurons),
                              device=device, dtype=torch.float)
            mem_rec = []
            spk_rec = []

            # Compute recurrent layer activity
            for t in range(nb_steps):
                # input activity plus last step output activity
                h1 = input_activity[:, t] + \
                    torch.einsum("ab,bc->ac", (out, layer))
                mthr = mem-1.0
                out = spike_fn(mthr)
                rst = out.detach()  # We do not want to backprop through the reset

                new_syn = torch.abs(alpha)*syn + h1
                new_mem = (torch.abs(beta)*mem + syn)*(1.0-rst)

                mem_rec.append(mem)
                spk_rec.append(out)

                mem = new_mem
                syn = new_syn

            # Now we merge the recorded membrane potentials into a single tensor
            mem_rec = torch.stack(mem_rec, dim=1)
            spk_rec = torch.stack(spk_rec, dim=1)
            return spk_rec, mem_rec

    class trainable_time_constants:
        def create_time_constants(nb_neurons, alpha_mean, beta_mean, trainable):
            alpha = torch.empty((nb_neurons),  device=device,
                                dtype=torch.float, requires_grad=trainable)
            torch.nn.init.normal_(
                alpha, mean=alpha_mean, std=alpha_mean/10)

            beta = torch.empty((nb_neurons),  device=device,
                               dtype=torch.float, requires_grad=trainable)
            torch.nn.init.normal_(
                beta, mean=beta_mean, std=beta_mean/10)
            return alpha, beta


    # create train test validation split
    x_train, y_train, x_test, y_test, x_validation, y_validation = train_test_validation_split(encoded_data, encoded_label, split=[70, 20, 10])


if __name__ == '__main__':
    main()

