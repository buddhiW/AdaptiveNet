"""
Author: Buddhi Wickramasinghe
Frame speech signals and save them along with labels
"""
import torch
from torch.autograd import Variable
from conf import * #This will import all the variables in the cfg file
import pdb
import soundfile as sf
import scipy.io

def create_framed_signal(idx, data_folder, wav_lst, N_snt, wlen, lab_dict, device, num_frames):

    [signal, fs] = sf.read(data_folder + wav_lst[idx])
    print(wav_lst[idx])

    shiftLen = wlen # No overlapping
    nsubFrames = num_frames
    sig_matrix = np.zeros([nsubFrames - 1, wlen+cnn_len_filt[0]-2])

    prevFrame = np.zeros([cnn_len_filt[0]-2,])

    # Framing the selected signal
    for k in range(num_frames - 1):

        snt_beg = k * shiftLen
        snt_end = snt_beg + wlen

        if snt_end > signal.shape[0]:
            signal = np.concatenate((signal,signal))

        subSignal = signal[snt_beg:snt_end]
        subSignalNew = np.concatenate((prevFrame,subSignal))
        sig_matrix[k, :] = np.squeeze(subSignalNew)
        y = lab_dict[wav_lst[idx]]

        ## to compensate for edge effects
        prevFrame = subSignal[wlen-cnn_len_filt[0]+2:wlen+1]

    inp = Variable(torch.from_numpy(sig_matrix).double().to(device).contiguous())

    return inp.unsqueeze(0), y

if __name__ == "__main__":

    wlen=int(fs*cw_len/1000.00)
    device = torch.device(device)

    labels_trainSel = np.zeros(((snt_tr), 1), dtype=np.int)
    labels_devSel = np.zeros(((snt_te),1), dtype=np.int)

    # Process training data
    print('Training set ...')
    for i in range(snt_tr):
        print(i)
        [FramedData, targetT] = create_framed_signal(i, data_folder, wav_lst_tr, snt_tr, wlen, lab_dictTr, device,
                                                  num_frames)
        labels_trainSel[i] = targetT
        dataEval = FramedData.squeeze().data.cpu().numpy()
        np.save('data/TrainSel/' + 'tr' + str(i) + '.npy', dataEval)
    scipy.io.savemat('labels_trainSel.mat', {'labels_trainSel': labels_trainSel})

    # Process development data
    print('Development set ...')
    for i in range(snt_te):
        print(i)
        [FramedData, targetT] = create_framed_signal(i, data_folder, wav_lst_te, snt_te, wlen, lab_dictTe, device, num_frames)

        labels_devSel[i] = targetT
        dataEval = FramedData.squeeze().data.cpu().numpy()
        np.save('data/DevSel/'+'eval'+str(i)+'.npy',dataEval)
    scipy.io.savemat('labels_devSel.mat', {'labels_devSel': labels_devSel})
