# comment definitions below are gathered from the internet, including but not limited to OpenAI ChatGpt

import tensorflow as tf
from text import symbols


class HParams(object):
    hparamdict = []

    def __init__(self, **hparams):
        self.hparamdict = hparams
        for k, v in hparams.items():
            setattr(self, k, v)
    def __repr__(self):
        return "HParams(" + repr([(k, v) for k, v in self.hparamdict.items()]) + ")"
    def __str__(self):
        return ','.join([(k + '=' + str(v)) for k, v in self.hparamdict.items()])
    def parse(self, params):
        for s in params.split(","):
            k, v = s.split("=", 1)
            k = k.strip()
            t = type(self.hparamdict[k])
            if t == bool:
                v = v.strip().lower()
                if v in ['true', '1']:
                    v = True
                elif v in ['false', '0']:
                    v = False
                else:
                    raise ValueError(v)
            else:
                v = t(v)
            self.hparamdict[k] = v
            setattr(self, k, v)
        return self

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=1000,
        iters_per_checkpoint=3000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="gloo",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='E:/AlanWatts/dataset/filelists/audio_text_train_filelist.txt',
        validation_files='E:/AlanWatts/dataset/filelists/audio_text_val_filelist.txt',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,  # specifies the number of units in each layer of the encoder

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,  # change to 10000 once trained for inference
        gate_threshold=0.5,

        p_attention_dropout=0.1,
        # change to 0.08 once almost trained (and 0.06 when loss levels off); 0.4 if needed for the alignment to form  # attention mechanism is used to align the text with the audio by selectively focusing on parts of the text that are relevant to the current audio frame
        p_decoder_dropout=0.1,
        # change to 0.08 once almost trained (and 0.06 when loss levels off); # num of steps the model takes to generate the output audio; decoding process involves a series of steps, where the model generates a new audio frame at each step based on the previous frames and the input text

        # Dropout is a commonly used technique in neural network models to prevent overfitting and improve generalization performance.
        # Decoder is responsible for generating the mel-spectrogram from the encoded audio features and the attention weights.
        # During training, the decoder dropout is applied to the output of the prenet layers and the output of the attention mechanism to regularize the model and reduce overfitting.
        # The amount of dropout applied during training is controlled by a hyperparameter called the dropout rate.
        # A dropout rate of 0.5 means that half of the units in the prenet and attention output are randomly dropped out during each training iteration.
        # This means that the remaining units must learn to work together and provide meaningful representations, leading to better generalization performance.
        # In general, a lower dropout rate (such as 0.1) may be more appropriate for larger datasets or more complex models, as these models are less likely to overfit and may benefit from less regularization.
        # Conversely, a higher dropout rate (such as 0.5) may be more appropriate for smaller datasets or simpler models, as these models are more prone to overfitting and may benefit from more regularization.
        # During inference, the dropout is turned off, and the model generates the mel-spectrogram without any regularization.

        # The relationship between attention and decoding steps is that the attention mechanism is used to align the text with the audio during the decoding process.
        # Attention helps the model to focus on the relevant parts of the input text and generate output audio that is aligned with the text.
        # However, increasing the number of decoding steps can improve the alignment of the text with the audio by giving the model more opportunities to generate new audio frames.
        # In practice, you can adjust the number of decoding steps and attention mechanism to find a balance between the quality of alignment and the computation time.
        # In general, a larger number of decoding steps will lead to better alignment, but will also increase the computation time.

        # Attention parameters
        attention_rnn_dim=1024,  # sets the number of units in the RNN
        attention_dim=128,  # sets the number of units in the attention mechanism
        #  These two values are relatively large and may require a significant amount of GPU memory during training and inference.

        # Location Layer parameters
        attention_location_n_filters=32,  # sets the number of filters in the CNN
        attention_location_kernel_size=31,  # sets the size of the filters
        # This means that the CNN has 32 filters and each filter has a kernel size of 31.
        # These values are also relatively large, which may require a significant amount of GPU memory during training and inference.

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        # When starting with a new dataset, it is usually a good idea to start with a higher learning rate (such as 1e-3) and then gradually reduce it if the training loss is not decreasing or if the validation loss starts to increase.
        # This approach allows you to quickly explore a wide range of learning rates and find the optimal value.
        # On the other hand, setting the learning rate too low (such as 1e-6) can make the training process very slow, as the model will take many epochs to converge.
        # This can also lead to the problem of getting stuck in local optima, where the model is not able to find the global minimum of the loss function.
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=32,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams