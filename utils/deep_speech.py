
import numpy as np
import warnings
import resampy
from scipy.io import wavfile
from python_speech_features import mfcc
import tensorflow as tf


class DeepSpeech():
    def __init__(self,model_path):
        self.graph, self.logits_ph, self.input_node_ph, self.input_lengths_ph \
            = self._prepare_deepspeech_net(model_path)
        self.target_sample_rate = 16000

    def _prepare_deepspeech_net(self,deepspeech_pb_path):
        with tf.io.gfile.GFile(deepspeech_pb_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        graph = tf.compat.v1.get_default_graph()
        tf.import_graph_def(graph_def, name="deepspeech")
        logits_ph = graph.get_tensor_by_name("deepspeech/logits:0")
        input_node_ph = graph.get_tensor_by_name("deepspeech/input_node:0")
        input_lengths_ph = graph.get_tensor_by_name("deepspeech/input_lengths:0")

        return graph, logits_ph, input_node_ph, input_lengths_ph

    def conv_audio_to_deepspeech_input_vector(self,audio,
                                              sample_rate,
                                              num_cepstrum,
                                              num_context):
        # Get mfcc coefficients:
        features = mfcc(
            signal=audio,
            samplerate=sample_rate,
            numcep=num_cepstrum)

        # We only keep every second feature (BiRNN stride = 2):
        features = features[::2]

        # One stride per time step in the input:
        num_strides = len(features)

        # Add empty initial and final contexts:
        empty_context = np.zeros((num_context, num_cepstrum), dtype=features.dtype)
        features = np.concatenate((empty_context, features, empty_context))

        # Create a view into the array with overlapping strides of size
        # numcontext (past) + 1 (present) + numcontext (future):
        window_size = 2 * num_context + 1
        train_inputs = np.lib.stride_tricks.as_strided(
            features,
            shape=(num_strides, window_size, num_cepstrum),
            strides=(features.strides[0],
                     features.strides[0], features.strides[1]),
            writeable=False)

        # Flatten the second and third dimensions:
        train_inputs = np.reshape(train_inputs, [num_strides, -1])

        train_inputs = np.copy(train_inputs)
        train_inputs = (train_inputs - np.mean(train_inputs)) / \
                       np.std(train_inputs)

        return train_inputs

    def compute_audio_feature(self,audio_path):
        audio_sample_rate, audio = wavfile.read(audio_path)
        if audio.ndim != 1:
            warnings.warn(
                "Audio has multiple channels, the first channel is used")
            audio = audio[:, 0]
        if audio_sample_rate != self.target_sample_rate:
            resampled_audio = resampy.resample(
                x=audio.astype(np.float),
                sr_orig=audio_sample_rate,
                sr_new=self.target_sample_rate)
        else:
            resampled_audio = audio.astype(np.float)
        with tf.compat.v1.Session(graph=self.graph) as sess:
            input_vector = self.conv_audio_to_deepspeech_input_vector(
                audio=resampled_audio.astype(np.int16),
                sample_rate=self.target_sample_rate,
                num_cepstrum=26,
                num_context=9)
            network_output = sess.run(
                    self.logits_ph,
                    feed_dict={
                        self.input_node_ph: input_vector[np.newaxis, ...],
                        self.input_lengths_ph: [input_vector.shape[0]]})
            ds_features = network_output[::2,0,:]
        return ds_features

if __name__ == '__main__':
    audio_path = r'./00168.wav'
    model_path = r'./output_graph.pb'
    DSModel = DeepSpeech(model_path)
    ds_feature = DSModel.compute_audio_feature(audio_path)
    print(ds_feature)