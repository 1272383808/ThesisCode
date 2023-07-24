"""Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint."""

import argparse
import os
import numpy as np
import tensorflow as tf
import torch

from transformers import BertForSequenceClassification,BertModel


def convert_pytorch_checkpoint_to_tf(model: BertModel, ckpt_dir: str, model_name: str):

    """
    Args:
        model: BertModel Pytorch model instance to be converted
        ckpt_dir: Tensorflow model directory
        model_name: model name

    Currently supported HF models:

        - Y BertModel
        - N BertForMaskedLM
        - N BertForPreTraining
        - N BertForMultipleChoice
        - N BertForNextSentencePrediction
        - N BertForSequenceClassification
        - N BertForQuestionAnswering
    """

    tensors_to_transpose = ("dense.weight", "attention.self.query", "attention.self.key", "attention.self.value")

    var_map = (
        ("layer.", "layer_"),
        ("word_embeddings.weight", "word_embeddings"),
        ("position_embeddings.weight", "position_embeddings"),
        ("token_type_embeddings.weight", "token_type_embeddings"),
        (".", "/"),
        ("LayerNorm/weight", "LayerNorm/gamma"),
        ("LayerNorm/bias", "LayerNorm/beta"),
        ("weight", "kernel"),
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    state_dict = model.state_dict()

    def to_tf_var_name(name: str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)

        return "{}".format(name)

    def create_tf_var(tensor: np.ndarray, name: str, session: tf.compat.v1.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.compat.v1.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.compat.v1.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as session:
        for var_name in state_dict:
            print("var_name",var_name)
            tf_name = to_tf_var_name(var_name)
            ####增添开始########
            if tf_name == "classifier/kernel":
                tf_name = "output_weights"
            if tf_name == "classifier/bias":
                tf_name = "output_bias"
            ####增添结束########
            print("tf_name",tf_name)
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))

        saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_") + ".ckpt"))


def main(raw_args=None):
    #roberta_wwm_wxt_large_pretrain为Transformers库学到的pytoch_model.bin的模型文件夹  num_labels为类别数
    model = BertForSequenceClassification.from_pretrained(
            '../BERT_BASE_DIR/codebert-cpp',
            num_labels = 2
        )
    # ckpt_dir为转换后的bert_model.ckpt文件夹，model_name为ckpt的名字
    convert_pytorch_checkpoint_to_tf(model=model, ckpt_dir="../BERT_BASE_DIR/codebert-cpp/ckpt/", model_name="bert_model")


if __name__ == "__main__":
    main()