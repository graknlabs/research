#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
#

import numpy as np
import tensorflow as tf


def loss_ops_from_difference(target_op, output_ops, use_weighted=False):
    """
    Loss operation which directly compares the target with the output over all nodes and edges
    Args:
        target_op: The target of the model
        output_ops: A list of the outputs of the model, one for each message-passing step
        use_weighted: Boolean indicating if the loss should be weighted on the number of label prevalence (default: False)

    Returns: The loss for each message-passing step

    """
    if use_weighted:
        label_prevalence = tf.reduce_sum(target_op.nodes, axis=0)
        class_weights = tf.expand_dims(1 / (label_prevalence + 1), axis=0)
        # specify the weights for each sample in the batch (without having to compute the onehot label matrix)
        weights = tf.reduce_sum(class_weights * target_op.nodes, axis=1)
        loss_ops = [
            tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes, weights)
            for output_op in output_ops
        ]
    else:
        loss_ops = [
            tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes)
            for output_op in output_ops
        ]

    return loss_ops


def loss_ops_preexisting_no_penalty(target_op, output_ops, use_weighted=False):
    """
    Loss operation which doesn't penalise the output values for pre-existing nodes and edges, treating them as slack
    variables

    Args:
        target_op: The target of the model
        output_ops: A list of the outputs of the model, one for each message-passing step
        use_weighted: Boolean indicating if the loss should be weighted on the number of label prevalence (default: False)

    Returns: The loss for each message-passing step

    """
    loss_ops = []
    for output_op in output_ops:
        node_mask_op = tf.math.reduce_any(
            tf.math.not_equal(target_op.nodes, tf.constant(np.array([1., 0., 0.]), dtype=tf.float32)), axis=1)
        target_nodes = tf.boolean_mask(target_op.nodes, node_mask_op)
        output_nodes = tf.boolean_mask(output_op.nodes, node_mask_op)

        if use_weighted:
            label_prevalence = tf.reduce_sum(target_nodes, axis=0)
            class_weights = tf.expand_dims(1 / (label_prevalence + 1), axis=0)
            # specify the weights for each sample in the batch (without having to compute the onehot label matrix)
            weights = tf.reduce_sum(class_weights * target_nodes, axis=1)
            loss_op = tf.losses.softmax_cross_entropy(target_nodes, output_nodes, weights)
        else:
            loss_op = tf.losses.softmax_cross_entropy(target_nodes, output_nodes)


        loss_ops.append(loss_op)

    return loss_ops
