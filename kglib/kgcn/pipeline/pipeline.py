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

import networkx as nx
import numpy as np
from pathlib import Path
from graph_nets.utils_np import graphs_tuple_to_networkxs

from kglib.kgcn.learn.learn import KGCNLearner
from kglib.kgcn.models.core import softmax, KGCN
from kglib.kgcn.models.embedding import ThingEmbedder, RoleEmbedder
from kglib.kgcn.pipeline.encode import encode_types, create_input_graph, create_target_graph, encode_values
from kglib.kgcn.pipeline.utils import apply_logits_to_graphs, duplicate_edges_in_reverse
from kglib.kgcn.plot.plotting import plot_across_training, plot_predictions
from kglib.utils.graph.iterate import multidigraph_node_data_iterator, multidigraph_data_iterator, \
    multidigraph_edge_data_iterator


def pipeline(graphs,
             tr_ge_split,
             node_types,
             edge_types,
             num_processing_steps_tr=10,
             num_processing_steps_ge=10,
             num_training_iterations=10000,
             continuous_attributes=None,
             categorical_attributes=None,
             type_embedding_dim=5,
             attr_embedding_dim=6,
             edge_output_size=3,
             node_output_size=3,
             output_dir=None,
             do_test=False,
             save_fle="test_model.ckpt",
             reload_fle=""):

    ############################################################
    # Manipulate the graph data
    ############################################################

    # Encode attribute values
    graphs = [encode_values(graph, categorical_attributes, continuous_attributes) for graph in graphs]

    indexed_graphs = [nx.convert_node_labels_to_integers(graph, label_attribute='concept') for graph in graphs]
    graphs = [duplicate_edges_in_reverse(graph) for graph in indexed_graphs]

    graphs = [encode_types(graph, multidigraph_node_data_iterator, node_types) for graph in graphs]
    graphs = [encode_types(graph, multidigraph_edge_data_iterator, edge_types) for graph in graphs]

    input_graphs = [create_input_graph(graph) for graph in graphs]
    target_graphs = [create_target_graph(graph) for graph in graphs]

    tr_input_graphs = input_graphs[:tr_ge_split]
    tr_target_graphs = target_graphs[:tr_ge_split]
    ge_input_graphs = input_graphs[tr_ge_split:]
    ge_target_graphs = target_graphs[tr_ge_split:]

    ############################################################
    # Build and run the KGCN
    ############################################################

    thing_embedder = ThingEmbedder(node_types, type_embedding_dim, attr_embedding_dim, categorical_attributes,
                                   continuous_attributes)

    role_embedder = RoleEmbedder(len(edge_types), type_embedding_dim)

    kgcn = KGCN(thing_embedder,
                role_embedder,
                edge_output_size=edge_output_size,
                node_output_size=node_output_size)

    learner = KGCNLearner(kgcn,
                          num_processing_steps_tr=num_processing_steps_tr, # These processing steps indicate how many message-passing iterations to do for every training / testing step
                          num_processing_steps_ge=num_processing_steps_ge,
                          save_fle=f'{output_dir}/{save_fle}',
                          reload_fle=f'{output_dir}/{reload_fle}')

    # only test
    if not (Path(output_dir) / reload_fle).is_dir() and do_test is True:
        test_values, tr_info = learner.infer(ge_input_graphs,
                                            ge_target_graphs,
                                            log_dir=output_dir)
    # train
    else:
        train_values, test_values, tr_info = learner.train(input_graphs,
                                                 tr_target_graphs,
                                                 ge_input_graphs,
                                                 ge_target_graphs,
                                                 num_training_iterations=num_training_iterations,
                                                 log_dir=output_dir)


    plot_across_training(*tr_info, output_file=f'{output_dir}/learning.png')
    plot_predictions(graphs[tr_ge_split:], test_values, num_processing_steps_ge, output_file=f'{output_dir}/graph.png')

    logit_graphs = graphs_tuple_to_networkxs(test_values["outputs"][-1])

    indexed_ge_graphs = indexed_graphs[tr_ge_split:]
    ge_graphs = [apply_logits_to_graphs(graph, logit_graph) for graph, logit_graph in
                 zip(indexed_ge_graphs, logit_graphs)]

    for ge_graph in ge_graphs:
        for data in multidigraph_data_iterator(ge_graph):
            data['probabilities'] = softmax(data['logits'])
            data['prediction'] = int(np.argmax(data['probabilities']))

    _, _, _, _, _, solveds_tr, solveds_ge = tr_info
    return ge_graphs, solveds_tr, solveds_ge
