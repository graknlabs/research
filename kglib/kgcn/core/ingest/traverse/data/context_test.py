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

import unittest

import grakn

import kglib.kgcn.core.ingest.traverse.data.neighbour as neighbour
import kglib.kgcn.core.ingest.traverse.data.sample.sample as samp
import kglib.kgcn.core.ingest.traverse.data.context as context
import kglib.kgcn.core.ingest.traverse.data.sample.ordered as ordered
import kglib.kgcn.core.ingest.traverse.data.context_mocks as mocks


def _neighbourhood_traverser_factory(neighbour_sample_sizes):
    sampling_method = ordered.ordered_sample

    samplers = []
    for sample_size in neighbour_sample_sizes:
        samplers.append(samp.Sampler(sample_size, sampling_method, limit=sample_size * 2))

    context_builder = context.ContextBuilder(samplers)
    return context_builder


class TestContextBuilderFromEntity(unittest.TestCase):

    session = None

    @classmethod
    def setUpClass(cls):
        client = grakn.Grakn(uri="localhost:48555")
        cls.session = client.session(keyspace="test_schema")

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def setUp(self):
        self._tx = self.session.transaction(grakn.TxType.WRITE)

        # identifier = "Jacob J. Niesz"
        # entity_query = "match $x isa person, has identifier '{}'; get $x;".format(identifier)
        entity_query = "match $x isa person, has name 'Sundar Pichai'; get;"

        self._thing = neighbour.build_thing(list(self._tx.query(entity_query))[0].get('x'))

    def tearDown(self):
        self._tx.close()

    def _assert_types_correct(self, thing_context):
        """
        Check that all of the types in the structure are as expected
        :param thing_context:
        :return:
        """
        self.assertIsInstance(thing_context, context.ThingContext)
        self.assertIsInstance(thing_context.thing, neighbour.Thing)
        self.assertIn(type(thing_context.neighbourhood).__name__, ('list',))

        try:
            neighbour_role = thing_context.neighbourhood[0]

            self.assertIsInstance(neighbour_role, context.Neighbour)

            self.assertTrue(
                isinstance(neighbour_role.role_label, str)
                or neighbour_role.role_label in [neighbour.UNKNOWN_ROLE_TARGET_PLAYS_LABEL,
                                                 neighbour.UNKNOWN_ROLE_NEIGHBOUR_PLAYS_LABEL])
            self.assertIn(neighbour_role.role_direction, [neighbour.TARGET_PLAYS,
                                                          neighbour.NEIGHBOUR_PLAYS])
            self.assertTrue(self._assert_types_correct(neighbour_role.context))
        except IndexError:
            pass

        return True

    def _assert_depth_correct(self, thing_context):
        neighbour_role = next(thing_context.neighbourhood, None)
        if neighbour_role is not None:
            self._assert_depth_correct(neighbour_role.neighbour)

    def test_context_structure_types(self):
        data = ((1,), (2, 3), (2, 3, 4))
        for sample_sizes in data:
            with self.subTest(sample_sizes=str(data)):
                self._thing_context = _neighbourhood_traverser_factory(sample_sizes).build(self._tx, self._thing)
                self._assert_types_correct(self._thing_context)

    def test_context_check_depth(self):
        data = ((1,), (2, 3), (2, 3, 4))
        for sample_sizes in data:
            with self.subTest(sample_sizes=str(sample_sizes)):
                self._thing_context = _neighbourhood_traverser_factory(sample_sizes).build(self._tx, self._thing)

                collected_tree = context.collect_to_tree(self._thing_context)

                with self.subTest("Check number of immediate neighbours"):
                    self.assertEqual(len(collected_tree.neighbourhood), sample_sizes[0])
                with self.subTest("Check max depth of tree"):
                    self.assertEqual(len(sample_sizes), context.get_max_depth(self._thing_context))

    def test_context_is_deterministic(self):
        data = ((1,), (2, 3), (2, 3, 4))
        for sample_sizes in data:
            def to_test():
                return context.collect_to_tree(
                    _neighbourhood_traverser_factory(sample_sizes).build(self._tx, self._thing))

            with self.subTest(sample_sizes=str(data)):
                thing_context = to_test()

                for i in range(10):
                    new_thing_context = to_test()
                    self.assertEqual(new_thing_context, thing_context)


class TestIsolated(unittest.TestCase):
    session = None

    @classmethod
    def setUpClass(cls):
        client = grakn.Grakn(uri="localhost:48555")
        cls.session = client.session(keyspace="test_schema")

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def setUp(self):
        self._tx = self.session.transaction(grakn.TxType.WRITE)

    def tearDown(self):
        self._tx.close()

    def test_input_output(self):

        neighbour_sample_sizes = (2, 3)

        samplers = [lambda x: x for sample_size in neighbour_sample_sizes]

        starting_thing = neighbour.Thing("0", "person", "entity")

        context_builder = context.ContextBuilder(samplers, neighbour_finder=mocks.mock_neighbour_finder)

        thing_context = context_builder.build(self._tx, starting_thing)

        a, b = context.collect_to_tree(thing_context), context.collect_to_tree(mocks.mock_traversal_output())
        self.assertEqual(a, b)

    def test_input_output_integration(self):
        """
        Runs using real samplers
        :return:
        """

        sampling_method = ordered.ordered_sample

        samplers = [samp.Sampler(2, sampling_method, limit=2), samp.Sampler(3, sampling_method, limit=1)]

        starting_thing = neighbour.Thing("0", "person", "entity")

        context_builder = context.ContextBuilder(samplers, neighbour_finder=mocks.mock_neighbour_finder)

        thing_context = context_builder.build(self._tx, starting_thing)

        a, b = context.collect_to_tree(thing_context), context.collect_to_tree(mocks.mock_traversal_output())
        self.assertEqual(a, b)


class BaseTestFlattenedTree:
    class TestFlattenedTree(unittest.TestCase):
        def test_role_label_not_absent(self):
            role_label_absent = [f[0] in ['', None] for f in self._flattened[1:]]
            self.assertFalse(any(role_label_absent))

        def test_type_label_not_absent(self):
            type_label_absent = [f[2] in ['', None] for f in self._flattened[1:]]
            self.assertFalse(any(type_label_absent))

        def test_attribute_values_not_none(self):
            attribute_value_none = [f[3] == 'attribute' and f[-1] is None for f in self._flattened]
            self.assertFalse(any(attribute_value_none))

        def test_attribute_datatype_not_none(self):
            attribute_value_none = [f[3] == 'attribute' and f[-2] is None for f in self._flattened]
            self.assertFalse(any(attribute_value_none))


class TestIntegrationFlattened(BaseTestFlattenedTree.TestFlattenedTree):
    def setUp(self):
        entity_query = "match $x isa company, has name 'Google'; get;"
        uri = "localhost:48555"
        keyspace = "test_schema"
        client = grakn.Grakn(uri=uri)
        session = client.session(keyspace=keyspace)
        self._tx = session.transaction(grakn.TxType.WRITE)

        neighbour_sample_sizes = (4, 3)

        sampling_method = ordered.ordered_sample

        samplers = []
        for sample_size in neighbour_sample_sizes:
            samplers.append(samp.Sampler(sample_size, sampling_method, limit=sample_size * 2))

        grakn_things = [answermap.get('x') for answermap in list(self._tx.query(entity_query))]

        things = [neighbour.build_thing(grakn_thing) for grakn_thing in grakn_things]

        context_builder = context.ContextBuilder(samplers)

        self._neighbourhood_depths = [context_builder.build(self._tx, thing) for thing in things]

        self._neighbour_roles = context.convert_thing_contexts_to_neighbours(self._neighbourhood_depths)

        self._flattened = context.flatten_tree(self._neighbour_roles)


class TestIsolatedFlattened(BaseTestFlattenedTree.TestFlattenedTree):

    session = None

    @classmethod
    def setUpClass(cls):
        client = grakn.Grakn(uri="localhost:48555")
        cls.session = client.session(keyspace="test_schema")

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def tearDown(self):
        self._tx.close()

    def setUp(self):
        self._tx = self.session.transaction(grakn.TxType.WRITE)
        neighbour_sample_sizes = (2, 3)

        samplers = [lambda x: x for sample_size in neighbour_sample_sizes]

        starting_thing = neighbour.Thing("0", "person", "entity")
        things = [starting_thing]

        context_builder = context.ContextBuilder(samplers, neighbour_finder=mocks.mock_neighbour_finder)

        self._neighbourhood_depths = [context_builder.build(self._tx, thing) for thing in things]

        self._neighbour_roles = context.convert_thing_contexts_to_neighbours(self._neighbourhood_depths)

        self._flattened = context.flatten_tree(self._neighbour_roles)


if __name__ == "__main__":
    unittest.main()
