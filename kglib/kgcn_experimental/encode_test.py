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

import numpy as np
from mock import Mock

from kglib.kgcn_experimental.encode import augment_data_fields
from kglib.kgcn_experimental.test.utils import get_call_args


class TestAugmentDataFields(unittest.TestCase):

    def test_numpy_fields_augmented_as_expected(self):
        data = [dict(attr1=np.array([0, 1, 0]), attr2=np.array([5]))]

        augment_data_fields(data, ('attr1', 'attr2'), 'features')

        expected_data = [dict(attr1=np.array([0, 1, 0]), attr2=np.array([5]), features=np.array([0, 1, 0, 5]))]

        np.testing.assert_equal(expected_data, data)

    def test_augmenting_non_numpy_numeric(self):
        data = [dict(attr1=np.array([0, 1, 0]), attr2=5)]

        augment_data_fields(data, ('attr1', 'attr2'), 'features')

        expected_data = [dict(attr1=np.array([0, 1, 0]), attr2=5, features=np.array([0, 1, 0, 5]))]

        np.testing.assert_equal(expected_data, data)


class TestAttributeEncoder(unittest.TestCase):
    def test_attribute_encoding_stages_are_as_expected(self):

        def op_mock():
            return Mock(return_value=np.array([0.121, 1.621, 1.437, -0.194, -0.216], dtype=np.float32))

        def attr_mock():
            return Mock(return_value=np.array([0.22632198, 0.29790161, 0.44993045], dtype=np.float32))

        encode = AttributeEncoder(5, 0, op=op_mock, attr_op=attr_mock)
        encoding = encode(np.array([2, 0.1234], dtype=np.float32))

        op_mock_call_args = get_call_args(op_mock)
        expected_intermediate_encoding = np.array([0, 0, 1, 0, 0, 0.22632198, 0.29790161, 0.44993045], dtype=np.float32)
        np.testing.assert_array_equal(op_mock_call_args, [[expected_intermediate_encoding]])

        attr_mock_call_args = get_call_args(attr_mock)
        expected_attribute_value = np.array([0.1234], dtype=np.float32)
        np.testing.assert_array_equal(attr_mock_call_args, [[expected_attribute_value]])

        expected_encoding = np.array([0.121, 1.621, 1.437, -0.194, -0.216], dtype=np.float32)
        np.testing.assert_array_equal(expected_encoding, encoding)


if __name__ == "__main__":
    unittest.main()
