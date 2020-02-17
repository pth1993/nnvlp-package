#!/usr/bin/env python
# -*- coding: utf-8 -*-
from unittest import TestCase
import nnvlp


class TestNNVLP(TestCase):
    def test_download(self):
        nnvlp.download()

    def test_output(self):
        model = nnvlp.NNVLP()
        output = model.predict(u'Ông Nam là giảng viên trường Bách Khoa.', display_format='JSON')
        print(output)
