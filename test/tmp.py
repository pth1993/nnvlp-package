#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nnvlp


nnvlp.download()
model = nnvlp.NNVLP()
output = model.predict(u'Ông Nam là giảng viên trường Bách Khoa.', display_format='JSON')
print output
