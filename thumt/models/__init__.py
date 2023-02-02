# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.msp
import thumt.models.prompt
import thumt.models.prefix


def get_model(name):
    name = name.lower()

    if name == "xlmr_sga":
        return thumt.models.msp.mXLMR_SGA
    elif name == "xlmr_sga_iterative":
        return thumt.models.msp.mXLMR_SGA_iterative
    elif name == "xlmr_sga_iterative_withdenoiser":
        return thumt.models.msp.mXLMR_SGA_iterative_withdenoiser
    else:
        raise LookupError("Unknown model %s" % name)
