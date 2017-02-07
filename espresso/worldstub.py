# -*- coding: utf-8 -*-

# ****************************************************************************
# Original work Copyright (C) 2013-2015 SUNCAT
# Modified work Copyright 2015-2017 Lukasz Mentel
#
# This file is distributed under the terms of the
# GNU General Public License. See the file 'COPYING'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
# ****************************************************************************


#mpi stub for object having number of nodes as "size" attribute
class world:
    def __init__(self, size):
        self.size = size
