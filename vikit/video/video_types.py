# Copyright 2024 Vikit.ai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from enum import Enum


class VideoType(Enum):
    COMPROOT = 0
    COMPCHILD = 1
    IMPORTED = 2
    RAWTEXT = 3
    TRANSITION = 4
    PRMPTBASD = 5
    RAWIMAGE = 6

    def __str__(self):
        return self.name.lower()
