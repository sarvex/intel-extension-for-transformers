# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
import unittest
from datasets import Dataset, Audio
from intel_extension_for_transformers.neural_chat.tests.restful.config import HOST, API_AUDIO
from intel_extension_for_transformers.neural_chat.cli.log import logger


class UnitTest(unittest.TestCase):

    def __init__(self, *args):
        super(UnitTest, self).__init__(*args)
        self.host = HOST

    def test_voicechat_text_out(self):
        logger.info(f'Testing POST request: {self.host+API_AUDIO} with text output.')
        audio_path = "../../../assets/audio/pat.wav"

        with open(audio_path, "rb") as wav_file:
            files = {
                "file": ("audio.wav", wav_file, "audio/wav"),
                "voice": (None, "pat"),
                "audio_output_path": (None, " ")
            }
            response = requests.post(self.host+API_AUDIO, files=files)

            logger.info('Response status code: {}'.format(response.status_code))
            logger.info('Response text: {}'.format(response.text))
            self.assertEqual(response.status_code, 200, msg="Abnormal response status code.")

    def test_voicechat_audio_out(self):
        logger.info(f'Testing POST request: {self.host+API_AUDIO} with audio output.')
        audio_path = "../../../assets/audio/pat.wav"

        with open(audio_path, "rb") as wav_file:
            files = {
                "file": ("audio.wav", wav_file, "audio/wav"),
                "voice": (None, "pat"),
                "audio_output_path": (None, "./response.wav")
            }
            response = requests.post(self.host+API_AUDIO, files=files)

            logger.info('Response status code: {}'.format(response.status_code))
            logger.info('Response text: {}'.format(response.text))
            self.assertEqual(response.status_code, 200, msg="Abnormal response status code.")
            

if __name__ == "__main__":
    unittest.main()