#!/usr/bin/env python
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

from typing import Optional, Dict

from pydantic import BaseModel, Extra


class RequestBaseModel(BaseModel):
    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid


class RetrievalRequest(RequestBaseModel):
    query: str
    domain: str
    blob: Optional[str]
    filename: Optional[str]
    embedding: Optional[str] = 'dense'
    params: Optional[Dict] = None
    debug: Optional[bool] = False


class Text2ImageRequest(RequestBaseModel):
    prompt: str
    steps: Optional[int] = 25
    seed: Optional[int] = 42
    guidance_scale: Optional[int] = 7.5
    sd_inference_token: Optional[str] = None


class TextRequest(RequestBaseModel):
    text: str


class FinetuneRequest(RequestBaseModel):
    content: str
