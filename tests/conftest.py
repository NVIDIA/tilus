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
from pathlib import Path
import pytest
import tilus
import hidet


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and before performing collection and entering the run test loop.
    """
    # set the cache directory to a subdirectory of the current directory
    tilus.option.cache_dir(Path(tilus.option.get_option('cache_dir')) / '..' / '.test_cache')
    print('Cache directory: {}'.format(hidet.option.get_cache_dir()))

@pytest.fixture(autouse=True)
def clear_before_test():
    """
    Clear the memory cache before each test.
    """
    import gc
    import torch

    torch.cuda.empty_cache()
    if hidet.cuda.available():
        hidet.runtime.storage.current_memory_pool('cuda').clear()
    gc.collect()  # release resources with circular references but are unreachable
    yield
    # run after each test
    pass
