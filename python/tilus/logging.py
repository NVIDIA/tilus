# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

# Name for your application's logger
LOGGER_NAME = "tilus"


def init_logging(level: str = "INFO") -> None:
    """
    Initial logging setup for this application only.

    This should be called once during startup.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)  # Set to the lowest level you want to allow dynamically

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    # If handlers are already attached, don't re-add them
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    set_logging_level(level)


def set_logging_level(level: str) -> None:
    """
    Dynamically change the log level of this application's logger.

    Parameters
    ----------
    level: str
        The new logging level (e.g., "DEBUG", "INFO", etc.)
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    new_level = level_map.get(level.upper(), logging.INFO)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(new_level)

    for handler in logger.handlers:
        handler.setLevel(new_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance. If name is provided, it becomes a child logger.

    Parameters
    ----------
    name: str or None
        Optional name for sub-logger. If None, returns the main app logger.
    """
    if not name.startswith(LOGGER_NAME) and name != "__main__":
        raise RuntimeError(f"Logger name '{name}' must start with '{LOGGER_NAME}'.")
    return logging.getLogger(name)


# Initialize on module import if desired
init_logging("INFO")
