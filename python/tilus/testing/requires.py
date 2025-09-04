from typing import Callable

import pytest
from tilus.target import Target, get_current_target


def requires(target: Target) -> Callable[[Callable], Callable]:
    """
    Pytest fixture decorator that skips tests if the current GPU doesn't support the required architecture.

    Parameters
    ----------
    target : Target
        The required target architecture. Examples include 'sm_90a', 'sm_80',
    """

    def decorator(test_func):
        try:
            required_target = target
            current_target = get_current_target()
            current_capability = current_target.properties.compute_capability

            if not current_target.supports(required_target):
                return pytest.mark.skip(
                    f"Test requires architecture {required_target}, but current GPU capability is {current_capability}"
                )(test_func)
            return test_func
        except ValueError as e:
            # If we can't parse the architecture string, skip the test
            return pytest.mark.skip(f"Invalid architecture requirement: {e}")(test_func)
        except Exception as e:
            # If we can't determine current capability, skip the test
            return pytest.mark.skip(f"Cannot determine current GPU capability: {e}")(test_func)

    return decorator
