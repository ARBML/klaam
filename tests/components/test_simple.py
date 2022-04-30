import random

import pytest


def naive_way_of_sorting(arr):
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
    return arr


@pytest.mark.parametrize(
    "size",
    [
        0,
        1,
        10,
        100,
        1000,
        10000,
    ],
)
def test_random_horizontal_shift(size):
    arr = [random.randint(0, size) for _ in range(size)]

    # naive implementation that I know forsure it works
    expected_sorted_arr = naive_way_of_sorting(arr)

    # the targeted implementation to be compared to
    actual_sorted_arr = sorted(arr)

    # comparing expectation vs. actual value
    assert expected_sorted_arr == actual_sorted_arr
