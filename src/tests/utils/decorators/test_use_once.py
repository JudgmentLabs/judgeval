from judgeval.utils.decorators.use_once import use_once


def test_body_runs_once():
    call_count = []

    @use_once
    def fn():
        call_count.append(1)
        return "result"

    fn()
    fn()
    fn()
    assert len(call_count) == 1


def test_returns_first_call_result():
    counter = [0]

    @use_once
    def fn():
        counter[0] += 1
        return counter[0]

    assert fn() == 1
    assert fn() == 1
    assert fn() == 1


def test_same_args_cached():
    call_count = []

    @use_once
    def fn(x):
        call_count.append(x)
        return x

    fn("key")
    fn("key")
    fn("key")
    assert len(call_count) == 1
