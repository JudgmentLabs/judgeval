# NOTE: This builds once, can be tweaked if we are missing / capturing other unncessary modules
# @link https://docs.python.org/3.13/library/sysconfig.html
import os
import site
import sysconfig


_TRACE_FILEPATH_BLOCKLIST = tuple(
    os.path.realpath(p) + os.sep
    for p in {
        sysconfig.get_paths()["stdlib"],
        sysconfig.get_paths().get("platstdlib", ""),
        *site.getsitepackages(),
        site.getusersitepackages(),
        os.path.dirname(__file__),
        *(
            [os.path.join(os.path.dirname(__file__), "../../judgeval/")]
            if os.environ.get("JUDGMENT_DEV")
            else []
        ),
    }
    if p
)
