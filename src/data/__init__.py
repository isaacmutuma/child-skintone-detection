from .records import Record, MINOR, ADULT_FPR, AMBIGUOUS, OTHER
from .utkface import load_utkface
from .fairface import load_fairface_val
from .subset import dev_subset

LOADERS = {"utkface": load_utkface, "fairface_val": load_fairface_val}
