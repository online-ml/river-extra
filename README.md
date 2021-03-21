# river-extra

## Context

This package contains additional estimators that have not been put into the main [River](https://github.com/online-ml/river/) package. These estimators still need to be polished and vetted before making the cut to the main package. This reduces the clutter in the main repository. This repository is not (always) a graveyard: an estimator may be moved to the main repository if it is provably "good".

## Code layout

The layout is the same as that in River. Where in River you would do:

```py
from river import cluster
```

Here you would do:

```py
from river_extra import cluster
```

## Documentation

At the moment there isn't any. We encourage you to dive into the codebase.

## License

River is free and open-source software licensed under the [3-clause BSD license](https://github.com/online-ml/river/blob/master/LICENSE). The same is true for this repository.
