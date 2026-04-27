from typing import Any, Callable, Optional, Sequence, Union
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from acid.image_measurement.measure_glob_image_stat import normalized_intensity_skewness
from acid.image_measurement.measure_local_image_stat import aggregate_local_stat_along_axis
from acid.image_quality_control.measure_mean_over_std import mean_over_std


def measure_local_mean_over_std(
    image: ArrayLike,
    aggregate_func: Callable,
    axis: Optional[int] = None,
    size: Optional[Union[int, Sequence[int]]] = None,
    footprint: Optional[ArrayLike] = None,
    dtype: Optional[DTypeLike] = None,
    agg_kwargs: Optional[dict[str, Any]] = None,
    local_kwargs: dict|None=None,
) -> Union[int, float, NDArray]:
    
    if local_kwargs==None:
        local_kwargs={'axis':None}
    else:
        if 'axis' in local_kwargs:
            assert local_kwargs['axis']==None, "if axis is passed to local kwargs, it must be None"
        else:
            local_kwargs['axis']=None


    return aggregate_local_stat_along_axis(image,
                                           local_func=mean_over_std,
                                           aggregate_func=aggregate_func,
                                           axis=axis,
                                           size=size,
                                           footprint=footprint,
                                           dtype=dtype,
                                           agg_kwargs=agg_kwargs,
                                           **local_kwargs)
